/*
    This file is part of SAI, which is a fork of Leela Zero.
    Copyright (C) 2017-2019 Gian-Carlo Pascutto and contributors
    Copyright (C) 2018-2019 SAI Team

    SAI is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    SAI is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with SAI.  If not, see <http://www.gnu.org/licenses/>.

    Additional permission under GNU GPL version 3 section 7

    If you modify this Program, or any covered work, by linking or
    combining it with NVIDIA Corporation's libraries from the
    NVIDIA CUDA Toolkit and/or the NVIDIA CUDA Deep Neural
    Network library and/or the NVIDIA TensorRT inference library
    (or a modified version of those libraries), containing parts covered
    by the terms of the respective license agreement, the licensors of
    this Program grant you additional permission to convey the resulting
    work.
*/

#include "config.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <boost/utility.hpp>
#include <boost/format.hpp>
#include <boost/spirit/home/x3.hpp>
#ifndef USE_BLAS
#include <Eigen/Dense>
#endif

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif
#ifdef USE_MKL
#include <mkl.h>
#endif
#ifdef USE_OPENBLAS
#include <cblas.h>
#endif
#include "zlib.h"

#include "Network.h"
#include "CPUPipe.h"
#ifdef USE_OPENCL
#include "OpenCLScheduler.h"
#include "UCTNode.h"
#endif
#include "FastBoard.h"
#include "FastState.h"
#include "FullBoard.h"
#include "GameState.h"
#include "GTP.h"
#include "NNCache.h"
#include "Random.h"
#include "ThreadPool.h"
#include "Timing.h"
#include "Utils.h"

namespace x3 = boost::spirit::x3;
using namespace Utils;

#ifndef USE_BLAS
// Eigen helpers
template <typename T>
using EigenVectorMap =
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>;
template <typename T>
using ConstEigenVectorMap =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>;
template <typename T>
using ConstEigenMatrixMap =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;
#endif

// Symmetry helper
std::array<std::array<int, NUM_INTERSECTIONS>,
                  Network::NUM_SYMMETRIES> symmetry_nn_idx_table;

float Network::benchmark_time(int centiseconds) {
    const auto cpus = cfg_num_threads;

    ThreadGroup tg(thread_pool);
    std::atomic<int> runcount{0};

    GameState state;
    state.init_game(BOARD_SIZE, cfg_komi, m_value_head_sai);

    // As a sanity run, try one run with self check.
    // Isn't enough to guarantee correctness but better than nothing,
    // plus for large nets self-check takes a while (1~3 eval per second)
    get_output(&state, Ensemble::RANDOM_SYMMETRY, -1, false, true, true);

    const Time start;
    for (auto i = size_t{0}; i < cpus; i++) {
        tg.add_task([this, &runcount, start, centiseconds, state]() {
            while (true) {
                runcount++;
                get_output(&state, Ensemble::RANDOM_SYMMETRY, -1, false);
                const Time end;
                const auto elapsed = Time::timediff_centis(start, end);
                if (elapsed >= centiseconds) {
                    break;
                }
            }
        });
    }
    tg.wait_all();

    const Time end;
    const auto elapsed = Time::timediff_centis(start, end);
    return 100.0f * runcount.load() / elapsed;
}

void Network::benchmark(const GameState* const state, const int iterations) {
    const auto cpus = cfg_num_threads;
    const Time start;

    ThreadGroup tg(thread_pool);
    std::atomic<int> runcount{0};

    for (auto i = size_t{0}; i < cpus; i++) {
        tg.add_task([this, &runcount, iterations, state]() {
            while (runcount < iterations) {
                runcount++;
                get_output(state, Ensemble::RANDOM_SYMMETRY, -1, false);
            }
        });
    }
    tg.wait_all();

    const Time end;
    const auto elapsed = Time::timediff_seconds(start, end);
    myprintf("%5d evaluations in %5.2f seconds -> %d n/s\n",
             runcount.load(), elapsed, int(runcount.load() / elapsed));
}

template<class container>
void process_bn_var(container& weights) {
    constexpr auto epsilon = 1e-5f;
    for (auto&& w : weights) {
        w = 1.0f / std::sqrt(w + epsilon);
    }
}

std::vector<float> Network::winograd_transform_f(const std::vector<float>& f,
                                                 const int outputs,
                                                 const int channels) {
    // F(4x4, 3x3) Winograd filter transformation
    // transpose(G.dot(f).dot(G.transpose()))
    // U matrix is transposed for better memory layout in SGEMM
    auto U = std::vector<float>(WINOGRAD_TILE * outputs * channels);
    const auto G = std::array<float, 3 * WINOGRAD_ALPHA>
                    { 1.0f,        0.0f,      0.0f,
                      -2.0f/3.0f, -SQ2/3.0f, -1.0f/3.0f,
                      -2.0f/3.0f,  SQ2/3.0f, -1.0f/3.0f,
                      1.0f/6.0f,   SQ2/6.0f,  1.0f/3.0f,
                      1.0f/6.0f,  -SQ2/6.0f,  1.0f/3.0f,
                      0.0f,        0.0f,      1.0f};

    auto temp = std::array<float, 3 * WINOGRAD_ALPHA>{};

    constexpr auto max_buffersize = 8;
    auto buffersize = max_buffersize;

    if (outputs % buffersize != 0) {
        buffersize = 1;
    }

    std::array<float, max_buffersize * WINOGRAD_ALPHA * WINOGRAD_ALPHA> buffer;

    for (auto c = 0; c < channels; c++) {
        for (auto o_b = 0; o_b < outputs/buffersize; o_b++) {
            for (auto bufferline = 0; bufferline < buffersize; bufferline++) {
                const auto o = o_b * buffersize + bufferline;

                for (auto i = 0; i < WINOGRAD_ALPHA; i++) {
                    for (auto j = 0; j < 3; j++) {
                        auto acc = 0.0f;
                        for (auto k = 0; k < 3; k++) {
                            acc += G[i*3 + k] * f[o*channels*9 + c*9 + k*3 + j];
                        }
                        temp[i*3 + j] = acc;
                    }
                }

                for (auto xi = 0; xi < WINOGRAD_ALPHA; xi++) {
                    for (auto nu = 0; nu < WINOGRAD_ALPHA; nu++) {
                        auto acc = 0.0f;
                        for (auto k = 0; k < 3; k++) {
                            acc += temp[xi*3 + k] * G[nu*3 + k];
                        }
                        buffer[(xi * WINOGRAD_ALPHA + nu) * buffersize + bufferline] = acc;
                    }
                }
            }
            for (auto i = 0; i < WINOGRAD_ALPHA * WINOGRAD_ALPHA; i++) {
                for (auto entry = 0; entry < buffersize; entry++) {
                    const auto o = o_b * buffersize + entry;
                    U[i * outputs * channels
                      + c * outputs
                      + o] =
                    buffer[buffersize * i + entry];
                }
            }
        }
    }

    return U;
}

void Network::add_zero_channels() {
    assert(m_val_outputs < 8);

    const auto old_channels = m_val_outputs;
    m_val_outputs = 8;

    m_fwd_weights->m_conv_val_w.resize(m_channels*m_val_outputs, 0.0f);
    m_fwd_weights->m_conv_val_b.resize(m_val_outputs, 0.0f);
    m_fwd_weights->m_bn_val_w1.resize(m_val_outputs, 0.0f);
    m_fwd_weights->m_bn_val_w2.resize(m_val_outputs, 0.0f);

    m_fwd_weights->m_conv_val_pool_w.resize(m_val_outputs*m_val_pool_outputs, 0.0f);
    for (auto i = m_val_pool_outputs - 1 ; ; i--) {
        for (auto j = m_val_outputs ; j > old_channels ; j--) {
            m_fwd_weights->m_conv_val_pool_w[i * m_val_outputs + j - 1] = 0.0f;
        }
        for (auto j = old_channels ; j > 0 ; j--) {
            m_fwd_weights->m_conv_val_pool_w[i * m_val_outputs + j - 1]
                = m_fwd_weights->m_conv_val_pool_w[i * old_channels + j - 1];
        }
        if (i == 0) break;
    }
}


bool Network::read_weights_line(std::istream& wtfile,
                                std::vector<float>& weights) {
    auto line = std::string{};
    weights.clear();

    if (std::getline(wtfile, line)) {
        auto it_line = line.cbegin();
        const auto ok = phrase_parse(it_line, line.cend(),
                                     *x3::float_, x3::space, weights);
        if (ok && it_line == line.cend()) {
            return true;
        }
    }

    return false;
}

bool Network::read_weights_block(std::istream& wtfile,
                                 std::array<std::vector<float>, 4> &layer,
                                 WeightsFileIndex &id) {
    // Reads up to 4 lines of the weights file.  Returns false if
    // there are less than 4 lines in the buffer after that (i.e. if
    // the end of file was reached).  If there is at least 1 line in
    // the buffer, identifies and stores a new layer, leaving in the
    // buffer any excess lines.

    auto buffer_complete = true;
    auto missing_lines = 0;
    assert(id.excess < 4);
    // read a layer of 4 lines
    for (size_t i = 0 ; i < 4 ; i++) {
        if (i < id.excess) {
            // there are leftovers from previous read of 4 lines
            layer[i] = std::move(layer[4 - id.excess + i]);
        } else if (read_weights_line(wtfile, layer[i])) {
            ++id.line;
        } else {
            layer[i].clear();
            buffer_complete = false;
            ++missing_lines;
        }
    }

    if (missing_lines < 4) {
        identify_layer(layer, id);
        set_network_parameters(layer, id);
        store_layer(layer, id);
    }

    return buffer_complete;
}


void Network::identify_layer(std::array<std::vector<float>, 4> &layer, WeightsFileIndex &id) {
    id.previous = id.section;

    switch(id.section) {

    case NONE:
        id.section = INPUT_CONV;
        break;

    case INPUT_CONV:
        id.section = RESCONV_TOWER;
        break;

    case RESCONV_TOWER:
        if (layer[0].size() == m_channels*9*m_channels) {
            id.section = RESCONV_TOWER;
        } else {
            id.section = POL_CONV_TOWER;
        }
        break;

    case POL_CONV_TOWER:
        if (layer[1].size() == layer[3].size()) {
            id.section = POL_CONV_TOWER;
        } else {
            id.section = POL_DENSE;
        }
        break;

    case POL_DENSE:
        id.section = VALUE_CONV;
        break;

    case VALUE_CONV:
        if (layer[0].size() % NUM_INTERSECTIONS != 0) {
            id.section = VALUE_AVGPOOL;
            break;
        }
        // fall through

    case VALUE_AVGPOOL:
    case VALUE_DENSE_TOWER:
        if (layer[1].size() == layer[3].size()) {
            id.section = VALUE_DENSE_TOWER;
        } else {
            id.section = VAL_DENSE_HIDDEN;
        }
        break;

    case VAL_DENSE_HIDDEN:
        id.section = VAL_DENSE_OUT;
        break;

    case VAL_DENSE_OUT:
        if (layer[2].size() > 0) {
            id.section = VBE_DENSE_HIDDEN;
        } else {
            id.section = VBE_DENSE_OUT;
        }
        break;

    case VBE_DENSE_HIDDEN:
        id.section = VBE_DENSE_OUT;
        break;

    default:
        break;
    }
}


void Network::set_network_parameters(std::array<std::vector<float>, 4> &layer, WeightsFileIndex &id) {
    switch(id.section) {

    case INPUT_CONV:
        // second line of weights, holds the biases for the
        // input convolutional layer, hence its size gives
        // the number of channels of subsequent resconv
        // layers

        m_channels = layer[1].size();    
        // we recover the number of input planes
        m_input_planes = layer[0].size()/9/m_channels;

        // if it is even, color of the current player is
        // used, if it is odd, only komi is used
        m_include_color = (0 == m_input_planes % 2);

        // we recover the number of input moves, knowing
        // that for each move there are 2 bitplanes with
        // stones positions and possibly 2 more bitplanes
        // with some advanced features (legal and atari)
        {
            const auto feature_planes = 2 + (m_adv_features ? 2 : 0)
                + (m_chainlibs_features ? CHAIN_LIBERTIES_PLANES : 0)
                + (m_chainsize_features ? CHAIN_SIZE_PLANES : 0);
            m_input_moves = (m_input_planes - (m_include_color ? 2 : 1)) / feature_planes;
            assert(m_input_planes == m_input_moves * feature_planes + (m_include_color ? 2 : 1));

            myprintf("%d input planes, %d input moves\n%d channels...",
                     m_input_planes, m_input_moves, m_channels);
        }

        break;

    case RESCONV_TOWER:
        break;

    case POL_CONV_TOWER:
        if (id.section != id.previous) {
            m_policy_outputs = m_policy_channels = layer[1].size();
            m_residual_blocks = (m_fwd_weights->m_conv_biases.size()-1)/2;
            assert(m_fwd_weights->m_conv_biases.size() == 1 + (2 * m_residual_blocks));

            myprintf(" %d blocks.\n", m_residual_blocks);
        } else {
            m_policy_outputs = layer[1].size();
        }

        break;

    case POL_DENSE:
        m_policy_conv_layers = m_fwd_weights->m_conv_pol_b.size();

        if (m_policy_conv_layers == 1) {
            myprintf("Legacy policy convolution with %d filters.\n", m_policy_outputs);
        } else {
            myprintf("Policy resconv tower with %d channels,", m_policy_channels);
            if (m_policy_channels != m_channels) {
                myprintf(" 1+%d", (m_policy_conv_layers-1) / 2);
            } else {
                myprintf(" %d", m_policy_conv_layers / 2);
            }
            if (m_policy_channels != m_policy_outputs) {
                myprintf("+1");
            }
            myprintf(" blocks and %d filters.\n", m_policy_outputs);
        }
            
        break;

    case VALUE_CONV:
        m_val_outputs = layer[1].size();
        m_val_dense_inputs = NUM_INTERSECTIONS * m_val_outputs;
        break;

    case VALUE_AVGPOOL:
        m_val_dense_inputs = m_val_pool_outputs = layer[1].size();
        myprintf("Value head pooling with %d channels.\n", m_val_pool_outputs);
        break;

    case VALUE_DENSE_TOWER:
        if (id.section != id.previous) {
            m_value_channels = layer[1].size();
        }
        break;

    case VAL_DENSE_HIDDEN:
        m_val_chans = layer[1].size();
        if (m_vh_dense_weights.size()) {
            const auto str_oddlayer = (m_vh_dense_weights.size() % 2) ? "1+" : "";
            myprintf("Value head residual tower with %d channels and %s%d blocks.\n",
                     m_value_channels, str_oddlayer, m_vh_dense_weights.size()/2);
        }
        break;

    case VAL_DENSE_OUT:
        m_value_head_rets = m_val_head_rets = layer[1].size();
        assert (m_value_head_rets == 1 || m_value_head_rets == 2 || m_value_head_rets == 3);
        if (m_value_head_rets == 1) {
            m_value_head_type = SINGLE;
        } else if (m_value_head_rets == 2 || m_value_head_rets == 3) {
            m_value_head_type = DOUBLE_I;
            m_val_head_rets = 1;
            m_vbe_head_rets = m_value_head_rets - 1;
        }
        id.complete = true;
        break;

    case VBE_DENSE_HIDDEN:
        assert (m_val_head_rets == 1);
        m_value_head_type = DOUBLE_Y;
        m_vbe_chans = layer[1].size();

        myprintf("Double value head. Type Y.\n");
        myprintf("Common convolution: %d filters.\n", m_val_outputs);
        myprintf("Alpha head: %d channels. Beta head: %d channels.\n", m_val_chans, m_vbe_chans);
        id.complete = false;
        break;

    case VBE_DENSE_OUT:
        assert (m_val_head_rets == 1);
        m_vbe_head_rets = layer[1].size();
        assert (m_vbe_head_rets == 1 || m_vbe_head_rets == 2);
        m_value_head_rets = m_val_head_rets + m_vbe_head_rets;

        if (m_value_head_type != DOUBLE_Y) {
            m_value_head_type = DOUBLE_T;

            myprintf("Double value head. Type T.\n");
            myprintf("Convolution with %d filters. Dense with %d channels.\n", m_val_outputs, m_val_chans);
        }
        id.complete = true;
        break;

    default:
        break;
    }
    if (m_quartile_encoding && m_vbe_head_rets > 1) {
        myprintf("\nMore than one beta head with quartile encoding is not supported!\n");
        exit(EXIT_FAILURE);
    }
}


void Network::print_network_details() {
    if (m_value_head_type == SINGLE) {
        myprintf("Single value head (LZ).\n");
        myprintf("Convolution with %d filters. Dense with %d channels.\n", m_val_outputs, m_val_chans);
    } else if (m_value_head_type == DOUBLE_I) {
        myprintf("Double value head. Type I.\n");
        myprintf("Convolution with %d filters. Dense with %d channels.\n", m_val_outputs, m_val_chans);
    }
    if (m_vbe_head_rets == 2) {
        myprintf("Beta head with double output.\n");
    }
}


void Network::store_layer(std::array<std::vector<float>, 4> &layer, WeightsFileIndex &id) {
    switch(id.section) {

    case INPUT_CONV:
        assert (layer[0].size() == m_input_planes * 9 * m_channels);
        assert (layer[1].size() == m_channels);
        assert (layer[2].size() == m_channels);
        assert (layer[3].size() == m_channels);
        m_fwd_weights->m_conv_weights.emplace_back(layer[0]);
        m_fwd_weights->m_conv_biases.emplace_back(layer[1]);
        m_fwd_weights->m_batchnorm_means.emplace_back(layer[2]);
        m_fwd_weights->m_batchnorm_stddevs.emplace_back(layer[3]);
        id.excess = 0;
        break;

    case RESCONV_TOWER:
        assert (layer[0].size() == m_channels * 9 * m_channels);
        assert (layer[1].size() == m_channels);
        assert (layer[2].size() == m_channels);
        assert (layer[3].size() == m_channels);
        m_fwd_weights->m_conv_weights.emplace_back(layer[0]);
        m_fwd_weights->m_conv_biases.emplace_back(layer[1]);
        m_fwd_weights->m_batchnorm_means.emplace_back(layer[2]);
        m_fwd_weights->m_batchnorm_stddevs.emplace_back(layer[3]);
        id.excess = 0;
        break;

    case POL_CONV_TOWER:
        if (id.section != id.previous) {
            assert (layer[0].size() == m_channels * m_policy_outputs);
        } else {
            assert (layer[0].size() == m_policy_channels * m_policy_outputs);
        }
        assert (layer[1].size() == m_policy_outputs);
        assert (layer[2].size() == m_policy_outputs);
        assert (layer[3].size() == m_policy_outputs);
        m_fwd_weights->m_conv_pol_w.emplace_back(layer[0]);
        m_fwd_weights->m_conv_pol_b.emplace_back(layer[1]);
        m_fwd_weights->m_bn_pol_w1.emplace_back(layer[2]);
        m_fwd_weights->m_bn_pol_w2.emplace_back(layer[3]);
        id.excess = 0;
        break;

    case POL_DENSE:
        if (layer[1].size() != POTENTIAL_MOVES) {
            const auto netboardsize = std::sqrt(layer[1].size()-1);
            myprintf("\nGiven network is for %.0fx%.0f, but this version "
                     "of SAI was compiled for %dx%d board!\n",
                     netboardsize, netboardsize, BOARD_SIZE, BOARD_SIZE);
            exit(EXIT_FAILURE);
        }
        assert (layer[0].size() == m_policy_outputs * NUM_INTERSECTIONS * POTENTIAL_MOVES);
        assert (layer[1].size() == POTENTIAL_MOVES);
        m_ip_pol_w = std::move(layer[0]);
        m_ip_pol_b = std::move(layer[1]);
        id.excess = 2;
        break;

    case VALUE_CONV:
        assert (layer[0].size() == m_channels * m_val_outputs);
        assert (layer[1].size() == m_val_outputs);
        assert (layer[2].size() == m_val_outputs);
        assert (layer[3].size() == m_val_outputs);
        m_fwd_weights->m_conv_val_w = std::move(layer[0]);
        m_fwd_weights->m_conv_val_b = std::move(layer[1]);
        m_fwd_weights->m_bn_val_w1 = std::move(layer[2]);
        m_fwd_weights->m_bn_val_w2 = std::move(layer[3]);
        id.excess = 0;
        break;

    case VALUE_AVGPOOL:
        assert (layer[0].size() == m_val_outputs * m_val_pool_outputs);
        assert (layer[1].size() == m_val_pool_outputs);
        assert (layer[2].size() == m_val_pool_outputs);
        assert (layer[3].size() == m_val_pool_outputs);
        m_fwd_weights->m_conv_val_pool_w = std::move(layer[0]);
        m_fwd_weights->m_conv_val_pool_b = std::move(layer[1]);
        m_fwd_weights->m_bn_val_pool_w1 = std::move(layer[2]);
        m_fwd_weights->m_bn_val_pool_w2 = std::move(layer[3]);
        if (m_val_outputs < 8) {
            add_zero_channels();
        }
        id.excess = 0;
        break;

    case VALUE_DENSE_TOWER:
        if (id.section != id.previous) {
            assert (layer[0].size() == m_val_dense_inputs * m_value_channels);
        } else {
            assert (layer[0].size() == m_value_channels * m_value_channels);
        }
        assert (layer[1].size() == m_value_channels);
        assert (layer[2].size() == m_value_channels);
        assert (layer[3].size() == m_value_channels);
        m_vh_dense_weights.emplace_back(layer[0]);
        m_vh_dense_biases.emplace_back(layer[1]);
        m_vh_dense_bn_means.emplace_back(layer[2]);
        m_vh_dense_bn_vars.emplace_back(layer[3]);
        id.excess = 0;
        break;

    case VAL_DENSE_HIDDEN:
        if (m_vh_dense_weights.size()) {
            assert (layer[0].size() == m_value_channels * m_val_chans);
        } else {
            assert (layer[0].size() == m_val_dense_inputs * m_val_chans);
        }
        assert (layer[1].size() == m_val_chans);
        m_ip1_val_w = std::move(layer[0]);
        m_ip1_val_b = std::move(layer[1]);
        id.excess = 2;
        break;

    case VAL_DENSE_OUT:
        assert (layer[0].size() == m_val_chans * m_value_head_rets);
        assert (layer[1].size() == m_value_head_rets);
        m_ip2_val_w = std::move(layer[0]);
        m_ip2_val_b = std::move(layer[1]);
        id.excess = 2;
        break;

    case VBE_DENSE_HIDDEN:
        if (m_vh_dense_weights.size()) {
            assert (layer[0].size() == m_value_channels * m_vbe_chans);
        } else {
            assert (layer[0].size() == m_val_dense_inputs * m_vbe_chans);
        }
        assert (layer[1].size() == m_vbe_chans);
        m_ip1_vbe_w = std::move(layer[0]);
        m_ip1_vbe_b = std::move(layer[1]);
        id.excess = 2;
        break;

    case VBE_DENSE_OUT:
        if (m_ip1_vbe_w.size()) {
            assert (layer[0].size() == m_vbe_chans * m_vbe_head_rets);
        } else {
            assert (layer[0].size() == m_val_chans * m_vbe_head_rets);
        }
        assert (layer[1].size() == m_vbe_head_rets);
        m_ip2_vbe_w = std::move(layer[0]);
        m_ip2_vbe_b = std::move(layer[1]);
        id.excess = 2;
        break;

    default:
        break;
    }
}


int Network::load_v1_network(std::istream& wtfile, int format_version) {
    // Count size of the network
    myprintf("Detecting residual layers... v%d\n", format_version);

    std::array<std::vector<float>, 4> layer;
    WeightsFileIndex id;

    while(read_weights_block(wtfile, layer, id));

    if (id.complete) {
        print_network_details();
    } else {
        myprintf("\Error in reading network file at line %d.\n", id.line);
        return 1;
    }

    return 0;
}

int Network::load_network_file(const std::string& filename) {
    // gzopen supports both gz and non-gz files, will decompress
    // or just read directly as needed.
    auto gzhandle = gzopen(filename.c_str(), "rb");
    if (gzhandle == nullptr) {
        myprintf("Could not open weights file: %s\n", filename.c_str());
        return 1;
    }
    // Stream the gz file in to a memory buffer stream.
    auto buffer = std::stringstream{};
    constexpr auto chunkBufferSize = 64 * 1024;
    std::vector<char> chunkBuffer(chunkBufferSize);
    while (true) {
        auto bytesRead = gzread(gzhandle, chunkBuffer.data(), chunkBufferSize);
        if (bytesRead == 0) break;
        if (bytesRead < 0) {
            myprintf("Failed to decompress or read: %s\n", filename.c_str());
            gzclose(gzhandle);
            return 1;
        }
        assert(bytesRead <= chunkBufferSize);
        buffer.write(chunkBuffer.data(), bytesRead);
    }
    gzclose(gzhandle);

    // Read format version
    auto line = std::string{};
    auto format_version = -1;
    if (std::getline(buffer, line)) {
        auto iss = std::stringstream{line};
        // First line is the file format version id
        iss >> format_version;
        m_adv_features = bool(format_version & 16);
        //        m_komi_policy = bool(format_version & 32);
        m_chainlibs_features = bool(format_version & 64);
        m_chainsize_features = bool(format_version & 128);
        m_quartile_encoding = bool(format_version & 256);
        auto extra_bits = format_version - (format_version & 511);
        auto lz_or_elf = format_version & 3;
        if (iss.fail() || (lz_or_elf != 1 && lz_or_elf != 2) || extra_bits != 0) {
            myprintf("Weights file is the wrong version.\n");
            return 1;
        } else {
            myprintf("Version %d weights file", format_version);
            auto open_parenthesis = false;
            const auto plusconj = " + ";
            auto conj = "";
            // Version 2 networks are identical to v1, except
            // that they return the value for black instead of
            // the player to move. This is used by ELF Open Go.
            if (lz_or_elf == 2) {
                myprintf(" (ELF");
                m_value_head_not_stm = true;
                open_parenthesis = true;
                conj = plusconj;
            } else {
                m_value_head_not_stm = false;
            }
            if (format_version != lz_or_elf && !open_parenthesis) {
                    myprintf(" (");
                    open_parenthesis = true;
                }
            if (m_adv_features) {
                myprintf("%sadvanced board features", conj);
                conj = plusconj;
            }
            if (m_chainlibs_features) {
                myprintf("%schain liberties", conj);
                conj = plusconj;
            }
            if (m_chainsize_features) {
                myprintf("%schain size", conj);
                conj = plusconj;
            }
            if (m_quartile_encoding) {
                myprintf("%squartile encoding", conj);
            }
            if (open_parenthesis) {
                    myprintf(")");
            }
            myprintf(".\n");
            return load_v1_network(buffer, format_version);
        }
    }
    return 1;
}

std::unique_ptr<ForwardPipe>&& Network::init_net(int channels,
    std::unique_ptr<ForwardPipe>&& pipe) {

    pipe->initialize(channels);
    pipe->push_weights(WINOGRAD_ALPHA, m_input_planes, channels, m_fwd_weights);

    return std::move(pipe);
}

#ifdef USE_HALF
void Network::select_precision(int channels) {
    if (cfg_precision == precision_t::AUTO) {
        auto score_fp16 = float{-1.0};
        auto score_fp32 = float{-1.0};

        myprintf("Initializing OpenCL (autodetecting precision).\n");

        // Setup fp16 here so that we can see if we can skip autodetect.
        // However, if fp16 sanity check fails we will return a fp32 and pray it works.
        auto fp16_net = std::make_unique<OpenCLScheduler<half_float::half>>();
        if (!fp16_net->needs_autodetect()) {
            try {
                myprintf("OpenCL: using fp16/half or tensor core compute support.\n");
                m_forward = init_net(channels, std::move(fp16_net));
                benchmark_time(1); // a sanity check run
            } catch (...) {
                myprintf("OpenCL: fp16/half or tensor core failed despite driver claiming support.\n");
                myprintf("Falling back to single precision\n");
                m_forward.reset();
                m_forward = init_net(channels,
                    std::make_unique<OpenCLScheduler<float>>());
            }
            return;
        }

        // Start by setting up fp32.
        try {
            m_forward.reset();
            m_forward = init_net(channels,
                std::make_unique<OpenCLScheduler<float>>());
            score_fp32 = benchmark_time(100);
        } catch (...) {
            // empty - if exception thrown just throw away fp32 net
        }

        // Now benchmark fp16.
        try {
            m_forward.reset();
            m_forward = init_net(channels, std::move(fp16_net));
            score_fp16 = benchmark_time(100);
        } catch (...) {
            // empty - if exception thrown just throw away fp16 net
        }

        if (score_fp16 < 0.0f && score_fp32 < 0.0f) {
            myprintf("Both single precision and half precision failed to run.\n");
            throw std::runtime_error("Failed to initialize net.");
        } else if (score_fp16 < 0.0f) {
            myprintf("Using OpenCL single precision (half precision failed to run).\n");
            m_forward.reset();
            m_forward = init_net(channels,
                std::make_unique<OpenCLScheduler<float>>());
        } else if (score_fp32 < 0.0f) {
            myprintf("Using OpenCL half precision (single precision failed to run).\n");
        } else if (score_fp32 * 1.05f > score_fp16) {
            myprintf("Using OpenCL single precision (less than 5%% slower than half).\n");
            m_forward.reset();
            m_forward = init_net(channels,
                std::make_unique<OpenCLScheduler<float>>());
        } else {
            myprintf("Using OpenCL half precision (at least 5%% faster than single).\n");
        }
        return;
    } else if (cfg_precision == precision_t::SINGLE) {
        myprintf("Initializing OpenCL (single precision).\n");
        m_forward = init_net(channels,
            std::make_unique<OpenCLScheduler<float>>());
        return;
    } else if (cfg_precision == precision_t::HALF) {
        myprintf("Initializing OpenCL (half precision).\n");
        m_forward = init_net(channels,
            std::make_unique<OpenCLScheduler<half_float::half>>());
        return;
    }
}
#endif

void Network::initialize(int playouts, const std::string & weightsfile) {
#ifdef USE_BLAS
#ifndef __APPLE__
#ifdef USE_OPENBLAS
    openblas_set_num_threads(1);
    myprintf("BLAS Core: %s\n", openblas_get_corename());
#endif
#ifdef USE_MKL
    //mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL);
    mkl_set_num_threads(1);
    MKLVersion Version;
    mkl_get_version(&Version);
    myprintf("BLAS core: MKL %s\n", Version.Processor);
#endif
#endif
#else
    myprintf("BLAS Core: built-in Eigen %d.%d.%d library.\n",
             EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION);
#endif

    m_fwd_weights = std::make_shared<ForwardPipeWeights>();

    // Make a guess at a good size as long as the user doesn't
    // explicitly set a maximum memory usage.
    if (cfg_use_nncache) {
        m_nncache.set_size_from_playouts(playouts);
    } else {
        m_nncache.resize(10);
    }

    // Prepare symmetry table
    for (auto s = 0; s < NUM_SYMMETRIES; ++s) {
        for (auto v = 0; v < NUM_INTERSECTIONS; ++v) {
            const auto newvtx =
                get_symmetry({v % BOARD_SIZE, v / BOARD_SIZE}, s);
            symmetry_nn_idx_table[s][v] =
                (newvtx.second * BOARD_SIZE) + newvtx.first;
            assert(symmetry_nn_idx_table[s][v] >= 0
                   && symmetry_nn_idx_table[s][v] < NUM_INTERSECTIONS);
        }
    }

    // Load network from file
    if (load_network_file(weightsfile)) {
        exit(EXIT_FAILURE);
    }
    m_value_head_sai = (m_value_head_type != SINGLE);

    auto weight_index = size_t{0};
    // Input convolution
    // Winograd transform convolution weights
    m_fwd_weights->m_conv_weights[weight_index] =
        winograd_transform_f(m_fwd_weights->m_conv_weights[weight_index],
                             m_channels, m_input_planes);
    weight_index++;

    // Residual block convolutions
    for (auto i = size_t{0}; i < m_residual_blocks * 2; i++) {
        m_fwd_weights->m_conv_weights[weight_index] =
            winograd_transform_f(m_fwd_weights->m_conv_weights[weight_index],
                                 m_channels, m_channels);
        weight_index++;
    }

    // Biases are not calculated and are typically zero but some networks might
    // still have non-zero biases.
    // Move biases to batchnorm means to make the output match without having
    // to separately add the biases.
    auto bias_size = m_fwd_weights->m_conv_biases.size();
    for (auto i = size_t{0}; i < bias_size; i++) {
        auto means_size = m_fwd_weights->m_batchnorm_means[i].size();
        for (auto j = size_t{0}; j < means_size; j++) {
            m_fwd_weights->m_batchnorm_means[i][j] -= m_fwd_weights->m_conv_biases[i][j];
            m_fwd_weights->m_conv_biases[i][j] = 0.0f;
        }
        process_bn_var(m_fwd_weights->m_batchnorm_stddevs[i]);
    }

    for (auto i = size_t{0}; i < m_fwd_weights->m_bn_val_w1.size(); i++) {
        m_fwd_weights->m_bn_val_w1[i] -= m_fwd_weights->m_conv_val_b[i];
        m_fwd_weights->m_conv_val_b[i] = 0.0f;
    }
    process_bn_var(m_fwd_weights->m_bn_val_w2);

    for (auto i = size_t{0}; i < m_fwd_weights->m_bn_val_pool_w1.size(); i++) {
        m_fwd_weights->m_bn_val_pool_w1[i] -= m_fwd_weights->m_conv_val_pool_b[i];
        m_fwd_weights->m_conv_val_pool_b[i] = 0.0f;
    }
    process_bn_var(m_fwd_weights->m_bn_val_pool_w2);

    for (auto i = size_t{0}; i < m_fwd_weights->m_conv_pol_b.size(); i++) {
        for (auto j = size_t{0}; j < m_fwd_weights->m_conv_pol_b[i].size(); j++) {
            m_fwd_weights->m_bn_pol_w1[i][j] -= m_fwd_weights->m_conv_pol_b[i][j];
            m_fwd_weights->m_conv_pol_b[i][j] = 0.0f;
        }
        process_bn_var(m_fwd_weights->m_bn_pol_w2[i]);
    }

    for (auto i = size_t{0}; i < m_vh_dense_biases.size(); i++) {
        for (auto j = size_t{0}; j < m_vh_dense_biases[i].size(); j++) {
            m_vh_dense_bn_means[i][j] -= m_vh_dense_biases[i][j];
            m_vh_dense_biases[i][j] = 0.0f;
        }
        process_bn_var(m_vh_dense_bn_vars[i]);
    }

 
#ifdef USE_OPENCL
    if (cfg_cpu_only) {
        myprintf("Initializing CPU-only evaluation.\n");
        m_forward = init_net(m_channels, std::make_unique<CPUPipe>());
    } else {
#ifdef USE_OPENCL_SELFCHECK
        // initialize CPU reference first, so that we can self-check
        // when doing fp16 vs. fp32 detections
        m_forward_cpu = init_net(m_channels, std::make_unique<CPUPipe>());
#endif
#ifdef USE_HALF
        // HALF support is enabled, and we are using the GPU.
        // Select the precision to use at runtime.
        select_precision(m_channels);
#else
        myprintf("Initializing OpenCL (single precision).\n");
        m_forward = init_net(m_channels,
                             std::make_unique<OpenCLScheduler<float>>());
#endif
    }

#else //!USE_OPENCL
    myprintf("Initializing CPU-only evaluation.\n");
    m_forward = init_net(m_channels, std::make_unique<CPUPipe>());
#endif

    // Need to estimate size before clearing up the pipe.
    get_estimated_size();
    m_fwd_weights.reset();
}

template<bool ReLU>
std::vector<float> innerproduct(const std::vector<float>& input,
                                const std::vector<float>& weights,
                                const std::vector<float>& biases) {
    const auto inputs = input.size();
    const auto outputs = biases.size();
    std::vector<float> output(outputs);
    //    myprintf("***ip: %d * %d == %d\n", inputs, outputs, weights.size());
    assert(inputs*outputs == weights.size());
#ifdef USE_BLAS
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                // M     K
                outputs, inputs,
                1.0f, &weights[0], inputs,
                &input[0], 1,
                0.0f, &output[0], 1);
#else
    EigenVectorMap<float> y(output.data(), outputs);
    y.noalias() =
        ConstEigenMatrixMap<float>(weights.data(),
                                   inputs,
                                   outputs).transpose()
        * ConstEigenVectorMap<float>(input.data(), inputs);
#endif
    const auto lambda_ReLU = [](const auto val) { return (val > 0.0f) ?
                                                          val : 0.0f; };
    for (unsigned int o = 0; o < outputs; o++) {
        auto val = biases[o] + output[o];
        if (ReLU) {
            val = lambda_ReLU(val);
        }
        output[o] = val;
    }
    return output;
}

template <size_t spatial_size>
void batchnorm(const size_t channels,
               std::vector<float>& data,
               const float* const means,
               const float* const stddivs,
               const float* const eltwise = nullptr) {
    const auto lambda_ReLU = [](const auto val) { return (val > 0.0f) ?
                                                          val : 0.0f; };
    for (auto c = size_t{0}; c < channels; ++c) {
        const auto mean = means[c];
        const auto scale_stddiv = stddivs[c];
        const auto arr = &data[c * spatial_size];

        if (eltwise == nullptr) {
            // Classical BN
            for (auto b = size_t{0}; b < spatial_size; b++) {
                arr[b] = lambda_ReLU(scale_stddiv * (arr[b] - mean));
            }
        } else {
            // BN + residual add
            const auto res = &eltwise[c * spatial_size];
            for (auto b = size_t{0}; b < spatial_size; b++) {
                arr[b] = lambda_ReLU((scale_stddiv * (arr[b] - mean)) + res[b]);
            }
        }
    }
}

#ifdef USE_OPENCL_SELFCHECK
void Network::compare_net_outputs(const Netresult& data,
                                  const Netresult& ref) {
    // Calculates L2-norm between data and ref.
    constexpr auto max_error = 0.2f;

    auto error = 0.0f;

    for (auto idx = size_t{0}; idx < data.policy.size(); ++idx) {
        const auto diff = data.policy[idx] - ref.policy[idx];
        error += diff * diff;
    }
    const auto diff_pass = data.policy_pass - ref.policy_pass;
    const auto diff_winrate = data.value - ref.value;
    error += diff_pass * diff_pass;
    error += diff_winrate * diff_winrate;

    error = std::sqrt(error);

    if (error > max_error || std::isnan(error)) {
        printf("Error in OpenCL calculation: Update your device's OpenCL drivers "
               "or reduce the amount of games played simultaneously.\n");
        throw std::runtime_error("OpenCL self-check mismatch.");
    }
}
#endif

std::vector<float> softmax(const std::vector<float>& input,
                           const float temperature = 1.0f) {
    auto output = std::vector<float>{};
    output.reserve(input.size());

    const auto alpha = *std::max_element(cbegin(input), cend(input));
    auto denom = 0.0f;

    for (const auto in_val : input) {
        auto val = std::exp((in_val - alpha) / temperature);
        denom += val;
        output.push_back(val);
    }

    for (auto& out : output) {
        out /= denom;
    }

    return output;
}

std::pair<float,float> sigmoid(float alpha, float beta, float bonus, float beta2) {
    if (beta2 < 0) {
        beta2 = beta;
    }
    const double arg = (alpha+bonus > 0 ? beta2 : beta) * (alpha+bonus);
    const auto absarg = std::abs(arg);
    const auto ret = (absarg > 30) ?
        std::exp(-absarg) :
        1.0 / (1.0 + std::exp(absarg));

    return (arg < 0) ?
        std::make_pair(float(ret), float(1.0 - ret)) :
        std::make_pair(float(1.0 - ret), float(ret));
}

bool Network::probe_cache(const GameState* const state,
                          Network::Netresult& result) {
    auto cache_success = m_nncache.lookup(state->board.get_hash(), result);

    // If we are not generating a self-play game, try to find
    // symmetries if we are in the early opening.
    if (!cache_success && !cfg_noise && !cfg_random_cnt
        && state->get_movenum()
           < (state->get_timecontrol().opening_moves(BOARD_SIZE) / 2)) {
        for (auto sym = 0; sym < Network::NUM_SYMMETRIES; ++sym) {
            if (sym == Network::IDENTITY_SYMMETRY) {
                continue;
            }
            const auto hash = state->get_symmetry_hash(sym);
            if (m_nncache.lookup(hash, result)) {
                decltype(result.policy) corrected_policy;
                for (auto idx = size_t{0}; idx < NUM_INTERSECTIONS; ++idx) {
                    const auto sym_idx = symmetry_nn_idx_table[sym][idx];
                    corrected_policy[idx] = result.policy[sym_idx];
                }
                result.policy = std::move(corrected_policy);
                cache_success = true;
                break;
            }
        }
    }

    if (cache_success && result.is_sai) {
        get_sai_winrate(result, state);
    }

    return cache_success;
}


float Network::get_sai_winrate(Network::Netresult& result,
                               const GameState* const state) {
    const auto komi = state->get_komi_adj();
    const auto white = (FastBoard::WHITE == state->get_to_move());
    result.value = sigmoid(result.alpha, result.beta, white ? komi : -komi, result.beta2).first;
    return result.value;
}


Network::Netresult Network::get_output(const GameState* const state,
                                       const Ensemble ensemble,
                                       const int symmetry,
                                       const bool read_cache,
                                       const bool write_cache,
                                       const bool force_selfcheck) {
    Netresult result;
    if (state->board.get_boardsize() != BOARD_SIZE) {
        return result;
    }

    if (read_cache && ensemble != AVERAGE) {
        // See if we already have this in the cache.
        if (probe_cache(state, result)) {
            return result;
        }
    }

    if (ensemble == DIRECT) {
        assert(symmetry >= 0 && symmetry < NUM_SYMMETRIES);
        result = get_output_internal(state, symmetry);
    } else if (ensemble == AVERAGE) {
        assert(symmetry == -1);
        for (auto sym = 0; sym < NUM_SYMMETRIES; ++sym) {
            auto tmpresult = get_output_internal(state, sym);
            result.policy_pass +=
                tmpresult.policy_pass / static_cast<float>(NUM_SYMMETRIES);
            result.value += tmpresult.value / static_cast<float>(NUM_SYMMETRIES);;
            result.alpha += tmpresult.alpha / static_cast<float>(NUM_SYMMETRIES);;
            result.beta += tmpresult.beta / static_cast<float>(NUM_SYMMETRIES);;
            result.beta2 += tmpresult.beta2 / static_cast<float>(NUM_SYMMETRIES);;
            result.is_sai = tmpresult.is_sai;

            for (auto idx = size_t{0}; idx < NUM_INTERSECTIONS; idx++) {
                result.policy[idx] +=
                    tmpresult.policy[idx] / static_cast<float>(NUM_SYMMETRIES);
            }
        }
    } else {
        assert(ensemble == RANDOM_SYMMETRY);
        assert(symmetry == -1);
        const auto rand_sym = Random::get_Rng().randfix<NUM_SYMMETRIES>();
        result = get_output_internal(state, rand_sym);
#ifdef USE_OPENCL_SELFCHECK
        // Both implementations are available, self-check the OpenCL driver by
        // running both with a probability of 1/2000.
        // selfcheck is done here because this is the only place NN
        // evaluation is done on actual gameplay.
        if (m_forward_cpu != nullptr
            && (force_selfcheck || Random::get_Rng().randfix<SELFCHECK_PROBABILITY>() == 0)
        ) {
            auto result_ref = get_output_internal(state, rand_sym, true);
            compare_net_outputs(result, result_ref);
        }
#else
        (void)force_selfcheck;
#endif
    }

    // v2 format (ELF Open Go) returns black value, not stm
    if (m_value_head_not_stm) {
        if (state->board.get_to_move() == FastBoard::WHITE) {
            result.value = 1.0f - result.value;
        }
    }

    if (write_cache) {
        // Insert result into cache.
        // Notice that when ensemble == AVERAGE, the cache is in fact
        // updated with the average result, unless of course it
        // already contained that board state. Don't know if this is
        // wanted.
        m_nncache.insert(state->board.get_hash(), result);
    }

    return result;
}

// void Network::dump_array(std::string name, std::vector<float> &array) {
//     auto maxvalue = array[0];
//     auto minvalue = array[0];
//     for (auto x : array) {
//         maxvalue = std::max(maxvalue, x);
//         minvalue = std::min(minvalue, x);
//     }
//     myprintf("%s, min: %f, max: %f\n", name.c_str(), minvalue, maxvalue);
//     for (int i=0; i<std::min(20, int(array.size())); i++) {
//         myprintf("%.3f\t", array[i]);
//     }
//     myprintf("\n");
// }


void Network::reduce_mean(std::vector<float> &layer, size_t area) {
    const auto channels = layer.size() / area;
    assert (area * channels == layer.size());

    std::vector<float> output(channels);
    for (auto c = size_t{0} ; c < channels ; c++) {
        output[c] = 0.0f;
        for (auto i = size_t{0} ; i < area ; i++) {
            output[c] += layer[area*c + i];
        }
        output[c] /= area;
    }
    layer = std::move(output);
}


Network::Netresult Network::get_output_internal(
    const GameState* const state, const int symmetry, bool selfcheck) {
    assert(symmetry >= 0 && symmetry < NUM_SYMMETRIES);
    constexpr auto width = BOARD_SIZE;
    constexpr auto height = BOARD_SIZE;

    // if the input planes of the loaded network are even, then the
    // color of the current player is encoded in the last two planes
    const auto include_color = (0 == m_input_planes % 2);

    const auto input_data = gather_features(state, symmetry, m_input_moves,
                                            m_adv_features, m_chainlibs_features,
                                            m_chainsize_features, include_color);
    std::vector<float> policy_data(m_policy_outputs * width * height);
    const auto value_outputs = (m_val_pool_outputs > 0) ? m_val_pool_outputs : m_val_outputs;
    std::vector<float> val_data(value_outputs * width * height);

#ifdef USE_OPENCL_SELFCHECK
    if (selfcheck) {
        m_forward_cpu->forward(input_data, policy_data, val_data);
    } else {
        m_forward->forward(input_data, policy_data, val_data);
    }
#else
    m_forward->forward(input_data, policy_data, val_data);
    (void) selfcheck;
#endif

    // Get the moves
    const auto policy_out =
        innerproduct<false>(
            policy_data, m_ip_pol_w, m_ip_pol_b);
    const auto outputs = softmax(policy_out, cfg_softmax_temp);

    // Now get the value
    if (m_val_pool_outputs) {
        reduce_mean(val_data, width * height);
    }
    std::vector<float> res(val_data.size());
    unsigned int parity = 0;
    for (auto i = size_t{0} ; i<m_vh_dense_weights.size() ; i++) {
        if (i == 0 && val_data.size() != m_vh_dense_biases[0].size()) {
            val_data = innerproduct<false>(val_data, m_vh_dense_weights[i],
                                           m_vh_dense_biases[i]);
            batchnorm<1>(m_vh_dense_biases[i].size(), val_data,
                         m_vh_dense_bn_means[i].data(), m_vh_dense_bn_vars[i].data());
            parity = 1;
        } else if (!RESDENSE_IN_VALUE_HEAD || i % 2 == parity) {
            std::swap(val_data, res);
            val_data = innerproduct<false>(res, m_vh_dense_weights[i],
                                           m_vh_dense_biases[i]);
            batchnorm<1>(m_vh_dense_biases[i].size(), val_data,
                         m_vh_dense_bn_means[i].data(), m_vh_dense_bn_vars[i].data());
        } else {
            val_data = innerproduct<false>(val_data, m_vh_dense_weights[i],
                                           m_vh_dense_biases[i]);
            batchnorm<1>(m_vh_dense_biases[i].size(), val_data,
                         m_vh_dense_bn_means[i].data(), m_vh_dense_bn_vars[i].data(),
                         res.data());
        }
        
    }

    const auto val_channels =
        innerproduct<true>(
            val_data, m_ip1_val_w, m_ip1_val_b);
    const auto val_output =
        innerproduct<false>(val_channels, m_ip2_val_w, m_ip2_val_b);

    Netresult result;

    if (m_value_head_type==SINGLE) {
        result.alpha = 2 * val_output[0]; // logits of the winrate for LZ networks
        result.beta = 1.0f; // conventional value
        result.value = sigmoid(result.alpha, 1, 0).first;
        result.is_sai = false;
    } else {
        if (m_value_head_type==DOUBLE_Y) {
            const auto vbe_channels =
                innerproduct<true>(val_data, m_ip1_vbe_w, m_ip1_vbe_b);
            const auto vbe_output =
                innerproduct<false>(vbe_channels, m_ip2_vbe_w, m_ip2_vbe_b);

            result.beta = vbe_output[0];
            if (m_vbe_head_rets == 2) {
                result.beta2 = vbe_output[1];
            }
        } else if (m_value_head_type==DOUBLE_T) {
            const auto vbe_output =
                innerproduct<false>(val_channels, m_ip2_vbe_w, m_ip2_vbe_b);
            result.beta = vbe_output[0];
            if (m_vbe_head_rets == 2) {
                result.beta2 = vbe_output[1];
            }
        } else if (m_value_head_type==DOUBLE_I) {
            result.beta = val_output[1];
            if (m_vbe_head_rets == 2) {
                result.beta2 = val_output[2];
            }
        }

        if (!m_quartile_encoding) {
            result.alpha = val_output[0];

            // ln(x) = log2(x) * ln(2)
            const auto beta_nat_tune = cfg_betatune * 0.69314718055994530941723212145818;

            result.beta = std::exp(result.beta + beta_nat_tune) * 10.0 / NUM_INTERSECTIONS;
            if (m_vbe_head_rets == 2) {
                result.beta2 = std::exp(result.beta2 + beta_nat_tune) * 10.0 / NUM_INTERSECTIONS;
            }
        } else {
            assert(m_vbe_head_rets == 1);

            auto q1 = val_output[0];
            const auto q2 = result.beta;
            constexpr auto eps = 0.05;
            constexpr auto log3 = 1.0986122886681096913952452369225;
            result.alpha = 0.5 * ( q1 + q2 );
            result.beta = 2.0 * log3 / ( eps + std::max(0.0f, q2 - q1) );
        }

        result.is_sai = true;
        get_sai_winrate(result, state);
    }

    for (auto idx = size_t{0}; idx < NUM_INTERSECTIONS; idx++) {
        const auto sym_idx = symmetry_nn_idx_table[symmetry][idx];
        result.policy[sym_idx] = outputs[idx];
    }

    result.policy_pass = outputs[NUM_INTERSECTIONS];

    return result;
}

void Network::show_heatmap(const FastState* const state,
                           const Netresult& result,
                           const bool topmoves, const AgentEval &agent) {
    std::vector<std::string> display_map;
    std::string line;

    float legal_policy = result.policy_pass;
    float illegal_policy = 0.0f;

    std::array<float, NUM_INTERSECTIONS> policies;

    const auto color = state->get_to_move();
    for (unsigned int y = 0; y < BOARD_SIZE ; y++) {
        for (unsigned int x = 0; x < BOARD_SIZE ; x++) {
            const auto vertex = state->board.get_vertex(x, y);
            const auto policy = result.policy[y * BOARD_SIZE + x];
            if (state->is_move_legal(color, vertex)) {
                legal_policy += policy;
                policies[y * BOARD_SIZE + x] = policy;
            } else {
                illegal_policy += policy;
                policies[y * BOARD_SIZE + x] = 0.0f;
            }
        }
    }

    for (unsigned int y = 0; y < BOARD_SIZE ; y++) {
        for (unsigned int x = 0; x < BOARD_SIZE ; x++) {
            const auto clean_policy = int(policies[y * BOARD_SIZE + x] * 1000.0f / legal_policy);
            line += boost::str(boost::format("%3d ") % clean_policy);
        }

        display_map.push_back(line);
        line.clear();
    }

    for (int i = display_map.size() - 1; i >= 0; --i) {
        myprintf("%s\n", display_map[i].c_str());
    }
    const auto pass_policy = int(result.policy_pass * 1000 / legal_policy);
    const auto illegal_millis = int(illegal_policy * 1000);

    myprintf("pass: %d, illegal: %d\n", pass_policy, illegal_millis);
    if (result.is_sai) {
        auto x = agent.quantile_lambda;
        auto y = agent.quantile_mu;
        if (y<x) std::swap(x,y);
        myprintf("alpha: %5.2f    ", result.alpha);
        if (result.beta2 > 0) {
            myprintf("betas: %.2f %.2f ", result.beta, result.beta2);
        } else {
            myprintf("beta: %.2f     ", result.beta);
        }
        myprintf("winrate: %2.1f%%\n", result.value*100);
        myprintf("komi: %2.1f       ", state->get_komi());
        myprintf("handicap: %d    ", state->get_handicap());
        if (result.beta2 > 0) {
            myprintf("  ");
        }
        myprintf("alpkt tree: %3.2f\n", agent.alpkt_tree);
        myprintf("lambda: %.2f    ", agent.lambda);
        myprintf("mu: %.2f       ", agent.mu);
        if (result.beta2 > 0) {
            myprintf("  ");
        }
        myprintf("interval: [%.1f, %.1f]\n", x, y);
    } else {
        myprintf("value: %.1f%%\n", result.value*100);
    }

    if (topmoves) {
        std::vector<Network::PolicyVertexPair> moves;
        for (auto i=0; i < NUM_INTERSECTIONS; i++) {
            const auto x = i % BOARD_SIZE;
            const auto y = i / BOARD_SIZE;
            const auto vertex = state->board.get_vertex(x, y);
            if (state->board.get_state(vertex) == FastBoard::EMPTY) {
                moves.emplace_back(result.policy[i], vertex);
            }
        }
        moves.emplace_back(result.policy_pass, FastBoard::PASS);

        std::stable_sort(rbegin(moves), rend(moves));

        auto cum = 0.0f;
        for (const auto& move : moves) {
            if (cum > 0.85f || move.first < 0.01f) break;
            myprintf("%1.3f (%s)\n",
                    move.first,
                    state->board.move_to_text(move.second).c_str());
            cum += move.first;
        }
    }
}

void Network::fill_input_plane_pair(const FullBoard& board,
                                    std::vector<float>::iterator black,
                                    std::vector<float>::iterator white,
                                    const int symmetry) {
    for (auto idx = 0; idx < NUM_INTERSECTIONS; idx++) {
        const auto sym_idx = symmetry_nn_idx_table[symmetry][idx];
        const auto x = sym_idx % BOARD_SIZE;
        const auto y = sym_idx / BOARD_SIZE;
        const auto color = board.get_state(x, y);
        if (color == FastBoard::BLACK) {
            black[idx] = float(true);
        } else if (color == FastBoard::WHITE) {
            white[idx] = float(true);
        }
    }
}

void Network::fill_input_plane_advfeat(std::shared_ptr<const KoState> const state,
                                       std::vector<float>::iterator legal,
                                       std::vector<float>::iterator atari,
                                       const int symmetry) {
    for (auto idx = 0; idx < NUM_INTERSECTIONS; idx++) {
        const auto sym_idx = symmetry_nn_idx_table[symmetry][idx];
        const auto x = sym_idx % BOARD_SIZE;
        const auto y = sym_idx / BOARD_SIZE;
        const auto vertex = state->board.get_vertex(x,y);
        const auto tomove = state->get_to_move();
        const auto is_legal = state->is_move_legal(tomove, vertex);
        legal[idx] = !is_legal;
        atari[idx] = is_legal && (1 == state->board.liberties_to_capture(vertex));
    }
}

void Network::fill_input_plane_chainlibsfeat(std::shared_ptr<const KoState> const state,
                                             std::vector<float>::iterator chainlibs,
                                             const int symmetry) {
    for (auto idx = 0; idx < NUM_INTERSECTIONS; idx++) {
        const auto sym_idx = symmetry_nn_idx_table[symmetry][idx];
        const auto x = sym_idx % BOARD_SIZE;
        const auto y = sym_idx / BOARD_SIZE;
        const auto peek = state->board.get_state(x,y);
        const auto is_stone = (peek == FastBoard::BLACK || peek == FastBoard::WHITE);
        const auto vtx = state->board.get_vertex(x,y);
        // if there is no stone, then put 0 in all planes
        // if there is a stone, then put 1 if its chain has only 1 liberty,
        //                               1 if its chain has <= 2 liberies,
        //                               1 if its chain has <= 3 liberies,
        //                               1 if its chain has <= 4 liberies
        for (auto plane = size_t{0} ; plane < CHAIN_LIBERTIES_PLANES ; plane++) {
            chainlibs[idx + plane * NUM_INTERSECTIONS] = is_stone &&
                (state->board.chain_liberties(vtx) <= plane + 1);
        }
    }
}

void Network::fill_input_plane_chainsizefeat(std::shared_ptr<const KoState> const state,
                                             std::vector<float>::iterator chainsize,
                                             const int symmetry) {
    for (auto idx = 0; idx < NUM_INTERSECTIONS; idx++) {
        const auto sym_idx = symmetry_nn_idx_table[symmetry][idx];
        const auto x = sym_idx % BOARD_SIZE;
        const auto y = sym_idx / BOARD_SIZE;
        const auto peek = state->board.get_state(x,y);
        const auto is_stone = (peek == FastBoard::BLACK || peek == FastBoard::WHITE);
        const auto vtx = state->board.get_vertex(x,y);
        // if there is no stone, then put 0 in all planes
        // if there is a stone, then put 1 if its chain has >= 2 stones,
        //                               1 if its chain has >= 4 stones,
        //                               1 if its chain has >= 6 stones,
        //                               1 if its chain has >= 8 stones
        for (auto plane = size_t{0} ; plane < CHAIN_SIZE_PLANES ; plane++) {
            chainsize[idx + plane * NUM_INTERSECTIONS] = is_stone &&
                (state->board.chain_stones(vtx) >= 2 * plane + 2);
        }
    }
}

std::vector<float> Network::gather_features(const GameState* const state,
                                            const int symmetry,
                                            const int input_moves,
                                            const bool adv_features,
                                            const bool chainlibs_features,
                                            const bool chainsize_features,
                                            const bool include_color) {
    assert(symmetry >= 0 && symmetry < NUM_SYMMETRIES);

    // if advanced board features are included, for every input move
    // in addition to 2 planes with the stones there are 2 planes with
    // legal moves for current player and "atari" intersections for
    // either player
    // if chain liberties feature is included, there are 4 additional
    // planes with the number of liberties of the chain to which this
    // stone bolongs; encoding is ==1, <=2, <=3, <=4
    // if chain size feature is included, there are 4 additional
    // planes with the number of stones of the chain to which this
    // stone bolongs; encoding is >=2, >=4, >=6, >=8
    auto moves_planes = input_moves * (2 +
                                       (adv_features ? 2 : 0) +
                                       (chainlibs_features ? CHAIN_LIBERTIES_PLANES : 0) +
                                       (chainsize_features ? CHAIN_SIZE_PLANES : 0));
    const auto plane_block = input_moves * NUM_INTERSECTIONS;

    // if the color of the current player is included, two more input
    // planes are needed, otherwise one input plane filled with ones
    // will provide information on the border of the board for the CNN
    const auto input_planes = moves_planes + (include_color ? 2 : 1);

    auto input_data = std::vector<float>(input_planes * NUM_INTERSECTIONS);

    const auto current_it = begin(input_data);
    const auto opponent_it = current_it + plane_block;
    const auto legal_it = opponent_it + (adv_features ? plane_block : 0);
    const auto atari_it = legal_it + (adv_features ? plane_block : 0);
    const auto chainlibs_it = atari_it + (chainlibs_features ? plane_block : 0);
    const auto chainsize_it = chainlibs_it + (chainsize_features ? plane_block : 0) +
        (chainlibs_features ? (CHAIN_LIBERTIES_PLANES-1) * plane_block : 0);

    const auto to_move = state->get_to_move();
    const auto blacks_move = to_move == FastBoard::BLACK;
    const auto black_it = blacks_move ? current_it : opponent_it;
    const auto white_it = blacks_move ? opponent_it : current_it;

    // we fill one plane with ones: this is the only one remaining
    // when the color of current player is not included, otherwise it
    // is one of the two last plane, depending on current player
    const auto onesfilled_it =  blacks_move || !include_color ?
        begin(input_data) + moves_planes * NUM_INTERSECTIONS :
        begin(input_data) + (moves_planes + 1) * NUM_INTERSECTIONS;
    std::fill(onesfilled_it, onesfilled_it + NUM_INTERSECTIONS, float(true));

    const auto moves = std::min<size_t>(state->get_movenum() + 1, input_moves);
    // Go back in time, fill history boards
    for (auto h = size_t{0}; h < moves; h++) {
        // collect white, black occupation planes
        fill_input_plane_pair(state->get_past_state(h)->board,
                              black_it + h * NUM_INTERSECTIONS,
                              white_it + h * NUM_INTERSECTIONS,
                              symmetry);
        if (adv_features) {
            fill_input_plane_advfeat(state->get_past_state(h),
                                     legal_it + h * NUM_INTERSECTIONS,
                                     atari_it + h * NUM_INTERSECTIONS,
                                     symmetry);
        }
        if (chainlibs_features) {
            fill_input_plane_chainlibsfeat(state->get_past_state(h),
                                           chainlibs_it + h * NUM_INTERSECTIONS,
                                           symmetry);
        }
        if (chainsize_features) {
            fill_input_plane_chainsizefeat(state->get_past_state(h),
                                           chainsize_it + h * NUM_INTERSECTIONS,
                                           symmetry);
        }
    }

    return input_data;
}

std::pair<int, int> Network::get_symmetry(const std::pair<int, int>& vertex,
                                          const int symmetry,
                                          const int board_size) {
    auto x = vertex.first;
    auto y = vertex.second;
    assert(x >= 0 && x < board_size);
    assert(y >= 0 && y < board_size);
    assert(symmetry >= 0 && symmetry < NUM_SYMMETRIES);

    if ((symmetry & 4) != 0) {
        std::swap(x, y);
    }

    if ((symmetry & 2) != 0) {
        x = board_size - x - 1;
    }

    if ((symmetry & 1) != 0) {
        y = board_size - y - 1;
    }

    assert(x >= 0 && x < board_size);
    assert(y >= 0 && y < board_size);
    assert(symmetry != IDENTITY_SYMMETRY || vertex == std::make_pair(x, y));
    return {x, y};
}

size_t Network::get_estimated_size() {
    if (estimated_size != 0) {
        return estimated_size;
    }
    auto result = size_t{0};

    const auto lambda_vector_size =  [](const std::vector<std::vector<float>> &v) {
        auto result = size_t{0};
        for (auto it = begin(v); it != end(v); ++it) {
            result += it->size() * sizeof(float);
        }
        return result;
    };

    result += lambda_vector_size(m_fwd_weights->m_conv_weights);
    result += lambda_vector_size(m_fwd_weights->m_conv_biases);
    result += lambda_vector_size(m_fwd_weights->m_batchnorm_means);
    result += lambda_vector_size(m_fwd_weights->m_batchnorm_stddevs);

    result += m_fwd_weights->m_conv_pol_w.size() * sizeof(float);
    result += m_fwd_weights->m_conv_pol_b.size() * sizeof(float);

    // Policy head
    result += m_policy_outputs * sizeof(float); // m_bn_pol_w1
    result += m_policy_outputs * sizeof(float); // m_bn_pol_w2
    result += m_policy_outputs * NUM_INTERSECTIONS
                             * POTENTIAL_MOVES * sizeof(float); //m_ip_pol_w
    result += POTENTIAL_MOVES * sizeof(float); // m_ip_pol_b

    // Value head
    result += m_fwd_weights->m_conv_val_w.size() * sizeof(float);
    result += m_fwd_weights->m_conv_val_b.size() * sizeof(float);
    result += m_fwd_weights->m_conv_val_pool_w.size() * sizeof(float);
    result += m_fwd_weights->m_conv_val_pool_b.size() * sizeof(float);
    result += m_val_outputs * sizeof(float); // m_bn_val_w1
    result += m_val_outputs * sizeof(float); // m_bn_val_w2

    result += m_val_outputs * NUM_INTERSECTIONS
                            * m_val_chans * sizeof(float); // m_ip1_val_w
    result += m_val_chans * sizeof(float);  // m_ip1_val_b

    result += m_val_chans * sizeof(float); // m_ip2_val_w
    result += sizeof(float); // m_ip2_val_b
    return estimated_size = result;
}

size_t Network::get_estimated_cache_size() {
    return m_nncache.get_estimated_size();
}

void Network::nncache_resize(int max_count) {
    return m_nncache.resize(max_count);
}

void Network::nncache_clear() {
    m_nncache.clear();
}

void Network::drain_evals() {
    m_forward->drain();
}

void Network::resume_evals() {
    m_forward->resume();
}
