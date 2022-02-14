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

#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED

#include "config.h"

#include <deque>
#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <fstream>
#include <tuple>

#include "NNCache.h"
#include "FastState.h"
#ifdef USE_OPENCL
#include "OpenCLScheduler.h"
#endif
#include "GameState.h"
#include "ForwardPipe.h"
#ifdef USE_OPENCL
#include "OpenCLScheduler.h"
#endif
#ifdef USE_OPENCL_SELFCHECK
#include "SMP.h"
#endif

// Winograd filter transformation changes 3x3 filters to M + 3 - 1
constexpr auto WINOGRAD_M = 4;
constexpr auto WINOGRAD_ALPHA = WINOGRAD_M + 3 - 1;
constexpr auto WINOGRAD_WTILES = BOARD_SIZE / WINOGRAD_M + (BOARD_SIZE % WINOGRAD_M != 0);
constexpr auto WINOGRAD_TILE = WINOGRAD_ALPHA * WINOGRAD_ALPHA;
constexpr auto WINOGRAD_P = WINOGRAD_WTILES * WINOGRAD_WTILES;
constexpr auto SQ2 = 1.4142135623730951f; // Square root of 2

std::pair<float, float> sigmoid(float alpha, float beta, float bonus, float beta2=-1.0f);

extern std::array<std::array<int, NUM_INTERSECTIONS>, 8>
    symmetry_nn_idx_table;

struct AgentEval {
    float lambda;
    float mu;
    float quantile_lambda;
    float quantile_mu;
    float alpkt_tree;
};

// See drain_evals() / resume_evals() for details.
class NetworkHaltException : public std::exception {};

class Network {
    using ForwardPipeWeights = ForwardPipe::ForwardPipeWeights;

  public:
    static constexpr auto NUM_SYMMETRIES = 8;
    static constexpr auto IDENTITY_SYMMETRY = 0;
    enum Ensemble
    {
        DIRECT,
        RANDOM_SYMMETRY,
        AVERAGE
    };
    enum WeightsSection
    {
        NONE,
        INPUT_CONV,
        RESCONV_TOWER,
        POL_CONV_TOWER,
        POL_DENSE,
        VALUE_CONV,
        VALUE_AVGPOOL,
        VALUE_DENSE_TOWER,
        VAL_DENSE_HIDDEN,
        VAL_DENSE_OUT,
        VBE_DENSE_HIDDEN,
        VBE_DENSE_OUT
    };
    struct WeightsFileIndex {
        enum WeightsSection section = NONE;
        enum WeightsSection previous = NONE;
        size_t line = size_t{0};
        size_t excess = size_t{0}; // excess lines previously read
        bool complete = false;
    };

    using PolicyVertexPair = std::pair<float, int>;
    using Netresult = NNCache::Netresult;

    Netresult get_output(const GameState *const state,
                         const Ensemble ensemble,
                         const int symmetry = -1,
                         const bool read_cache = true,
                         const bool write_cache = true,
                         const bool force_selfcheck = false);

    static constexpr unsigned short int SINGLE = 1;
    static constexpr unsigned short int DOUBLE_V = 2;
    static constexpr unsigned short int DOUBLE_Y = 3;
    static constexpr unsigned short int DOUBLE_T = 4;
    static constexpr unsigned short int DOUBLE_I = 5;
    static constexpr unsigned int DEFAULT_INPUT_MOVES = 8;
    static constexpr unsigned int REDUCED_INPUT_MOVES = 4;
    static constexpr unsigned int MINIMIZED_INPUT_MOVES = 1;
    static constexpr unsigned int DEFAULT_ADV_FEATURES = 0;
    static constexpr unsigned int CHAIN_LIBERTIES_PLANES = 4; // must be even
    static constexpr unsigned int CHAIN_SIZE_PLANES = 4; // must be even
    static constexpr auto DEFAULT_COLOR_INPUT_PLANES = (2 + DEFAULT_ADV_FEATURES) * DEFAULT_INPUT_MOVES + 2;

    void initialize(int playouts, const std::string &weightsfile);

    float benchmark_time(int centiseconds);
    void benchmark(const GameState *const state,
                   const int iterations = 1600);
    static void show_heatmap(const FastState *const state,
                             const Netresult &netres, const bool topmoves,
                             const AgentEval &agent);
    static std::vector<float> gather_features(const GameState *const state,
                                              const int symmetry,
                                              const int input_moves = DEFAULT_INPUT_MOVES,
                                              const bool adv_features = false,
                                              const bool chainlibs_features = false,
                                              const bool chainsize_features = false,
                                              const bool include_color = false);
    static std::pair<int, int> get_symmetry(const std::pair<int, int> &vertex,
                                            const int symmetry,
                                            const int board_size = BOARD_SIZE);

    size_t get_estimated_size();
    size_t get_estimated_cache_size();
    void nncache_resize(int max_count);
    void nncache_clear();

    int m_value_head_type = 0;
    bool m_value_head_sai = false;
    size_t m_residual_blocks = size_t{0};
    size_t m_channels = size_t{0};
    size_t m_input_moves = size_t{DEFAULT_INPUT_MOVES};
    size_t m_input_planes = size_t{DEFAULT_COLOR_INPUT_PLANES};
    bool m_adv_features = false;
    bool m_chainlibs_features = false;
    bool m_chainsize_features = false;
    bool m_quartile_encoding = false;
    bool m_include_color = true;
    size_t m_policy_conv_layers = size_t{0};
    size_t m_policy_channels = size_t{0};
    size_t m_policy_outputs = size_t{0};
    size_t m_value_channels = size_t{0};
    size_t m_val_dense_inputs = size_t{0};
    size_t m_val_outputs = size_t{1};
    size_t m_val_pool_outputs = size_t{0};
    size_t m_val_chans = size_t{0};
    size_t m_vbe_chans = size_t{0};
    size_t m_value_head_rets = size_t{0};
    size_t m_val_head_rets = size_t{0};
    size_t m_vbe_head_rets = size_t{0};

    
    // 'Drain' evaluations.  Threads with an evaluation will throw a NetworkHaltException
    // if possible, or will just proceed and drain ASAP.  New evaluation requests will
    // also result in a NetworkHaltException.
    virtual void drain_evals();

    // Flag the network to be open for business.
    virtual void resume_evals();
    
  private:
    void add_zero_channels();
    bool read_weights_line(std::istream& wtfile, std::vector<float>& weights);
    bool read_weights_block(std::istream& wtfile, std::array<std::vector<float>, 4> &layer,
                            WeightsFileIndex &id);
    void identify_layer(std::array<std::vector<float>, 4> &layer, WeightsFileIndex &id);
    void set_network_parameters(std::array<std::vector<float>, 4> &layer, WeightsFileIndex &id);
    void print_network_details();
    void store_layer(std::array<std::vector<float>, 4> &layer, WeightsFileIndex &id);
    int load_v1_network(std::istream &wtfile, int format_version);
    int load_network_file(const std::string &filename);

    static std::vector<float> winograd_transform_f(const std::vector<float> &f,
                                                   const int outputs, const int channels);
    static std::vector<float> zeropad_U(const std::vector<float> &U,
                                        const int outputs, const int channels,
                                        const int outputs_pad, const int channels_pad);
    static void winograd_transform_in(const std::vector<float> &in,
                                      std::vector<float> &V,
                                      const int C);
    static void winograd_transform_out(const std::vector<float> &M,
                                       std::vector<float> &Y,
                                       const int K);
    static void winograd_convolve3(const int outputs,
                                   const std::vector<float> &input,
                                   const std::vector<float> &U,
                                   std::vector<float> &V,
                                   std::vector<float> &M,
                                   std::vector<float> &output);
    static void winograd_sgemm(const std::vector<float> &U,
                               const std::vector<float> &V,
                               std::vector<float> &M, const int C, const int K);
    void reduce_mean(std::vector<float> &layer, size_t area);
    Netresult get_output_internal(const GameState *const state,
                                  const int symmetry, bool selfcheck = false);
    static void fill_input_plane_pair(const FullBoard &board,
                                      std::vector<float>::iterator black,
                                      std::vector<float>::iterator white,
                                      const int symmetry);
    static void fill_input_plane_advfeat(std::shared_ptr<const KoState> const state,
                                         std::vector<float>::iterator legal,
                                         std::vector<float>::iterator atari,
                                         const int symmetry);
    static void fill_input_plane_chainlibsfeat(std::shared_ptr<const KoState> const state,
                                               std::vector<float>::iterator chainlibs,
                                               const int symmetry);
    static void fill_input_plane_chainsizefeat(std::shared_ptr<const KoState> const state,
                                               std::vector<float>::iterator chainsize,
                                               const int symmetry);

    bool probe_cache(const GameState *const state, Network::Netresult &result);
    float get_sai_winrate(Network::Netresult& result, const GameState* const state);
    std::unique_ptr<ForwardPipe> &&init_net(int channels,
                                            std::unique_ptr<ForwardPipe> &&pipe);
    void dump_array(std::string name, std::vector<float> &array);
#ifdef USE_HALF
    void select_precision(int channels);
#endif
    std::unique_ptr<ForwardPipe> m_forward;
#ifdef USE_OPENCL_SELFCHECK
    void compare_net_outputs(const Netresult &data, const Netresult &ref);
    std::unique_ptr<ForwardPipe> m_forward_cpu;
#endif

    NNCache m_nncache;

    size_t estimated_size{0};

    // Residual tower
    std::shared_ptr<ForwardPipeWeights> m_fwd_weights;

    // Policy head
    // std::vector<float> m_bn_pol_w1; // policy_outputs
    // std::vector<float> m_bn_pol_w2; // policy_outputs

    //    std::vector<float> m_kp1_pol_w; // (board_sq*policy_outputs+1)*komipolicy_chans
    //    std::vector<float> m_kp1_pol_b; // komipolicy_chans

    //    std::vector<float> m_kp2_pol_w; // komipolicy_chans*komipolicy_chans
    //    std::vector<float> m_kp2_pol_b; // board_sq*policy_outputs*(board_sq+1)

    std::vector<float> m_ip_pol_w;  // (board_sq*policy_outputs + komipolicy_chans)*(board_sq+1)
    std::vector<float> m_ip_pol_b;  // board_sq+1

    // Value head alpha (val=Value ALpha)
    // std::vector<float> m_bn_val_w1; // val_outputs
    // std::vector<float> m_bn_val_w2; // val_outputs

    std::vector<std::vector<float>> m_vh_dense_weights;
    std::vector<std::vector<float>> m_vh_dense_biases1;
    std::vector<std::vector<float>> m_vh_dense_biases;
    std::vector<std::vector<float>> m_vh_dense_bn_means;
    std::vector<std::vector<float>> m_vh_dense_bn_vars;

    std::vector<float> m_ip1_val_w; // board_sq*val_outputs*val_chans
    std::vector<float> m_ip1_val_b; // val_chans

    std::vector<float> m_ip2_val_w; // val_chans (*2 in SINGLE head type)
    std::vector<float> m_ip2_val_b; // 1 (2 in SINGLE head type)

    bool m_value_head_not_stm;

    // Value head beta (vbe=Value BEta)
    // std::vector<float> m_bn_vbe_w1; // vbe_outputs
    // std::vector<float> m_bn_vbe_w2; // vbe_outputs

    std::vector<float> m_ip1_vbe_w; // board_sq*vbe_outputs*vbe_chans
    std::vector<float> m_ip1_vbe_b; // vbe_chans

    std::vector<float> m_ip2_vbe_w; // vbe_chans
    std::vector<float> m_ip2_vbe_b; // 1
};
#endif
