/*
    This file is part of SAI, which is a fork of Leela Zero.
    Copyright (C) 2017-2019 Gian-Carlo Pascutto
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

#include <cassert>
#include <cstdio>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "UCTNode.h"
#include "FastBoard.h"
#include "FastState.h"
#include "GTP.h"
#include "GameState.h"
#include "Network.h"
#include "Random.h"
#include "Utils.h"
#include "UCTSearch.h"

using namespace Utils;

UCTNode::UCTNode(int vertex, float policy) : m_move(vertex), m_policy(policy) {
}

bool UCTNode::first_visit() const {
    return m_visits == 0;
}

bool UCTNode::create_children(Network & network,
                              std::atomic<int>& nodecount,
                              GameState& state,
                              float& value,
                              float& alpkt,
                              float& beta,
                              float& beta2,
                              float min_psa_ratio) {

    // no successors in final state
    if (state.get_passes() >= 2) {
        return false;
    }

    // acquire the lock
    if (!acquire_expanding()) {
        return false;
    }

    // can we actually expand?
    if (!expandable(min_psa_ratio)) {
        expand_done();
        return false;
    }

    NNCache::Netresult raw_netlist;
    try {
        raw_netlist = network.get_output(
            &state, Network::Ensemble::RANDOM_SYMMETRY,
            -1, cfg_use_nncache, cfg_use_nncache);
    } catch (NetworkHaltException&) {
        expand_cancel();
        throw;
    }

    // DCNN returns value as side to move
    const auto stm_eval = raw_netlist.value;
    const auto to_move = state.board.get_to_move();
    // our search functions evaluate from black's point of view
    // notice that 'value' is used only for LZ networks
    value = (to_move == FastBoard::BLACK) ? stm_eval : 1.0f - stm_eval;

    if (network.m_value_head_sai) {
        m_net_alpkt = alpkt = state.get_alpkt(raw_netlist.alpha);
        m_net_beta = beta = raw_netlist.beta;
        m_net_beta2 = beta2 = raw_netlist.beta2;
        m_net_pi = value;
    } else {
        const auto alpha = raw_netlist.alpha; // logits of winrate
        m_net_alpkt = alpkt = (to_move == FastBoard::BLACK) ? alpha : -alpha;
        m_net_beta = beta = 1.0f;
        m_net_beta2 = beta2 = 1.0f;
        m_net_pi = value;
    }

    std::vector<int> stabilizer_subgroup;

    for (auto i = 0; i < 8; i++) {
        if(i == 0 || (cfg_exploit_symmetries && state.is_symmetry_invariant(i))) {
            stabilizer_subgroup.emplace_back(i);
        }
    }

    std::vector<Network::PolicyVertexPair> nodelist;
    std::array<bool, NUM_INTERSECTIONS> taken_already{};
    auto unif_law = std::uniform_real_distribution<float>{0.0, 1.0};

    auto legal_sum = 0.0f;
    for (auto i = 0; i < NUM_INTERSECTIONS; i++) {
        const auto vertex = state.board.get_vertex(i);
        if (state.is_move_legal(to_move, vertex) && !taken_already[i]) {
            auto taken_policy = 0.0f;
            auto max_u = 0.0f;
            auto chosen_vertex = vertex;
            for (auto sym : stabilizer_subgroup) {
                const auto j_vertex = state.board.get_sym_move(vertex, sym);
                const auto j = state.board.get_index(j_vertex);
                if (!taken_already[j]) {
                    taken_already[j] = true;
                    taken_policy += raw_netlist.policy[j];

                    auto u = 0.0f;
                    if (cfg_symm_nonrandom) {
                        const auto p = state.board.get_xy(j_vertex);
                        u = p.first + 2.001 * p.second;
                    } else {
                        u = unif_law(Random::get_Rng());
                    }
                    if (u > max_u) {
                        max_u = u;
                        chosen_vertex = j_vertex;
                    }
                }
            }
            const auto warm_policy = std::pow(taken_policy,
                                              1.0f/cfg_policy_temp);
            nodelist.emplace_back(warm_policy, chosen_vertex);
            legal_sum += warm_policy;
        }
    }

    // Always try passes if we're not trying to be clever.
    auto allow_pass = cfg_dumbpass;

    // Less than 20 available intersections in a 19x19 game.
    if (int(nodelist.size()) <= std::max(5, BOARD_SIZE)) {
        allow_pass = true;
    }

    // If we're clever, only try passing if we're winning on the
    // net score and on the board count.
    if (!allow_pass && stm_eval > 0.8f) {
        const auto relative_score =
            (to_move == FastBoard::BLACK ? 1 : -1) * state.final_score();
        if (relative_score >= 0) {
            allow_pass = true;
        }
    }

    if (allow_pass) {
        const auto warm_pass_policy = std::pow(raw_netlist.policy_pass,
                                           1.0f/cfg_policy_temp);
        nodelist.emplace_back(warm_pass_policy, FastBoard::PASS);
        legal_sum += warm_pass_policy;
    }

    if (legal_sum > std::numeric_limits<float>::min()) {
        // re-normalize after removing illegal moves.
        for (auto& node : nodelist) {
            node.first /= legal_sum;
        }
    } else {
        // This can happen with new randomized nets.
        auto uniform_prob = 1.0f / nodelist.size();
        for (auto& node : nodelist) {
            node.first = uniform_prob;
        }
    }

    link_nodelist(nodecount, nodelist, min_psa_ratio);
    // Increment visit and assign eval.
    const auto result = SearchResult::from_eval(value, alpkt, beta, beta2,
                                                network.m_value_head_sai);
    update(result);
    if (network.m_value_head_sai) {
        set_lambda_mu(state);
        update_all_quantiles(alpkt, beta, beta2);
    }
    expand_done();
    return true;
}

void UCTNode::link_nodelist(std::atomic<int>& nodecount,
                            std::vector<Network::PolicyVertexPair>& nodelist,
                            float min_psa_ratio) {
    assert(min_psa_ratio < m_min_psa_ratio_children);

    if (nodelist.empty()) {
        return;
    }

    // Use best to worst order, so highest go first
    std::stable_sort(rbegin(nodelist), rend(nodelist));

    const auto max_psa = nodelist[0].first;
    const auto old_min_psa = max_psa * m_min_psa_ratio_children;
    const auto new_min_psa = max_psa * min_psa_ratio;
    if (new_min_psa > 0.0f) {
        m_children.reserve(
            std::count_if(cbegin(nodelist), cend(nodelist),
                [=](const auto& node) { return node.first >= new_min_psa; }
            )
        );
    } else {
        m_children.reserve(nodelist.size());
    }

    auto skipped_children = false;
    for (const auto& node : nodelist) {
        if (node.first < new_min_psa) {
            skipped_children = true;
        } else if (node.first < old_min_psa) {
            m_children.emplace_back(node.second, node.first);
            ++nodecount;
        }
    }

    m_min_psa_ratio_children = skipped_children ? min_psa_ratio : 0.0f;
}

const std::vector<UCTNodePointer>& UCTNode::get_children() const {
    return m_children;
}


int UCTNode::get_move() const {
    return m_move;
}

void UCTNode::virtual_loss() {
    m_virtual_loss += VIRTUAL_LOSS_COUNT;
}

void UCTNode::virtual_loss_undo() {
    m_virtual_loss -= VIRTUAL_LOSS_COUNT;
}

float UCTNode::update(const SearchResult &result, bool forced) {
    const auto eval = result.is_sai_head() ?
        result.eval_with_bonus(get_father_quantile_lambda(),
                               get_father_quantile_mu()) :
        result.eval();

    // Cache values to avoid race conditions.
    auto old_eval = static_cast<float>(m_blackevals);
    auto old_visits = static_cast<int>(m_visits);
    auto old_delta = old_visits > 0 ? eval - old_eval / old_visits : 0.0f;
    m_visits++;
    accumulate_eval(eval);
    auto new_delta = eval - (old_eval + eval) / (old_visits + 1);
    // Welford's online algorithm for calculating variance.
    auto delta = old_delta * new_delta;
    atomic_add(m_squared_eval_diff, delta);
    if (forced) {
        m_forced++;
    }
    atomic_add(m_pi_sum, result.eval());
    return eval;
}

void UCTNode::update_gxx_sums(std::atomic<float> &old_gxgp_sum,
                              std::atomic<float> &old_gp_sum,
                              float old_quantile, float new_alpkt,
                              float new_beta, float new_beta2) {
    const auto g_func = sigmoid(new_alpkt, new_beta, old_quantile, new_beta2);
    const auto right_beta = (new_beta2 > 0 && new_alpkt + old_quantile > 0) ? new_beta2 : new_beta;
    const auto gp_term = right_beta * g_func.first * g_func.second;
    const auto gxgp_term = g_func.first - old_quantile * gp_term;
    atomic_add(old_gxgp_sum, gxgp_term);
    atomic_add(old_gp_sum, gp_term);
}

void UCTNode::update_quantile(std::atomic<float> &old_quantile,
                              float old_gxgp_sum, float old_gp_sum,
                              float parameter, int new_visits, float avg_pi,
                              float new_alpkt, float new_beta, float new_beta2) {
    if (std::abs(parameter) < 1e-5) {
        old_quantile = 0.0f;
        return;
    }
    if (new_visits <= 0) return; // should never happen
    const auto avg_p = 0.5f * parameter + (1.0f - parameter) * avg_pi;

    // Sometimes this function is not called when visits==0 so be
    // flexible and set the first value also in those cases.
    if (new_visits <= 8 && old_quantile == 0.0f) {
        // No numerical issues here, as avg_p is away from 0 and 1
        const auto right_beta = (new_beta2 > 0 && avg_p > 0.5) ? new_beta2 : new_beta;
        old_quantile = (std::log(avg_p) - std::log1p(-avg_p)) / std::max(0.01f, right_beta) - new_alpkt;
    } else {
        const auto avg_f_prime = old_gp_sum / float(new_visits);
        const auto avg_f = old_gxgp_sum / float(new_visits)
            + static_cast<float>(old_quantile) * avg_f_prime;
        const auto delta = (avg_p - avg_f) / std::max(0.1f, avg_f_prime);
        atomic_add(old_quantile, delta);
    }
}

void UCTNode::update_all_quantiles(float new_alpkt, float new_beta, float new_beta2) {
    // Cache values to avoid race conditions.
    const auto avg_pi = get_avg_pi();
    const auto old_q_lambda = static_cast<float>(m_quantile_lambda);
    const auto old_q_mu = static_cast<float>(m_quantile_mu);
    const auto old_q_one = static_cast<float>(m_quantile_one);
    const auto new_visits = static_cast<int>(++m_quantile_updates);
    update_gxx_sums(m_gxgp_sum_lambda, m_gp_sum_lambda, old_q_lambda,
                    new_alpkt, new_beta, new_beta2);
    update_gxx_sums(m_gxgp_sum_mu, m_gp_sum_mu, old_q_mu,
                    new_alpkt, new_beta, new_beta2);
    update_gxx_sums(m_gxgp_sum_one, m_gp_sum_one, old_q_one,
                    new_alpkt, new_beta, new_beta2);
    update_quantile(m_quantile_lambda,
                    static_cast<float>(m_gxgp_sum_lambda),
                    static_cast<float>(m_gp_sum_lambda),
                    get_lambda(), new_visits, avg_pi, new_alpkt, new_beta, new_beta2);
    update_quantile(m_quantile_mu,
                    static_cast<float>(m_gxgp_sum_mu),
                    static_cast<float>(m_gp_sum_mu),
                    get_mu(), new_visits, avg_pi, new_alpkt, new_beta, new_beta2);
    update_quantile(m_quantile_one,
                    static_cast<float>(m_gxgp_sum_one),
                    static_cast<float>(m_gp_sum_one),
                    1, new_visits, avg_pi, new_alpkt, new_beta, new_beta2);
}

bool UCTNode::has_children() const {
    return m_min_psa_ratio_children <= 1.0f;
}

bool UCTNode::expandable(const float min_psa_ratio) const {
#ifndef NDEBUG
    if (m_min_psa_ratio_children == 0.0f) {
        // If we figured out that we are fully expandable
        // it is impossible that we stay in INITIAL state.
        assert(m_expand_state.load() != ExpandState::INITIAL);
    }
#endif
    return min_psa_ratio < m_min_psa_ratio_children;
}

float UCTNode::get_policy() const {
    return m_policy;
}

void UCTNode::set_values(float value, float alpkt, float beta, float beta2) {
    m_net_pi = value;
    m_net_alpkt = alpkt;
    m_net_beta = beta;
    m_net_beta2 = beta2;
}

void UCTNode::set_policy(float policy) {
    m_policy = policy;
}

#ifdef USE_EVALCMD
void UCTNode::set_progid(int id) {
    assert(id >= 0);
    m_progid.push_back(id);
}

std::vector<int>& UCTNode::get_progid() {
    return m_progid;
}
#endif

bool UCTNode::low_visits_child(UCTNode* const child) const {
    const auto father_visits = get_visits();
    const auto child_visits = child->get_visits();
    // This formula encodes the following table:
    // father  1-3  child always low
    // father  4-6  child up to 3 low
    // father  7-12 child up to 4 low
    // father 13-20 child up to 5 low
    // father 21-30 child up to 6 low ...
    // If the child visits are high, then the child node is surely
    // good and reliable, otherwise it may be a wrong move that is
    // going to get dropped from tree search.
    return (child_visits * (child_visits - 3) < father_visits - 2);
}

float UCTNode::get_eval_variance(float default_var) const {
    return m_visits > 1 ? m_squared_eval_diff / (m_visits - 1) : default_var;
}

int UCTNode::get_visits() const {
    return m_visits;
}

int UCTNode::get_denom() const {
    if (cfg_laddercode) {
        return 1 + m_visits - m_forced;
    } else {
        return 1 + m_visits;
    }
}

#ifndef NDEBUG
void UCTNode::set_urgency(float urgency,
                          float psa,
                          float q,
                          float den,
                          float num) {
    m_last_urgency = {urgency, psa, q, den, num};
}

std::array<float, 5> UCTNode::get_urgency() const {
    return m_last_urgency;
}
#endif

float UCTNode::get_eval_lcb(int color) const {
    // Lower confidence bound of winrate.
    auto visits = get_visits();
    if (visits < 2) {
        // Return large negative value if not enough visits.
        return -1e6f + visits;
    }
    auto mean = get_raw_eval(color);

    auto stddev = std::sqrt(get_eval_variance(1.0f) / visits);
    auto z = cached_t_quantile(visits - 1);

    return mean - z * stddev;
}

float UCTNode::get_raw_eval(int tomove, int virtual_loss) const {
    auto visits = get_visits() + virtual_loss;
    assert(visits > 0);
    auto blackeval = get_blackevals();
    if (tomove == FastBoard::WHITE) {
        blackeval += static_cast<double>(virtual_loss);
    }
    auto eval = static_cast<float>(blackeval / double(visits));
    if (tomove == FastBoard::WHITE) {
        eval = 1.0f - eval;
    }
    return eval;
}

float UCTNode::get_eval(int tomove) const {
    // Due to the use of atomic updates and virtual losses, it is
    // possible for the visit count to change underneath us. Make sure
    // to return a consistent result to the caller by caching the values.
    return get_raw_eval(tomove, m_virtual_loss);
}

float UCTNode::get_net_pi(int tomove) const {
    if (tomove == FastBoard::WHITE) {
        return 1.0f - m_net_pi;
    }
    return m_net_pi;
}

float UCTNode::get_avg_pi(int tomove) const {
    const auto visits = static_cast<float>(m_visits);
    const auto avg_pi = visits > 0.5f ? static_cast<float>(m_pi_sum) / visits : 0.5f;
    if (tomove == FastBoard::WHITE) {
        return 1.0f - avg_pi;
    }
    return avg_pi;
}

float UCTNode::get_quantile_lambda(int tomove) const {
    if (tomove == FastBoard::WHITE) {
        return -m_quantile_lambda;
    }
    return m_quantile_lambda;
}

float UCTNode::get_quantile_mu(int tomove) const {
    if (tomove == FastBoard::WHITE) {
        return -m_quantile_mu;
    }
    return m_quantile_mu;
}

double UCTNode::get_blackevals() const {
    return m_blackevals;
}

void UCTNode::accumulate_eval(float eval) {
    atomic_add(m_blackevals, double(eval));
}

float UCTNode::get_fpu_eval(int color, bool is_root, size_t &parentvisits) const {
    // fpu reduction is computed on the largest of the children which
    // have already been visited,
    auto total_visited_policy = 0.0f;
    auto max_eval = 0.0f;
    parentvisits = size_t{0};

    // fpu average requires these variables
    auto n = 0;
    auto avg_eval = 0.0f;

    for (const auto& child : m_children) {
        if (child.valid()) {
            if (child.get_visits() > 0) {
                const auto child_eval = child.get()->get_raw_eval(color);
                max_eval = std::max (max_eval, child_eval);
                parentvisits += child.get_visits();
                total_visited_policy += child.get_policy();

                ++n;
                avg_eval += (child_eval - avg_eval) / n;
            }
        }
    }
 
    if (cfg_fpuavg) {
        // We want the average of children except for the best one
        if (n>1) {
            avg_eval -= (max_eval - avg_eval) / (n-1);
        }
        return avg_eval;
    }

    const auto fpu_reduction = (is_root ? cfg_fpu_root_reduction : cfg_fpu_reduction) * std::sqrt(total_visited_policy);
    // Estimated eval for unknown nodes = parent (not NN) eval - reduction
    return cfg_fpuzero ? 0.0f : max_eval - fpu_reduction;
}

float UCTNode::compute_numerator(int visits) {
    return std::sqrt(double(visits) *
                     std::log(cfg_logpuct * double(visits) + cfg_logconst));
}


float UCTNode::get_uct_root(const UCTNode &root, int color) const {
    if (get_visits()) {
        return get_uct_internal(get_raw_eval(color), get_policy()/2, compute_numerator(root.get_visits()));
    } else {
        auto parentvisits = size_t{0};
        const auto fpu_eval = root.get_fpu_eval(color, true, parentvisits);
        return get_uct_internal(fpu_eval, get_policy()/2, compute_numerator(parentvisits));
    }
}


float UCTNode::get_uct_internal(float winrate, float policy, double numerator) const {
    return get_uct_internal(winrate, policy, numerator, get_denom());
}


float UCTNode::get_uct_internal(float winrate, float policy, double numerator, int denom) {
    return winrate + cfg_puct * policy * numerator / double(denom);
}


UCTNode* UCTNode::uct_select_child(const GameState & currstate,
                                   bool is_root,
                                   int max_visits,
                                   const std::vector<int> & move_list,
                                   bool nopass) {
    wait_expanded();
    auto parentvisits = size_t{0};

    const auto color = currstate.get_to_move();
    // Count parentvisits manually to avoid issues with transpositions.
    const auto fpu_eval = get_fpu_eval(color, is_root, parentvisits);

    const auto numerator = compute_numerator(parentvisits);

    auto best = static_cast<UCTNodePointer*>(nullptr);
    auto best_value = std::numeric_limits<double>::lowest();

#ifndef NDEBUG
    auto b_psa = 0.0f;
    auto b_q = 0.0f;
    auto b_denom = 0.0f;
#endif

    for (auto& child : m_children) {
        if (!child.active()) {
            continue;
        }

        if( !move_list.empty() &&
            std::find( begin(move_list), end(move_list),
                       child.get_move() ) == end(move_list) ) {
          continue;
        }

        const auto visits = child.get_visits();

        // If max_visits is specified, then stop choosing nodes that
        // already have enough visits. This guarantees that
        // exploration is wide enough and not too deep when doing fast
        // roll-outs in the endgame exploration.
        if (max_visits > 0 && visits >= max_visits) {
            continue;
        }

        auto winrate = fpu_eval;
        if (child.is_inflated() &&
            child->m_expand_state.load() == ExpandState::EXPANDING) {
            // Someone else is expanding this node, never select it
            // if we can avoid so, because we'd block on it.
            winrate = -1.0f;
        } else if (visits > 0) {
            winrate = child.get_eval(color);
        }
        auto psa = child.get_policy();

        if (nopass && child.get_move() == FastBoard::PASS) {
            psa = 0.0;
            winrate -= 0.05; // is this correct?
        }

        if (currstate.get_passes() >= 1 &&
            child.get_move() == FastBoard::PASS) {
            psa += 0.2;
        }

        if (cfg_stdevuct) {
            const auto stdev = std::sqrt(child.get_eval_variance(0.25f));
            // maximum stdev is 0.5 so double it to get something of
            // order 1; still this term will increase the relative
            // weight of winrate, so also consider increasing cfg_puct
            psa *= 2.0f * stdev;
        }

        auto value = child.get_uct_internal(winrate, psa, numerator);
        assert(value > std::numeric_limits<double>::lowest());

        if (value > best_value) {
            best_value = value;
            best = &child;
#ifndef NDEBUG
            b_psa = psa;
            b_q = winrate;
            b_denom = get_denom();
#endif
        }
    }

    assert(best != nullptr);
    if(best->get_visits() == 0) {
        best->inflate();
        best->get()->set_values(m_net_pi, m_net_alpkt, m_net_beta, m_net_beta2);
    }
#ifndef NDEBUG
    best->get()->set_urgency(best_value, b_psa, b_q,
                             b_denom, numerator);
    // if (best->get()->get_move() == FastBoard::PASS) {
    //   const auto score = ( color == FastBoard::BLACK ? 1.0 : -1.0 ) *
    //             currstate.final_score();
    //   myprintf("\nUCT selected PASS. Passes %d, color %d, score %f, winrate %f, visits %d\n",
    //         currstate.get_passes(),
    //         color,
    //         score,
    //         Utils::winner(score),
    //         best->get()->get_visits());
    //    }
#endif
    return best->get();
}

class NodeComp : public std::binary_function<UCTNodePointer&,
                                             UCTNodePointer&, bool> {
public:
    NodeComp(int color, float lcb_min_visits) : m_color(color),
        m_lcb_min_visits(lcb_min_visits){};

    // WARNING : on very unusual cases this can be called on multithread
    // contexts (e.g., UCTSearch::get_pv()) so beware of race conditions
    bool operator()(const UCTNodePointer& a,
                    const UCTNodePointer& b) {
        auto a_visit = a.get_visits();
        auto b_visit = b.get_visits();

        // Need at least 2 visits for LCB.
        if (m_lcb_min_visits < 2) {
            m_lcb_min_visits = 2;
        }

        // Calculate the lower confidence bound for each node.
        if ((a_visit > m_lcb_min_visits) && (b_visit > m_lcb_min_visits)
            && cfg_uselcb) {
            auto a_lcb = a.get_eval_lcb(m_color);
            auto b_lcb = b.get_eval_lcb(m_color);

            // Sort on lower confidence bounds
            if (a_lcb != b_lcb) {
                return a_lcb < b_lcb;
            }
        }

        // if visits are not same, sort on visits
        if (a_visit != b_visit) {
            return a_visit < b_visit;
        }

        // neither has visits, sort on policy prior
        if (a_visit == 0) {
            return a.get_policy() < b.get_policy();
        }

        // both have same non-zero number of visits
        return a.get_eval(m_color) < b.get_eval(m_color);
    }
private:
    int m_color;
    float m_lcb_min_visits;
};

void UCTNode::sort_children(int color, float lcb_min_visits) {
    std::stable_sort(rbegin(m_children), rend(m_children), NodeComp(color, lcb_min_visits));
}

class NodeCompByPolicy : public std::binary_function<UCTNodePointer&,
                                             UCTNodePointer&, bool> {
public:
    bool operator()(const UCTNodePointer& a,
                    const UCTNodePointer& b) {
        return a.get_policy() < b.get_policy();
    }
};

void UCTNode::sort_children_by_policy() {
    std::stable_sort(rbegin(m_children), rend(m_children), NodeCompByPolicy());
}

UCTNode& UCTNode::get_best_root_child(int color) {
    wait_expanded();

    assert(!m_children.empty());

    auto max_visits = 0;
    for (const auto& node : m_children) {
        max_visits = std::max(max_visits, node.get_visits());
    }

    auto ret = std::max_element(begin(m_children), end(m_children),
                                NodeComp(color, cfg_lcb_min_visit_ratio * max_visits));
    ret->inflate();

    return *(ret->get());
}

size_t UCTNode::count_nodes_and_clear_expand_state() {
    auto nodecount = size_t{0};
    nodecount += m_children.size();
    if (expandable()) {
        m_expand_state = ExpandState::INITIAL;
    }
    for (auto& child : m_children) {
        if (child.is_inflated()) {
            nodecount += child->count_nodes_and_clear_expand_state();
        }
    }
    return nodecount;
}

void UCTNode::invalidate() {
    m_status = INVALID;
}

void UCTNode::set_active(const bool active) {
    if (valid()) {
        m_status = active ? ACTIVE : PRUNED;
    }
}

bool UCTNode::valid() const {
    return m_status != INVALID;
}

bool UCTNode::active() const {
    return m_status == ACTIVE;
}

UCTNode* UCTNode::select_child(int move) {
    auto selected = static_cast<UCTNodePointer*>(nullptr);

    for (auto& child : m_children) {
        if (child.get_move() == move) {
            selected = &child;
            selected->inflate();
            return selected->get();
        }
    }
    return nullptr;
}

void UCTNode::get_subtree_alpkts(std::vector<float> & vector,
                                 int passes,
                                 bool is_tromptaylor_scoring) const {
    auto children_visits = 0;

    // check and correct: 'passes' doesn't do anything here.

    vector.emplace_back(get_net_alpkt());
    for (auto& child : m_children) {
        const auto child_visits = child.get_visits();
        if (child_visits > 0) {
            const auto pass = (child.get_move() == FastBoard::PASS) ? 1 : 0;
            child->get_subtree_alpkts(vector, ++passes * pass,
                                      is_tromptaylor_scoring);
                       children_visits += child_visits;
        }
    }

    const auto missing_nodes = get_visits() - children_visits - 1;
    if (missing_nodes > 0 && is_tromptaylor_scoring) {
        // check: this seems to happen only on second pass node, where
        // get_net_alpkt() would return a meningless value

        const std::vector<float> rep(missing_nodes, get_net_alpkt());
        vector.insert(vector.end(), std::begin(rep), std::end(rep));
    }

    return;
}

float UCTNode::estimate_alpkt(int passes,
                              bool is_tromptaylor_scoring) const {
    std::vector<float> subtree_alpkts;

    get_subtree_alpkts(subtree_alpkts, passes, is_tromptaylor_scoring);

    return Utils::median(subtree_alpkts);
}

void UCTNode::get_subtree_betas(std::vector<float> & vector) const {
    vector.emplace_back(get_net_beta());
    for (auto& child : m_children) {
        if (child.get_visits() > 0) {
            child->get_subtree_betas(vector);
        }
    }
}

float UCTNode::get_beta_median() const {
    std::vector<float> subtree_betas;

    get_subtree_betas(subtree_betas);

    return Utils::median(subtree_betas);
}

void UCTNode::az_sum_recursion(float& sum, size_t& n) const {
    sum += get_net_pi();
    n++;
    for (auto& child : m_children) {
        if (child.get_visits() > 0) {
            child->az_sum_recursion(sum, n);
        }
    }
}

float UCTNode::get_azwinrate_avg() const {
    auto sum = 0.0f;
    auto n = size_t{0};

    az_sum_recursion(sum, n);

    return static_cast<float>(sum / double(n));
}

UCTStats UCTNode::get_uct_stats() const {
    UCTStats stats;

    stats.alpkt_tree = -m_quantile_one;
    stats.beta_median = get_beta_median();
    stats.azwinrate_avg = get_azwinrate_avg();
    return stats;
}

std::tuple<float, float, float> UCTNode::score_stats() const {
    const auto alpkt_for_score = -get_quantile_one();
    const auto beta_for_score = get_net_beta();
    const auto eval_for_score = get_eval();
    return std::make_tuple(alpkt_for_score, beta_for_score, eval_for_score);
}


bool UCTNode::acquire_expanding() {
    auto expected = ExpandState::INITIAL;
    auto newval = ExpandState::EXPANDING;
    return m_expand_state.compare_exchange_strong(expected, newval);
}

void UCTNode::expand_done() {
    auto v = m_expand_state.exchange(ExpandState::EXPANDED);
#ifdef NDEBUG
    (void)v;
#endif
    assert(v == ExpandState::EXPANDING);
}
void UCTNode::expand_cancel() {
    auto v = m_expand_state.exchange(ExpandState::INITIAL);
#ifdef NDEBUG
    (void)v;
#endif
    assert(v == ExpandState::EXPANDING);
}
void UCTNode::wait_expanded() {
    while (m_expand_state.load() == ExpandState::EXPANDING) {}
    auto v = m_expand_state.load();
#ifdef NDEBUG
    (void)v;
#endif
    assert(v == ExpandState::EXPANDED);
}

StateEval UCTNode::state_eval() const {
    StateEval ev(get_visits(), m_net_alpkt, m_net_beta, m_net_pi,
                 m_quantile_lambda, m_quantile_mu,
                 get_eval(), -m_quantile_one);
    return ev;
}

void UCTNode::set_lambda_mu(const GameState &state) {
    auto i = 0;
    if (!state.is_cpu_color()) {
        i = 2;
    }
    if (get_raw_eval(state.get_to_move()) < 0.5f) {
        i++;
    }

    m_lambda = cfg_lambda[i];
    m_mu = cfg_mu[i];
}
