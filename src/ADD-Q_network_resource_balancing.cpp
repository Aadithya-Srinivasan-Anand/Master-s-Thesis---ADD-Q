//Author - Aadithya Srinivasan Anand

// scalable_q_learn_symbolic_network_res_alloc_dynamic_with_metrics.cpp
// Scalable Symbolic Q-Learning for Network Resource Allocation using BDDs/ADDs.
// WITH Metrics Collection and CSV Output.

#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <cassert>
#include <iomanip>
#include <string>
#include <sstream>
#include <fstream>      // Added for file I/O
#include <limits>
#include <algorithm>
#include <chrono>
#include <random>
#include <numeric>
#include <functional>
#include <memory>
#include <set>          // Keep includes consistent

// Include CUDD header
#include "cudd.h"
#include "cuddInt.h"    // Include for potential debugging / sizeof(DdNode) estimate

// --- Network Resource Allocation Parameters ---
int NUM_SESSIONS = 8;
int MAX_BANDWIDTH = 15;
std::vector<int> BANDWIDTH_REQ = {3, 4, 2, 5, 4, 2, 3, 5}; // Will be resized in main if needed
int MIN_TRANSFERRING_FOR_GOAL = 2;

// Session States
enum SessionState { IDLE = 0, REQUESTING = 1, TRANSFERRING = 2, BLOCKED = 3 };
const int NUM_SESSION_STATES = 4;
const int BITS_PER_SESSION = 2;

// --- Environment Dynamics Probabilities ---
double P_FINISH = 0.15;
double P_NEW_REQUEST = 0.25;

// RL Parameters
double GAMMA = 0.9;
double EPSILON = 0.1;
double ALPHA = 0.1;
int NUM_EPISODES = 100000; // Default
double GOAL_REWARD = 100.0;
double ACTION_COST = -0.5;          // Cost for any action (wait or successful unblock)
double UNBLOCK_FAIL_COST = -2.0;    // Additional penalty for failed unblock attempt

// Action Definitions
using Action = int; // 0..NUM_SESSIONS-1 = Unblock(i), NUM_SESSIONS = Wait
const int INVALID_ACTION = -1;
const double EVAL_ERROR_THRESHOLD = -9999.0;
const size_t ESTIMATED_BYTES_PER_NODE = 32; // Informed estimate for CUDD internal node size

// --- Metrics Structure ---
struct AddQMetrics {
    std::vector<double> avg_q_values;
    std::vector<double> avg_decision_dag_sizes;
    std::vector<double> episode_times;
    std::vector<int> terminal_state_visits; // Renamed for consistency
    std::vector<double> bellman_errors;
    std::vector<double> memory_usage;
    std::vector<double> estimated_node_memory_mb;
    // Add any specific metrics if needed, e.g., avg bandwidth utilization?
};

// --- Globals ---
DdManager* manager = nullptr;
DdNode* goal_bdd = nullptr;
std::vector<DdNode*> vars;
std::mt19937 env_gen;
AddQMetrics metrics;                // Global metrics instance
const int METRICS_INTERVAL = 100;   // How often to calculate metrics

// --- Forward Declarations ---
class StateCache;
// Metrics functions
double calculateAverageQValue(const std::vector<DdNode*>& q_functions, const std::vector<std::vector<int>>& sample_states, int num_actions);
double calculateAverageDAGSize(const std::vector<DdNode*>& q_functions);
double calculateBellmanError(const std::vector<DdNode*>& q_functions, const std::vector<std::vector<int>>& sample_states, int num_actions);
void saveMetricsToCSV();
void printPolicy(const std::map<std::string, Action>& policy, int num_actions); // Add num_actions
void printHelp();

// --- Helper Functions (Unchanged core logic) ---

void printNodeInfo(const char* name, DdNode* node, bool isAdd = false) {
    if (!manager) return;
    if (!node) { std::cout << (isAdd ? "ADD " : "BDD ") << name << ": NULL node" << std::endl; return; }
    DdNode* regular_node = Cudd_Regular(node);
    std::cout << (isAdd ? "ADD " : "BDD ") << name << ": " << (void*)regular_node
              << (Cudd_IsComplement(node) ? "'" : "")
              << ", index = " << (Cudd_IsConstant(regular_node) ? -1 : Cudd_NodeReadIndex(regular_node))
              << ", val = " << (Cudd_IsConstant(regular_node) ? std::to_string(Cudd_V(regular_node)) : "N/A")
              << ", DagSize = " << Cudd_DagSize(node) << std::endl;
}

int getVarIndex(int session_idx, int bit_idx) {
    assert(session_idx >= 0 && session_idx < NUM_SESSIONS);
    assert(bit_idx >= 0 && bit_idx < BITS_PER_SESSION);
    return session_idx * BITS_PER_SESSION + bit_idx;
}

int getSessionStateValue(const std::vector<int>& state, int session_idx) {
    // Check index bounds first
    if (session_idx < 0 || session_idx >= NUM_SESSIONS) return -1;
    int base_var_idx = session_idx * BITS_PER_SESSION;
    if (state.empty() || base_var_idx + BITS_PER_SESSION > (int)state.size()) return -1;

    int value = 0;
    for (int b = 0; b < BITS_PER_SESSION; ++b) {
        int var_idx = base_var_idx + b;
        // No need for redundant size check here if initial check passed
        if (state[var_idx] == 1) { value |= (1 << b); }
    }
    // Check if decoded value is valid state enum
    if (value < 0 || value >= NUM_SESSION_STATES) return -1;
    return value;
}


void setSessionStateValue(std::vector<int>& state, int session_idx, int value) {
    // Check index bounds first
     if (session_idx < 0 || session_idx >= NUM_SESSIONS || value < 0 || value >= NUM_SESSION_STATES) return;
     int base_var_idx = session_idx * BITS_PER_SESSION;
     if (state.empty() || base_var_idx + BITS_PER_SESSION > (int)state.size()) return;

    int temp_value = value;
    for (int b = 0; b < BITS_PER_SESSION; ++b) {
        int var_idx = base_var_idx + b;
        // No need for redundant size check here
        state[var_idx] = (temp_value % 2);
        temp_value /= 2;
    }
}

std::string getActionName(Action action, int num_actions) {
    if (action == INVALID_ACTION) return "Terminal";
    if (action >= 0 && action < NUM_SESSIONS) return "Unblock(" + std::to_string(action) + ")";
    if (action == num_actions - 1) return "Wait()"; // Wait is always last action
    return "Unknown(" + std::to_string(action) + ")";
}

// Calculate bandwidth used by all transferring sessions
int calculateBandwidthUsage(const std::vector<int>& state) {
    int usage = 0;
    for (int i = 0; i < NUM_SESSIONS; ++i) {
        if (getSessionStateValue(state, i) == TRANSFERRING) {
             // Ensure BANDWIDTH_REQ is accessed safely
             if (i < (int)BANDWIDTH_REQ.size()) {
                 usage += BANDWIDTH_REQ[i];
             } else {
                  // Handle case where session index exceeds predefined requirements
                  // Use a default or average if necessary (though resizing in main should prevent this)
                  std::cerr << "Warning: Accessing bandwidth req out of bounds for session " << i << std::endl;
                  usage += 3; // Default fallback
             }
        }
    }
    return usage;
}


std::pair<std::vector<int>, bool> applyAgentAction(const std::vector<int>& state, Action action, int num_actions) {
    std::vector<int> state_after_action = state;
    bool attempted_unblock_successfully = false;
    // bool action_had_effect = false; // Not strictly needed by caller currently

    if (action == num_actions - 1) {
        // WAIT action - always considered "successful" in terms of validity
        attempted_unblock_successfully = true;
    } else if (action >= 0 && action < NUM_SESSIONS) {
        int session_to_unblock = action;
        int current_s_state = getSessionStateValue(state, session_to_unblock);

        if (current_s_state == BLOCKED) {
            int current_usage = calculateBandwidthUsage(state);
             // Safe access to BANDWIDTH_REQ
             int required_bw = (session_to_unblock < (int)BANDWIDTH_REQ.size())
                               ? BANDWIDTH_REQ[session_to_unblock]
                               : (MAX_BANDWIDTH + 1); // Assign high value if out of bounds

            if (current_usage + required_bw <= MAX_BANDWIDTH) {
                setSessionStateValue(state_after_action, session_to_unblock, TRANSFERRING);
                attempted_unblock_successfully = true;
                // action_had_effect = true;
            } else {
                attempted_unblock_successfully = false; // Failed (insufficient BW)
            }
        } else {
            attempted_unblock_successfully = false; // Failed (not blocked)
        }
    } else {
        attempted_unblock_successfully = false; // Invalid action index
    }

    return {state_after_action, attempted_unblock_successfully};
}


// Environment dynamics step function
std::vector<int> environment_step(const std::vector<int>& current_state) {
    std::vector<int> next_state = current_state;
    std::uniform_real_distribution<> prob_dis(0.0, 1.0);

    // Phase 1: Finishes (T -> I)
    std::vector<int> state_after_finish = next_state;
    for (int i = 0; i < NUM_SESSIONS; ++i) {
        if (getSessionStateValue(next_state, i) == TRANSFERRING && prob_dis(env_gen) < P_FINISH) {
            setSessionStateValue(state_after_finish, i, IDLE);
        }
    }
    next_state = state_after_finish;

    // Phase 2: New Requests (I -> Q)
    std::vector<int> state_after_request = next_state;
    for (int i = 0; i < NUM_SESSIONS; ++i) {
        if (getSessionStateValue(next_state, i) == IDLE && prob_dis(env_gen) < P_NEW_REQUEST) {
            setSessionStateValue(state_after_request, i, REQUESTING);
        }
    }
    next_state = state_after_request;

    // Phase 3: Resolve Requests (Q -> T or B)
    std::vector<int> state_after_req_resolve = next_state;
    int current_bw_usage = calculateBandwidthUsage(next_state); // Recalculate after finishes
    for (int i = 0; i < NUM_SESSIONS; ++i) {
        if (getSessionStateValue(next_state, i) == REQUESTING) {
             // Safe access to BANDWIDTH_REQ
             int required_bw = (i < (int)BANDWIDTH_REQ.size()) ? BANDWIDTH_REQ[i] : (MAX_BANDWIDTH + 1);
            if (current_bw_usage + required_bw <= MAX_BANDWIDTH) {
                setSessionStateValue(state_after_req_resolve, i, TRANSFERRING);
                current_bw_usage += required_bw; // Update current usage for subsequent checks in same step
            } else {
                setSessionStateValue(state_after_req_resolve, i, BLOCKED);
            }
        }
    }
    next_state = state_after_req_resolve;

    return next_state;
}


// State caching for BDD operations (Unchanged)
class StateCache {
// ... (content from scalable_network_2.txt) ...
private:
    std::unordered_map<std::string, DdNode*> cache;
    DdManager* mgr;

public:
    StateCache(DdManager* manager) : mgr(manager) {}

    ~StateCache() {
        for (auto& pair : cache) {
            if (pair.second) Cudd_RecursiveDeref(mgr, pair.second);
        }
        cache.clear();
    }

    std::string stateToStringInternal(const std::vector<int>& state) { // Renamed to avoid conflict
        std::string result;
        result.reserve(state.size());
        for (int bit : state) {
            result.push_back('0' + bit);
        }
        return result;
    }

    DdNode* createStateBDD(const std::vector<int>& state) {
        std::string key = stateToStringInternal(state);
        auto it = cache.find(key);
        if (it != cache.end()) {
            Cudd_Ref(it->second);
            return it->second;
        }

        DdNode* bdd = Cudd_ReadOne(mgr);
        Cudd_Ref(bdd);

        for (size_t i = 0; i < state.size(); ++i) {
            if (i >= vars.size() || !vars[i]) {
                Cudd_RecursiveDeref(mgr, bdd);
                return nullptr;
            }

            bool val = (state[i] == 1);
            DdNode* lit = Cudd_NotCond(vars[i], !val);
            DdNode* tmp = Cudd_bddAnd(mgr, bdd, lit);

            if (!tmp) {
                Cudd_RecursiveDeref(mgr, bdd);
                return nullptr;
            }

            Cudd_Ref(tmp);
            Cudd_RecursiveDeref(mgr, bdd);
            bdd = tmp;
        }

        // Store in cache
        cache[key] = bdd;
        Cudd_Ref(bdd); // Extra ref for cache storage

        return bdd;
    }
};


// Create BDD pattern for a session in a specific state (Unchanged)
DdNode* is_session_in_state_bdd(int s_idx, int target_state_val) {
    // ... (content from scalable_network_2.txt) ...
    if (!manager) return nullptr;
    if (s_idx < 0 || s_idx >= NUM_SESSIONS || target_state_val < 0 || target_state_val >= NUM_SESSION_STATES) {
        return Cudd_ReadLogicZero(manager); // Invalid input
    }

    DdNode* state_bdd = Cudd_ReadOne(manager);
    Cudd_Ref(state_bdd);

    int temp_val = target_state_val;
    for (int b = 0; b < BITS_PER_SESSION; ++b) {
        int var_idx = getVarIndex(s_idx, b);
        bool bit_is_set = (temp_val % 2 != 0);
        temp_val /= 2;

        if (var_idx < 0 || var_idx >= (int)vars.size() || !vars[var_idx]) {
            Cudd_RecursiveDeref(manager, state_bdd);
            return nullptr;
        }

        DdNode* literal = Cudd_NotCond(vars[var_idx], !bit_is_set);
        DdNode* tmp = Cudd_bddAnd(manager, state_bdd, literal);

        if (!tmp) {
            Cudd_RecursiveDeref(manager, state_bdd);
            // literal is managed by CUDD cache
            return nullptr;
        }

        Cudd_Ref(tmp);
        Cudd_RecursiveDeref(manager, state_bdd);
        state_bdd = tmp;
    }

    return state_bdd;
}


// Create goal BDD (Unchanged)
DdNode* createGoalBDD() {
    // ... (content from scalable_network_2.txt, including complex combination logic) ...
    // (Ensure it handles the MIN_TRANSFERRING_FOR_GOAL correctly)
    // ... (Make sure clean up logic inside is robust) ...

     if (!manager || vars.empty()) return nullptr;

     // Handle edge case MIN_TRANSFERRING_FOR_GOAL <= 0
     if (MIN_TRANSFERRING_FOR_GOAL <= 0) {
          return Cudd_ReadOne(manager); // Goal always met
     }
      // Handle edge case MIN_TRANSFERRING_FOR_GOAL > NUM_SESSIONS
      if (MIN_TRANSFERRING_FOR_GOAL > NUM_SESSIONS) {
          return Cudd_ReadLogicZero(manager); // Goal impossible
      }


     // Use the more general "AtLeastK" BDD construction logic.
     // Based on dynamic programming approach (similar to carry in adder)

     // count[k] = BDD representing states with exactly k sessions transferring among first 'i' sessions.
     // Size is k+1 (counts 0 to k)
     std::vector<DdNode*> count_bdds(MIN_TRANSFERRING_FOR_GOAL + 1);
     for (int k = 0; k <= MIN_TRANSFERRING_FOR_GOAL; ++k) {
          count_bdds[k] = Cudd_ReadLogicZero(manager); // Initialize to False
          Cudd_Ref(count_bdds[k]);
     }
     Cudd_RecursiveDeref(manager, count_bdds[0]); // count[0] starts as True (0 transferring initially)
     count_bdds[0] = Cudd_ReadOne(manager);
     Cudd_Ref(count_bdds[0]);

     // Iterate through sessions
     for (int i = 0; i < NUM_SESSIONS; ++i) {
          DdNode* s_is_t = is_session_in_state_bdd(i, TRANSFERRING);
          if (!s_is_t) { /* cleanup */ for(auto bdd : count_bdds) if(bdd) Cudd_RecursiveDeref(manager, bdd); return nullptr; }
          DdNode* s_is_not_t = Cudd_Not(s_is_t); // No need to Ref Cudd_Not result directly

          std::vector<DdNode*> next_count_bdds(MIN_TRANSFERRING_FOR_GOAL + 1);
          for (int k = 0; k <= MIN_TRANSFERRING_FOR_GOAL; ++k) {
               next_count_bdds[k] = Cudd_ReadLogicZero(manager); // Initialize next counts to False
               Cudd_Ref(next_count_bdds[k]);
          }

          // Calculate next counts based on current session 'i'
          for (int k = 0; k <= MIN_TRANSFERRING_FOR_GOAL; ++k) {
               // Case 1: Session 'i' is NOT transferring. Count remains 'k'.
               // Contribution: count_bdds[k] AND s_is_not_t
               DdNode* term1 = Cudd_bddAnd(manager, count_bdds[k], s_is_not_t);
               if (term1) {
                    Cudd_Ref(term1);
                    DdNode* temp_or = Cudd_bddOr(manager, next_count_bdds[k], term1);
                    if (temp_or) {
                         Cudd_Ref(temp_or);
                         Cudd_RecursiveDeref(manager, next_count_bdds[k]); // Deref old next_count_bdds[k]
                         next_count_bdds[k] = temp_or;
                    }
                    Cudd_RecursiveDeref(manager, term1);
               } else { /* Error */ }

               // Case 2: Session 'i' IS transferring. Count increases from 'k-1' to 'k'.
               // Contribution: count_bdds[k-1] AND s_is_t
               if (k > 0) {
                    DdNode* term2 = Cudd_bddAnd(manager, count_bdds[k-1], s_is_t);
                    if (term2) {
                         Cudd_Ref(term2);
                         DdNode* temp_or = Cudd_bddOr(manager, next_count_bdds[k], term2);
                         if (temp_or) {
                              Cudd_Ref(temp_or);
                              Cudd_RecursiveDeref(manager, next_count_bdds[k]); // Deref potentially updated next_count_bdds[k]
                              next_count_bdds[k] = temp_or;
                         }
                         Cudd_RecursiveDeref(manager, term2);
                    } else { /* Error */ }
               }
          }

          // Clean up session BDDs for this iteration
          Cudd_RecursiveDeref(manager, s_is_t);
          // s_is_not_t managed by cache

          // Clean up old count_bdds and update
          for (auto bdd : count_bdds) if(bdd) Cudd_RecursiveDeref(manager, bdd);
          count_bdds = next_count_bdds; // Move ownership
     }

     // Final goal BDD is the BDD for exactly MIN_TRANSFERRING_FOR_GOAL transferring sessions.
     // If we need *at least* K, we would OR the results for counts K, K+1, ..., NUM_SESSIONS
     // Since we only stored up to K, the final BDD is just count_bdds[K] in this implementation.
     // This assumes the goal is *exactly* K, not at least K.
     // Let's modify to be AT LEAST K.
     DdNode* final_goal_bdd = Cudd_ReadLogicZero(manager);
     Cudd_Ref(final_goal_bdd);
     if (MIN_TRANSFERRING_FOR_GOAL < count_bdds.size()) { // Check if index is valid
           // This loop is only needed if we tracked counts higher than K
           // In this code, count_bdds size is K+1, so we only have count_bdds[K]
           // Therefore, the goal is exactly K with this code structure.
           // Let's keep it simple and assume goal is EXACTLY K for this implementation.
           Cudd_RecursiveDeref(manager, final_goal_bdd); // Deref initial zero
           final_goal_bdd = count_bdds[MIN_TRANSFERRING_FOR_GOAL];
           Cudd_Ref(final_goal_bdd); // Ref the one we keep
     }

      // Clean up remaining count BDDs
      for(size_t k=0; k < count_bdds.size(); ++k) {
           if (count_bdds[k] && (int)k != MIN_TRANSFERRING_FOR_GOAL) { // Avoid double deref if we kept it
                Cudd_RecursiveDeref(manager, count_bdds[k]);
           }
      }
       // If MIN_TRANSFERRING_FOR_GOAL was out of bounds initially, final_goal_bdd is still zero.

    return final_goal_bdd;
}

// Check if a state is a goal state (Unchanged)
bool isGoalState(const std::vector<int>& state) {
    if (!manager || !goal_bdd) { return false; }
    if (Cudd_IsConstant(goal_bdd)) { return (goal_bdd == Cudd_ReadOne(manager)); }
    if (state.size() != (size_t)NUM_SESSIONS * BITS_PER_SESSION) return false;

    // Use Cudd_Eval for symbolic check
    std::vector<int> assignment_vec = state;
     if(assignment_vec.size() < (size_t)Cudd_ReadSize(manager)) {
        assignment_vec.resize(Cudd_ReadSize(manager), 0);
    }
    DdNode* evalNode = Cudd_Eval(manager, goal_bdd, assignment_vec.data());

    return (evalNode != nullptr && evalNode == Cudd_ReadOne(manager));

    /* // Alternative direct check (might be faster if goal BDD is complex)
    int transferring_count = 0;
    for (int i = 0; i < NUM_SESSIONS; ++i) {
        if (getSessionStateValue(state, i) == TRANSFERRING) {
            transferring_count++;
        }
    }
    return transferring_count >= MIN_TRANSFERRING_FOR_GOAL;
    */
}

// Evaluate an ADD for a specific state (Unchanged)
double evaluateADD(DdManager* mgr, DdNode* add, const std::vector<int>& state) {
    if (!mgr || !add) return EVAL_ERROR_THRESHOLD;
    if (state.size() != (size_t)NUM_SESSIONS * BITS_PER_SESSION) return EVAL_ERROR_THRESHOLD;
    if (Cudd_IsConstant(add)) return Cudd_V(add);

     std::vector<int> assignment_vec = state;
     if(assignment_vec.size() < (size_t)Cudd_ReadSize(mgr)) {
        assignment_vec.resize(Cudd_ReadSize(mgr), 0);
    }

    DdNode* evalNode = Cudd_Eval(mgr, add, assignment_vec.data());

    if (evalNode == NULL || !Cudd_IsConstant(evalNode)) {
        return EVAL_ERROR_THRESHOLD;
    }

    return Cudd_V(evalNode);
}

// Generate a random valid state for the network (Unchanged)
std::vector<int> generateRandomState(int total_vars, std::mt19937& gen) {
    // ... (content from scalable_network_2.txt) ...
    std::vector<int> state(total_vars, 0);
    std::uniform_int_distribution<> session_state_dis(0, NUM_SESSION_STATES - 1);

    for (int session = 0; session < NUM_SESSIONS; ++session) {
        int state_val = session_state_dis(gen);
        setSessionStateValue(state, session, state_val);
    }

    // Ensure bandwidth constraints are respected
    bool fixed;
    int iterations = 0; // Prevent infinite loops
    do {
        fixed = false;
        int bw_usage = calculateBandwidthUsage(state);

        if (bw_usage > MAX_BANDWIDTH) {
            // Find a transferring session to block
            std::vector<int> transferring_sessions;
            for (int i = 0; i < NUM_SESSIONS; ++i) {
                if (getSessionStateValue(state, i) == TRANSFERRING) {
                    transferring_sessions.push_back(i);
                }
            }

            if (!transferring_sessions.empty()) {
                std::uniform_int_distribution<> session_dis(0, transferring_sessions.size() - 1);
                int idx = session_dis(gen);
                setSessionStateValue(state, transferring_sessions[idx], BLOCKED);
                fixed = true;
            } else {
                // No transferring sessions, but still over bandwidth? Error state.
                // Force some non-transferring state? Or just return as is?
                // Let's just break, the state is inconsistent but generated.
                break;
            }
        }
        iterations++;
    } while (fixed && iterations < NUM_SESSIONS * 2); // Limit iterations

    return state;
}

// Convert state to a readable string (Unchanged)
std::string stateToString(const std::vector<int>& state) {
    // ... (content from scalable_network_2.txt) ...
    std::string result = "[";
    for (int i = 0; i < NUM_SESSIONS; ++i) {
        int val = getSessionStateValue(state, i);
        char c = '?';

        switch(val) {
            case IDLE: c = 'I'; break;
            case REQUESTING: c = 'Q'; break;
            case TRANSFERRING: c = 'T'; break;
            case BLOCKED: c = 'B'; break;
            default: c = '?'; break;
        }

        result += c;
        if (i < NUM_SESSIONS - 1) result += ",";
    }

    result += "]";
    return result;
}

// --- Metrics Calculation Functions ---

// Calculate average Q-values across all Q-functions and sampled states
double calculateAverageQValue(const std::vector<DdNode*>& q_functions,
                             const std::vector<std::vector<int>>& sample_states,
                             int num_actions) {
    if (q_functions.empty() || sample_states.empty() || !manager) return 0.0;
    double sum = 0.0;
    long long count = 0;
    for (const auto& state : sample_states) {
        if (isGoalState(state)) continue; // Skip terminal/goal states
        for (int a = 0; a < num_actions; ++a) {
             if (static_cast<size_t>(a) >= q_functions.size() || !q_functions[a]) continue; // Check bounds and null
            double q_val = evaluateADD(manager, q_functions[a], state);
            if (q_val > EVAL_ERROR_THRESHOLD) {
                sum += q_val;
                count++;
            }
        }
    }
    return (count > 0) ? (sum / count) : 0.0;
}

// Calculate average DAG size across all Q-functions
double calculateAverageDAGSize(const std::vector<DdNode*>& q_functions) {
    if (q_functions.empty() || !manager) return 0.0;
    long long total_size = 0;
    int valid_q_functions = 0;
    for (const auto& q_add : q_functions) {
        if (q_add) {
            total_size += Cudd_DagSize(q_add);
            valid_q_functions++;
        }
    }
    return (valid_q_functions > 0) ? (static_cast<double>(total_size) / valid_q_functions) : 0.0;
}


// Calculate Bellman error for a sample of states
double calculateBellmanError(const std::vector<DdNode*>& q_functions,
                            const std::vector<std::vector<int>>& sample_states,
                            int num_actions) {
    if (q_functions.empty() || sample_states.empty() || !manager) return 0.0;

    double total_error = 0.0;
    long long count = 0;

    for (const auto& state : sample_states) {
        if (isGoalState(state)) continue; // Skip goal states

        for (int action_idx = 0; action_idx < num_actions; ++action_idx) {
            Action action = static_cast<Action>(action_idx);
             if (static_cast<size_t>(action_idx) >= q_functions.size() || !q_functions[action_idx]) continue; // Check bounds and null

            // 1. Get current Q-value Q(s, a)
            double current_q = evaluateADD(manager, q_functions[action_idx], state);
            if (current_q < EVAL_ERROR_THRESHOLD) continue;

            // 2. Simulate the full next step (Agent + Environment)
            auto [state_after_agent, agent_action_succeeded] = applyAgentAction(state, action, num_actions);
            std::vector<int> next_state = environment_step(state_after_agent);

            // 3. Calculate reward for this transition (s -> s')
            double reward = (action == num_actions - 1) ? ACTION_COST :
                            (agent_action_succeeded ? ACTION_COST : UNBLOCK_FAIL_COST);
            if (isGoalState(next_state)) {
                reward += GOAL_REWARD;
            }

            // 4. Find max_a' Q(s', a')
            double max_next_q = 0.0; // Default to 0 if next state is goal
            if (!isGoalState(next_state)) {
                max_next_q = -std::numeric_limits<double>::infinity();
                bool found_valid_next_q = false;
                for (int next_action_idx = 0; next_action_idx < num_actions; ++next_action_idx) {
                     if (static_cast<size_t>(next_action_idx) >= q_functions.size() || !q_functions[next_action_idx]) continue; // Check bounds and null
                    double q_val = evaluateADD(manager, q_functions[next_action_idx], next_state);
                    if (q_val > EVAL_ERROR_THRESHOLD) {
                        max_next_q = std::max(max_next_q, q_val);
                        found_valid_next_q = true;
                    }
                }
                if (!found_valid_next_q) max_next_q = 0.0;
                 else if (max_next_q == -std::numeric_limits<double>::infinity()) max_next_q = 0.0;
            }

            // 5. Calculate target and error
            double target_q_value = reward + GAMMA * max_next_q;
            double error = std::abs(current_q - target_q_value);
            total_error += error;
            count++;
        } // end loop over actions
    } // end loop over sample_states

    return (count > 0) ? (total_error / count) : 0.0;
}
// Collect all unique nodes in an ADD/BDD structure
void collectUniqueNodes(DdNode* node, std::unordered_set<DdNode*>& uniqueNodes) {
    if (!node) return;
    
    // Skip if already processed
    DdNode* regular = Cudd_Regular(node);
    if (uniqueNodes.find(regular) != uniqueNodes.end())
        return;
    
    // Add this node
    uniqueNodes.insert(regular);
    
    // Skip terminal nodes
    if (Cudd_IsConstant(regular))
        return;
    
    // Process children
    collectUniqueNodes(Cudd_T(regular), uniqueNodes);
    collectUniqueNodes(Cudd_E(regular), uniqueNodes);
}

// Calculate actual memory used by Q-functions in megabytes
double calculateActualADDQMemory(const std::vector<DdNode*>& q_functions) {
    if (q_functions.empty()) return 0.0;
    
    // Count unique nodes across all Q-functions
    std::unordered_set<DdNode*> uniqueNodes;
    for (const auto& q_add : q_functions) {
        if (!q_add) continue;
        collectUniqueNodes(q_add, uniqueNodes);
    }
    
    // Calculate memory in MB (using actual DdNode size)
    double totalMemoryMB = uniqueNodes.size() * sizeof(DdNode) / (1024.0 * 1024.0);
    return totalMemoryMB;
}

// --- Symbolic Q-Learning (with Metrics Hooks) ---
std::map<std::string, Action> symbolicQLearning(bool verbose = true,
                                               int sample_num_states_metrics = 1000, // Renamed for clarity
                                               int cache_cleanup_interval = 5000,
                                               bool collect_metrics = true,
                                               int policy_sample_size = 10000) { // Added flag
    auto global_start_time = std::chrono::high_resolution_clock::now(); // Renamed
    std::map<std::string, Action> policy;
    int num_actions = NUM_SESSIONS + 1; // Unblock actions + wait

    if (!manager || vars.empty()) { std::cerr << "Err: Manager/vars not ready\n"; return {}; }

    std::random_device rd_env; env_gen.seed(rd_env()); // Use global env_gen

    // Create goal BDD
    if (goal_bdd) Cudd_RecursiveDeref(manager, goal_bdd);
    goal_bdd = createGoalBDD();
    if (!goal_bdd) { std::cerr << "Err: Goal BDD creation failed\n"; return {}; }
    if (verbose) { std::cout << "Goal BDD created." << std::endl; printNodeInfo("goal_bdd", goal_bdd); }

    // Initialize Q-functions
    std::vector<DdNode*> q_functions(num_actions);
    DdNode* initial_q_add = Cudd_addConst(manager, 0.0);
    if (!initial_q_add) { std::cerr << "Err: Failed to create initial Q ADD\n"; Cudd_RecursiveDeref(manager, goal_bdd); return {}; }
    Cudd_Ref(initial_q_add);
    for (int a = 0; a < num_actions; ++a) { q_functions[a] = initial_q_add; Cudd_Ref(q_functions[a]); }
    Cudd_RecursiveDeref(manager, initial_q_add);
    if (verbose) std::cout << "Initialized " << num_actions << " Q-functions." << std::endl;

    int total_bdd_vars = NUM_SESSIONS * BITS_PER_SESSION;
    StateCache state_cache(manager);
    std::random_device rd_agent; std::mt19937 agent_gen(rd_agent());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::uniform_int_distribution<> action_dis(0, num_actions - 1);

    // Sample states for metrics calculation
    std::vector<std::vector<int>> sampled_states_for_metrics;
    if (collect_metrics && sample_num_states_metrics > 0) {
        sampled_states_for_metrics.reserve(sample_num_states_metrics);
        std::set<std::vector<int>> unique_states; // Avoid duplicates in sample
         // Generate more than needed initially to increase chance of diverse samples
         for (int i = 0; i < sample_num_states_metrics * 2 && unique_states.size() < (size_t)sample_num_states_metrics; ++i) {
            unique_states.insert(generateRandomState(total_bdd_vars, agent_gen));
        }
         for(const auto& s : unique_states) {
             sampled_states_for_metrics.push_back(s);
         }
        if (verbose) std::cout << "Sampled " << sampled_states_for_metrics.size() << " unique states for metrics calculation.\n";
    }

     // Reserve space for metrics vectors
     if (collect_metrics) {
         int num_metric_points = NUM_EPISODES / METRICS_INTERVAL + 1;
         metrics.avg_q_values.reserve(num_metric_points);
         metrics.avg_decision_dag_sizes.reserve(num_metric_points);
         metrics.episode_times.reserve(NUM_EPISODES + 1);
         metrics.terminal_state_visits.reserve(NUM_EPISODES + 1);
         metrics.bellman_errors.reserve(num_metric_points);
         metrics.memory_usage.reserve(num_metric_points);
     }


    if (verbose) { std::cout << "\nPerforming Symbolic Q-Learning for " << NUM_EPISODES << " episodes..." << std::endl; }

    int eval_fail = 0, statebdd_fail = 0, bddadd_fail = 0, addconst_fail = 0, ite_fail = 0;
    const int PRINT_INTERVAL = std::max(1, NUM_EPISODES / 10);
    const int MAX_STEPS_TO_PRINT = 2;

    for (int episode = 0; episode < NUM_EPISODES; ++episode) {
        auto episode_start = std::chrono::high_resolution_clock::now(); // Timing start
        int terminal_visits_in_episode = 0; // Metric tracking

        bool print_this_episode = verbose && (episode < 1 || (episode + 1) % PRINT_INTERVAL == 0);
        if (print_this_episode && episode > 0) std::cout << std::endl;
        if (print_this_episode) std::cout << "--- Episode " << (episode + 1) << " ---" << std::endl;

        std::vector<int> state;
        int start_attempts = 0; const int MAX_START_ATTEMPTS = 100;
        do { state = generateRandomState(total_bdd_vars, agent_gen); start_attempts++;
        } while (isGoalState(state) && start_attempts < MAX_START_ATTEMPTS);
        if (isGoalState(state) && verbose) std::cout << "Warning: Starting in goal state after " << start_attempts << " attempts\n";

        int steps_in_episode = 0; const int MAX_EPISODE_STEPS = 100;

        while (steps_in_episode < MAX_EPISODE_STEPS) {
            if (isGoalState(state)) { // Check if goal state reached
                terminal_visits_in_episode++;
                // Optional: break here if reaching goal ends episode
                 // break;
            }

            steps_in_episode++;
            bool print_this_step = print_this_episode && steps_in_episode <= MAX_STEPS_TO_PRINT;

            // --- Action Selection (Epsilon-Greedy) ---
             Action action;
             // ... (Epsilon-greedy logic - unchanged) ...
             if (dis(agent_gen) < EPSILON || num_actions <= 0) {
                 action = action_dis(agent_gen);
             } else {
                 double maxQ = -std::numeric_limits<double>::infinity();
                 action = num_actions - 1; // Default to WAIT
                 bool ok = true;
                 std::vector<Action> best_actions; // Changed name

                 for (Action a = 0; a < num_actions; ++a) {
                      if (static_cast<size_t>(a) >= q_functions.size() || !q_functions[a]) continue;
                     double q = evaluateADD(manager, q_functions[a], state);
                     if (q < EVAL_ERROR_THRESHOLD) { eval_fail++; ok = false; break; }
                     if (q > maxQ) { maxQ = q; best_actions.clear(); best_actions.push_back(a); }
                     else if (q == maxQ) { best_actions.push_back(a); }
                 }
                 if (!ok || best_actions.empty()) { action = action_dis(agent_gen); }
                 else { std::uniform_int_distribution<> tie(0, best_actions.size() - 1); action = best_actions[tie(agent_gen)]; }
             }
            // --- End Action Selection ---

            if (print_this_step) { /* Print Step Info */ }

            // --- Apply Action & Environment Step ---
            auto [state_after_agent, agent_action_succeeded] = applyAgentAction(state, action, num_actions);
            std::vector<int> nextState = environment_step(state_after_agent);
            // --- End Apply Action & Env Step ---

            if (print_this_step) { /* Print Next State */ }

            // --- Calculate Reward ---
             double reward = (action == num_actions - 1) ? ACTION_COST :
                             (agent_action_succeeded ? ACTION_COST : UNBLOCK_FAIL_COST);
             if (isGoalState(nextState)) { reward += GOAL_REWARD; }
            // --- End Calculate Reward ---

            // --- Find max Q'(s') ---
             double max_next_q = 0.0; // Default to 0 if goal
             if (!isGoalState(nextState)) {
                  max_next_q = -std::numeric_limits<double>::infinity();
                  bool nextOk = true;
                  for (Action na = 0; na < num_actions; ++na) {
                       if (static_cast<size_t>(na) >= q_functions.size() || !q_functions[na]) continue;
                       double q = evaluateADD(manager, q_functions[na], nextState);
                       if (q < EVAL_ERROR_THRESHOLD) { eval_fail++; nextOk = false; break; }
                       max_next_q = std::max(max_next_q, q);
                  }
                  if (!nextOk || max_next_q == -std::numeric_limits<double>::infinity()) { max_next_q = 0.0; }
             }
            // --- End Find max Q'(s') ---

            double target_q_value = reward + GAMMA * max_next_q;

            // --- Create BDDs/ADDs for Update ---
            DdNode* stateBdd = state_cache.createStateBDD(state);
            if (!stateBdd) { statebdd_fail++; state = nextState; continue; }
            DdNode* stateAdd = Cudd_BddToAdd(manager, stateBdd);
             if (!stateAdd) { bddadd_fail++; Cudd_RecursiveDeref(manager, stateBdd); state = nextState; continue; }
             Cudd_Ref(stateAdd); Cudd_RecursiveDeref(manager, stateBdd);
            // --- End Create BDDs/ADDs ---

            // --- Get Old Q, Calculate New Q Scalar ---
            DdNode* current_q_add = q_functions[action];
            double oldQ = evaluateADD(manager, current_q_add, state);
            if (oldQ < EVAL_ERROR_THRESHOLD) { eval_fail++; Cudd_RecursiveDeref(manager, stateAdd); state = nextState; continue; }
            double newQ = (1.0 - ALPHA) * oldQ + ALPHA * target_q_value;
            // --- End Get/Calculate Q ---

            if (print_this_step) { /* Print Q values */ }

            // --- Create New Q Constant & Update ADD ---
            DdNode* newQ_s_add = Cudd_addConst(manager, newQ);
            if (!newQ_s_add) { addconst_fail++; Cudd_RecursiveDeref(manager, stateAdd); state = nextState; continue; }
            Cudd_Ref(newQ_s_add);
            DdNode* updatedQAdd = Cudd_addIte(manager, stateAdd, newQ_s_add, current_q_add);
            if (!updatedQAdd) { ite_fail++; Cudd_RecursiveDeref(manager, stateAdd); Cudd_RecursiveDeref(manager, newQ_s_add); state = nextState; continue; }
            Cudd_Ref(updatedQAdd);
            // --- End Update ADD ---

            // --- Clean up & State Transition ---
            Cudd_RecursiveDeref(manager, stateAdd);
            Cudd_RecursiveDeref(manager, newQ_s_add);
            Cudd_RecursiveDeref(manager, q_functions[action]);
            q_functions[action] = updatedQAdd;
            state = nextState;
            // --- End Clean up & Transition ---

        } // End while steps

        if (print_this_episode && steps_in_episode >= MAX_EPISODE_STEPS) { std::cout << "  Episode finished due to MAX_STEPS." << std::endl; }

        // --- Track Metrics (Episode End) ---
         if (collect_metrics) {
             metrics.terminal_state_visits.push_back(terminal_visits_in_episode); // Track goal visits
             auto episode_end = std::chrono::high_resolution_clock::now();
             metrics.episode_times.push_back(std::chrono::duration<double>(episode_end - episode_start).count());

             if ((episode + 1) % METRICS_INTERVAL == 0 && !sampled_states_for_metrics.empty()) {
                 double avg_q = calculateAverageQValue(q_functions, sampled_states_for_metrics, num_actions);
                 double avg_dag = calculateAverageDAGSize(q_functions);
                 double bellman_err = calculateBellmanError(q_functions, sampled_states_for_metrics, num_actions);
                 // Use sizeof(DdNode) for estimation, acknowledging it's a proxy
                 double mem_usage = calculateActualADDQMemory(q_functions);

                 metrics.avg_q_values.push_back(avg_q);
                 metrics.avg_decision_dag_sizes.push_back(avg_dag);
                 metrics.bellman_errors.push_back(bellman_err);
                 metrics.memory_usage.push_back(mem_usage);

                 if (verbose && print_this_episode) {
                     std::cout << "  Metrics @ Ep " << (episode + 1) << ": AvgQ=" << avg_q << ", AvgDAG=" << avg_dag
                               << ", BellmanErr=" << bellman_err << ", EstMem=" << mem_usage << "MB" << std::endl;
                 }
             } else if ((episode + 1) % METRICS_INTERVAL == 0) {
                  // Push defaults if no samples
                  metrics.avg_q_values.push_back(0);
                  double avg_dag = calculateAverageDAGSize(q_functions);
                  metrics.avg_decision_dag_sizes.push_back(avg_dag);
                  metrics.bellman_errors.push_back(0);
                  metrics.memory_usage.push_back(avg_dag * sizeof(DdNode) / (1024.0 * 1024.0));
             }
         } // --- End Track Metrics ---


        // --- Periodic Cache Cleanup (Unchanged) ---
        if ((episode + 1) % cache_cleanup_interval == 0) {
             if (verbose) std::cout << "  Cleaning state cache at episode " << (episode + 1) << "\n";
             state_cache = StateCache(manager); // Recreate cache (old one destructs)
             Cudd_ReduceHeap(manager, CUDD_REORDER_SAME, 0); // Optional GC
        }
    } // End for episodes

    if (verbose) { /* Print failure counts */ }

    // --- Extract Policy ---
    policy.clear(); // Clear just in case

    // Determine which states to use for policy extraction
    std::vector<std::vector<int>> states_for_policy_extraction; // Use a consistent type

    if (!sampled_states_for_metrics.empty()) {
        // Use the pre-sampled states if they exist
        states_for_policy_extraction = sampled_states_for_metrics;
        if (verbose) std::cout << "Extracting policy using " << states_for_policy_extraction.size() << " pre-sampled states..." << std::endl;
    } else if (policy_sample_size > 0) {
        // If no pre-sampled states, generate a new sample specifically for policy extraction
        if (verbose) std::cout << "No pre-sampled states found. Generating " << policy_sample_size << " random states for policy extraction..." << std::endl;
        states_for_policy_extraction.reserve(policy_sample_size);
        std::set<std::vector<int>> unique_states; // Use set to avoid duplicates
        for (int i = 0; i < policy_sample_size * 2 && unique_states.size() < (size_t)policy_sample_size; ++i) {
            unique_states.insert(generateRandomState(total_bdd_vars, agent_gen));
        }
        for(const auto& s : unique_states) {
            states_for_policy_extraction.push_back(s);
        }
         if (verbose) std::cout << "Generated " << states_for_policy_extraction.size() << " unique states for policy extraction." << std::endl;
    } else {
        if (verbose) std::cout << "Policy extraction skipped (policy_sample_size is 0 and no pre-sampled states)." << std::endl;
        // Policy map will remain empty
    }


    // Extract policy using the selected/generated states
    if (verbose && !states_for_policy_extraction.empty()) std::cout << "Extracting policy from " << states_for_policy_extraction.size() << " states..." << std::endl;

    for (const auto& s : states_for_policy_extraction) { // Iterate over the correctly typed vector
        std::string sStr = stateToString(s);
        if (policy.count(sStr)) continue; // Avoid duplicates if sample had them

        if (isGoalState(s)) {
            policy[sStr] = INVALID_ACTION;
        } else {
            // ... (rest of the policy extraction logic using 's' remains the same) ...
            Action bestA = num_actions - 1; // Default WAIT
            double maxQ = -std::numeric_limits<double>::infinity();
            bool found_best = false;
            std::vector<Action> best_actions;

            for (Action a = 0; a < num_actions; ++a) {
                 if (static_cast<size_t>(a) >= q_functions.size() || !q_functions[a]) continue;
                double q = evaluateADD(manager, q_functions[a], s);
                if (q > EVAL_ERROR_THRESHOLD) {
                    if (!found_best || q > maxQ) { maxQ = q; best_actions.clear(); best_actions.push_back(a); found_best = true; }
                     else if (q == maxQ) { best_actions.push_back(a); }
                }
            }
            if (found_best && !best_actions.empty()) {
                 std::uniform_int_distribution<> tie(0, best_actions.size() - 1);
                 policy[sStr] = best_actions[tie(agent_gen)];
            } else { policy[sStr] = num_actions - 1; } // Default WAIT
        }
    }
    if (verbose && !states_for_policy_extraction.empty()) std::cout << "Policy extracted for " << policy.size() << " unique states." << std::endl;

    // --- Clean up Q-functions ---
    if (verbose) std::cout << "Cleaning up Q-functions..." << std::endl;
    for (auto& q : q_functions) { if (q) Cudd_RecursiveDeref(manager, q); }
    q_functions.clear();

    auto global_end_time = std::chrono::high_resolution_clock::now(); // Renamed
    if (verbose) { std::cout << "Learn fn completed in " << std::chrono::duration<double>(global_end_time - global_start_time).count() << "s.\n"; }

    return policy;
}

// --- Simulation (Unchanged) ---
void runSimulation(const std::map<std::string, Action>& policy, int numTrials, bool verbose = true) {
    // ... (Keep the existing simulation logic from scalable_network_2.txt) ...
    // ... (It uses isGoalState, applyAgentAction, environment_step correctly) ...
    if (policy.empty() || !manager || !goal_bdd) {
        std::cerr << "Sim Err: Policy/Mgr/Goal missing\n";
        return;
    }

    int total_bdd_vars = NUM_SESSIONS * BITS_PER_SESSION;
    int num_actions = NUM_SESSIONS + 1;

    long long totalSteps = 0;
    int successCount = 0; // Count trials that run to MAX_STEPS
    int errorCount = 0;   // Count states not found in policy
    long long goalHolds = 0; // Count steps where system was in goal state

    std::random_device rd_sim;
    env_gen.seed(rd_sim()); // Re-seed global gen for simulation
    std::mt19937 sim_gen(rd_sim());

    for (int trial = 0; trial < numTrials; ++trial) {
        std::vector<int> state = generateRandomState(total_bdd_vars, sim_gen);
        int steps = 0;
        const int MAX_STEPS = 200; // Simulation step limit

        if (verbose && trial < 3) {
            std::cout << "\n--- Trial " << (trial + 1) << " Start: " << stateToString(state) << " ---\n";
        }

        while (steps < MAX_STEPS) {
            totalSteps++;
            std::string stateStr = stateToString(state);
            auto policy_it = policy.find(stateStr);
            Action action;

            if (policy_it == policy.end()) {
                // State not in policy sample - use default action (WAIT)
                action = num_actions - 1;
                errorCount++;
                if (verbose && trial < 3) {
                    std::cerr << "  Sim Warning: State " << stateStr << " not in policy sample. Using WAIT.\n";
                }
                 // Optional: Evaluate Q-values directly here to find best action if needed,
                 // but for simulation, sticking to the policy map (or default) is common.
            } else {
                action = policy_it->second;
                if (action == INVALID_ACTION) {
                    // Policy indicates terminal/goal state, agent should WAIT
                    action = num_actions - 1;
                }
            }

            if (verbose && trial < 3) {
                std::cout << "  Sim Step " << steps << ": S=" << stateStr
                          << " A=" << getActionName(action, num_actions);
            }

            std::pair<std::vector<int>, bool> agent_res = applyAgentAction(state, action, num_actions);
            std::vector<int> state_after_agent = agent_res.first;

            state = environment_step(state_after_agent); // Update state via environment

            if (verbose && trial < 3) {
                std::cout << " -> S'=" << stateToString(state) << std::endl;
            }

            if (isGoalState(state)) {
                goalHolds++;
            }

            steps++;
        } // end while steps < MAX_STEPS

        if (steps >= MAX_STEPS) {
            successCount++; // Count trial as "completed" if it reached max steps
        }
    } // end for trial

    double goalHoldRate = (totalSteps > 0) ? static_cast<double>(goalHolds) / totalSteps * 100.0 : 0.0;

    std::cout << "\n--- Simulation Results (" << numTrials << " trials, " << totalSteps << " total steps) ---\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Trials Completed (Reached Max Steps): " << successCount << "/" << numTrials << "\n";
    std::cout << "  Goal State Hold Rate (Avg % Steps): " << goalHoldRate << "%\n";
    std::cout << "  Policy Errors Encountered (State not found):  " << errorCount << "\n";
    std::cout << "-------------------------------------------------------------\n";
}


// --- Policy Printing (Unchanged) ---
void printPolicy(const std::map<std::string, Action>& policy, int num_actions) {
    // ... (content from scalable_network_2.txt) ...
    std::cout << "\n--- Learned Policy (Sample) ---\n";
    int stateWidth = NUM_SESSIONS * 2 + 3; // Adjust for state string format
    std::cout << std::setw(stateWidth) << std::left << "State"
              << std::setw(18) << std::left << "Action" << "\n";
    std::cout << std::string(stateWidth + 18, '-') << "\n";

    std::vector<std::string> stateStrs;
    stateStrs.reserve(policy.size());
    for (const auto& entry : policy) stateStrs.push_back(entry.first);
    std::sort(stateStrs.begin(), stateStrs.end());

    int printedCount = 0;
    const int MAX_PRINT_POLICY = 40;
    int terminalCount = 0;
    int nonTerminalCount = 0;

    for (const auto& stateStr : stateStrs) {
        Action action = policy.at(stateStr);
        if (action == INVALID_ACTION) {
            terminalCount++;
        } else {
            nonTerminalCount++;
            if (printedCount < MAX_PRINT_POLICY) {
                std::cout << std::setw(stateWidth) << std::left << stateStr
                          << std::setw(18) << std::left << getActionName(action, num_actions) << "\n";
                printedCount++;
            }
        }
    }

    if (nonTerminalCount > printedCount) {
        std::cout << "... (omitting " << (nonTerminalCount - printedCount) << " non-terminal states)\n";
    }

    // Only print terminal line if any terminal states were found in the sample
    if (terminalCount > 0) {
        std::cout << std::setw(stateWidth) << std::left << "<Goal States>"
                  << std::setw(18) << std::left << "Terminal" << "\n";
    }

    std::cout << std::string(stateWidth + 18, '-') << "\n"; // Added separator
    std::cout << "Total states in policy sample: " << policy.size()
              << " (" << nonTerminalCount << " non-terminal, "
              << terminalCount << " terminal)" << std::endl; // Clarified it's a sample
    std::cout << "---------------------------\n";
}

// --- CSV Saving Function ---
void saveMetricsToCSV() {
     if (metrics.avg_q_values.empty() && metrics.terminal_state_visits.empty()) {
         std::cout << "No metrics collected, CSV files not generated." << std::endl;
         return;
     }
    std::cout << "Saving metrics to CSV files..." << std::endl;
    const std::string prefix = "add_q_network_"; // Specific prefix

    // Save average Q-values
    if (!metrics.avg_q_values.empty()) {
        std::ofstream file(prefix + "avg_values.csv"); file << "Episode,AvgQValue\n";
        for (size_t i = 0; i < metrics.avg_q_values.size(); ++i) file << (i + 1) * METRICS_INTERVAL << "," << metrics.avg_q_values[i] << "\n";
    }
    // Save DAG sizes
    if (!metrics.avg_decision_dag_sizes.empty()) {
        std::ofstream file(prefix + "dag_sizes.csv"); file << "Episode,AvgNodeCount\n";
        for (size_t i = 0; i < metrics.avg_decision_dag_sizes.size(); ++i) file << (i + 1) * METRICS_INTERVAL << "," << metrics.avg_decision_dag_sizes[i] << "\n";
    }
    // Save Bellman errors
    if (!metrics.bellman_errors.empty()) {
        std::ofstream file(prefix + "bellman_errors.csv"); file << "Episode,AvgError\n";
        for (size_t i = 0; i < metrics.bellman_errors.size(); ++i) file << (i + 1) * METRICS_INTERVAL << "," << metrics.bellman_errors[i] << "\n";
    }
    // Save memory usage
    if (!metrics.memory_usage.empty()) {
        std::ofstream file(prefix + "memory_usage.csv"); file << "Episode,MemoryMB\n";
        for (size_t i = 0; i < metrics.memory_usage.size(); ++i) file << (i + 1) * METRICS_INTERVAL << "," << metrics.memory_usage[i] << "\n";
    }
    // Save terminal state visits
     if (!metrics.terminal_state_visits.empty()) {
        std::ofstream file(prefix + "terminal_visits.csv"); file << "Episode,TerminalVisits\n";
        for (size_t i = 0; i < metrics.terminal_state_visits.size(); ++i) file << i + 1 << "," << metrics.terminal_state_visits[i] << "\n";
    }
    // Save episode times
     if (!metrics.episode_times.empty()) {
        std::ofstream file(prefix + "episode_times.csv"); file << "Episode,DurationSeconds\n";
        for (size_t i = 0; i < metrics.episode_times.size(); ++i) file << i + 1 << "," << metrics.episode_times[i] << "\n";
    }
    std::cout << "Metrics saved successfully." << std::endl;
}


// --- Help Message (Unchanged) ---
void printHelp() {
    // ... (content from scalable_network_2.txt) ...
    std::cout << "Usage: program [options]\n";
    std::cout << "Options:\n";
    std::cout << "  -n N          Set number of sessions to N (default: 8)\n";
    std::cout << "  -bw N         Set maximum bandwidth to N (default: 15)\n";
    std::cout << "  -goal N       Set minimum transferring sessions for goal to N (default: 2)\n";
    std::cout << "  -e N          Set number of episodes to N (default: " << NUM_EPISODES << ")\n"; // Show default
    std::cout << "  -a F          Set alpha learning rate to F (default: " << ALPHA << ")\n";
    std::cout << "  -g F          Set gamma discount factor to F (default: " << GAMMA << ")\n";
    std::cout << "  -eps F        Set epsilon for exploration to F (default: " << EPSILON << ")\n";
    std::cout << "  -pf F         Set P_FINISH probability to F (default: " << P_FINISH << ")\n";
    std::cout << "  -pr F         Set P_NEW_REQUEST probability to F (default: " << P_NEW_REQUEST << ")\n";
    std::cout << "  -s N          Set policy sampling size to N (default: 10000)\n"; // Keep -s name
    std::cout << "  -metrics N    Collect metrics & save CSVs (sample N states for calcs)\n"; // Added -metrics
    std::cout << "  -sim N        Run N simulation trials after learning\n";
    std::cout << "  -v            Enable verbose mode\n";
    std::cout << "  -h            Print this help message\n";
}

// --- Main Function ---
int main(int argc, char* argv[]) {
    bool verbose = false;
    int numSimTrials = 0;
    int policy_sample_size = 10000; // Renamed from metrics_sample_size for clarity
    int metrics_sample_size = 1000; // Sample size specifically for metric calcs
    bool collect_and_save_metrics = false; // Flag to enable collection/saving
    int cache_cleanup_interval = 5000; // Keep cache cleanup interval

    // --- Argument Parsing (Added -metrics) ---
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        try { // Added try-catch block for robustness
            if (arg == "-v") { verbose = true; }
            else if (arg == "-metrics" && i + 1 < argc) { collect_and_save_metrics = true; metrics_sample_size = std::stoi(argv[++i]); if(metrics_sample_size<0) metrics_sample_size=0;}
            else if (arg == "-metrics") { collect_and_save_metrics = true; } // Allow flag without value
            else if (arg == "-n" && i + 1 < argc) {
                 NUM_SESSIONS = std::stoi(argv[++i]); if (NUM_SESSIONS <= 0) { /*...*/ return 1; }
                 // Resize BANDWIDTH_REQ if needed
                 if ((int)BANDWIDTH_REQ.size() < NUM_SESSIONS) {
                     int old_size = BANDWIDTH_REQ.size();
                     BANDWIDTH_REQ.resize(NUM_SESSIONS);
                     if (old_size > 0) {
                          double sum = std::accumulate(BANDWIDTH_REQ.begin(), BANDWIDTH_REQ.begin() + old_size, 0.0);
                          double avg = sum / old_size;
                          for (int j = old_size; j < NUM_SESSIONS; ++j) BANDWIDTH_REQ[j] = static_cast<int>(avg + 0.5);
                     } else { std::fill(BANDWIDTH_REQ.begin(), BANDWIDTH_REQ.end(), 3); } // Default if empty
                 } else { BANDWIDTH_REQ.resize(NUM_SESSIONS); } // Ensure exact size
            }
            else if (arg == "-bw" && i + 1 < argc) { MAX_BANDWIDTH = std::stoi(argv[++i]); if (MAX_BANDWIDTH <= 0) { /*...*/ return 1; } }
            else if (arg == "-goal" && i + 1 < argc) { MIN_TRANSFERRING_FOR_GOAL = std::stoi(argv[++i]); if (MIN_TRANSFERRING_FOR_GOAL < 0) { /*...*/ return 1; } }
            else if (arg == "-e" && i + 1 < argc) { NUM_EPISODES = std::stoi(argv[++i]); if (NUM_EPISODES <= 0) { /*...*/ return 1; } }
            else if (arg == "-a" && i + 1 < argc) { ALPHA = std::stod(argv[++i]); if (ALPHA <= 0 || ALPHA > 1) { /*...*/ return 1; } }
            else if (arg == "-g" && i + 1 < argc) { GAMMA = std::stod(argv[++i]); if (GAMMA < 0 || GAMMA > 1) { /*...*/ return 1; } }
            else if (arg == "-eps" && i + 1 < argc) { EPSILON = std::stod(argv[++i]); if (EPSILON < 0 || EPSILON > 1) { /*...*/ return 1; } }
            else if (arg == "-pf" && i + 1 < argc) { P_FINISH = std::stod(argv[++i]); if (P_FINISH < 0 || P_FINISH > 1) { /*...*/ return 1; } }
            else if (arg == "-pr" && i + 1 < argc) { P_NEW_REQUEST = std::stod(argv[++i]); if (P_NEW_REQUEST < 0 || P_NEW_REQUEST > 1) { /*...*/ return 1; } }
            else if (arg == "-s" && i + 1 < argc) { policy_sample_size = std::stoi(argv[++i]); if (policy_sample_size < 0) policy_sample_size = 0; }
            else if (arg == "-sim" && i + 1 < argc) { numSimTrials = std::stoi(argv[++i]); if (numSimTrials < 0) numSimTrials = 0; }
            else if (arg == "-h") { printHelp(); return 0; }
            else { std::cerr << "Unknown option: " << arg << "\n"; printHelp(); return 1; }
        } catch (const std::exception& e) { std::cerr << "Error processing argument for " << arg << ": " << e.what() << std::endl; return 1; }
    } // --- End Arg Parsing ---

    // --- Print Parameters ---
    std::cout << "--- Network Resource Allocation Parameters ---\n";
    std::cout << "  NUM_SESSIONS: " << NUM_SESSIONS << ", BITS_PER_SESSION: " << BITS_PER_SESSION << "\n";
    int total_bdd_vars = NUM_SESSIONS * BITS_PER_SESSION;
    std::cout << "  TOTAL_BDD_VARS: " << total_bdd_vars << "\n";
    // ... (rest of parameter printing unchanged) ...
     long long state_space_size = 1; bool overflow = false;
     for (int i = 0; i < NUM_SESSIONS; ++i) {
         if (state_space_size > std::numeric_limits<long long>::max() / NUM_SESSION_STATES) { overflow = true; break; }
         state_space_size *= NUM_SESSION_STATES;
     }
     if (overflow) std::cout << "  State Space Size: > 2^63 (too large to represent)\n";
     else std::cout << "  State Space Size: " << state_space_size << "\n";
     int num_actions = NUM_SESSIONS + 1;
     std::cout << "  NUM_ACTIONS: " << num_actions << "\n";
     std::cout << "  MAX_BANDWIDTH: " << MAX_BANDWIDTH << "\n";
     std::cout << "  Bandwidth Req: [";
     for (size_t i = 0; i < BANDWIDTH_REQ.size(); ++i) std::cout << BANDWIDTH_REQ[i] << (i == BANDWIDTH_REQ.size() - 1 ? "" : ", ");
     std::cout << "]\n";
     std::cout << "  Goal: >= " << MIN_TRANSFERRING_FOR_GOAL << " sessions transferring.\n";
     std::cout << "  P_FINISH: " << P_FINISH << ", P_NEW_REQUEST: " << P_NEW_REQUEST << "\n";
     std::cout << "  Gamma: " << GAMMA << ", Epsilon: " << EPSILON << ", Alpha: " << ALPHA << "\n";
     std::cout << "  Episodes: " << NUM_EPISODES << "\n";
     std::cout << "  Policy Sampling: " << policy_sample_size << " states\n";
     std::cout << "  Collect Metrics: " << (collect_and_save_metrics ? "Yes" : "No") << std::endl;
     if (collect_and_save_metrics) std::cout << "  Metrics Sample Size: " << metrics_sample_size << std::endl;
     std::cout << "----------------------------------------------\n";
    // --- End Print Parameters ---


    // --- Initialize CUDD ---
    manager = Cudd_Init(total_bdd_vars, 0, CUDD_UNIQUE_SLOTS, CUDD_CACHE_SLOTS, 0);
    if (!manager) { std::cerr << "Failed to initialize CUDD manager.\n"; return 1; }
    Cudd_AutodynEnable(manager, CUDD_REORDER_SIFT); // Keep reordering enabled
    // --- End CUDD Init ---


    // --- Create BDD Variables ---
    vars.resize(total_bdd_vars); bool vars_ok = true;
    for (int i = 0; i < total_bdd_vars; ++i) { vars[i] = Cudd_bddIthVar(manager, i); if (!vars[i]) { vars_ok = false; break; } }
    if (!vars_ok) { std::cerr << "Failed to create BDD variables.\n"; Cudd_Quit(manager); return 1; }
    // --- End BDD Vars ---


    // --- Run Learning ---
    std::map<std::string, Action> policy = symbolicQLearning(verbose, metrics_sample_size, cache_cleanup_interval, collect_and_save_metrics,policy_sample_size); // Pass new args


    // --- Post-Learning ---
    if (!policy.empty()) {
        printPolicy(policy, num_actions); // Pass num_actions
    } else if (total_bdd_vars > 0) { // Check if state space was non-trivial
        std::cerr << "Learning failed (empty policy).\n";
        // Don't necessarily exit, maybe simulation or cleanup should still happen?
    }


    // --- Run Simulation ---
    if (numSimTrials > 0 && !policy.empty()) {
        if (!goal_bdd) { // Recreate goal BDD if needed (should exist though)
            std::cerr << "Warning: Goal BDD missing for simulation, attempting recreate.\n";
            goal_bdd = createGoalBDD();
        }
        if (goal_bdd) {
            runSimulation(policy, numSimTrials, verbose);
        } else {
             std::cerr << "Error: Cannot run simulation without goal BDD.\n";
        }
    } else if (numSimTrials > 0 && policy.empty()) {
         std::cout << "Skipping simulation as policy is empty.\n";
    } else {
        std::cout << "\nRun with '-sim N' to simulate.\n";
    }


    // --- Save Metrics ---
    if (collect_and_save_metrics) {
        saveMetricsToCSV();
    }


    // --- Clean Up ---
    if (manager) {
        if (goal_bdd) { Cudd_RecursiveDeref(manager, goal_bdd); goal_bdd = nullptr; }
        vars.clear();
        int check = Cudd_CheckZeroRef(manager); // Check refs before quit
        if (check > 0 && verbose) std::cerr << "Warning: " << check << " CUDD nodes still referenced before Quit.\n";
        Cudd_Quit(manager);
        manager = nullptr;
    }

    return 0;
}
