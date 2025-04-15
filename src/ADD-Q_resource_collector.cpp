// add_q_resource_collector.cpp
// Efficient ADD-Q algorithm implementation for resource collection in a grid world.
// Demonstrates handling composite state (location + resource status).

#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <cmath>
#include <cassert>
#include <iomanip>
#include <string>
#include <sstream>
#include <fstream>      // For file I/O
#include <limits>
#include <algorithm>
#include <chrono>
#include <random>
#include <numeric>
#include <functional>
#include <memory>
#include <set>          // For resource locations

// Include CUDD header
#include "cudd.h"
#include "cuddInt.h"

// --- Grid World Parameters ---
int GRID_SIZE_X = 5;        // Width of grid
int GRID_SIZE_Y = 5;        // Height of grid
int NUM_OBSTACLES = 3;      // Number of obstacles
int NUM_RESOURCES = 3;      // Number of resources to collect

// --- State Encoding ---
// Agent location encoding - log2 encoding
int BITS_PER_COORD = 3;     // ceil(log2(5)) - Needs update if grid size changes
int LOCATION_BITS = BITS_PER_COORD * 2;  // x and y coordinates
// Resource status encoding - 1 bit per resource (1=present, 0=collected)
int RESOURCE_BITS = NUM_RESOURCES;
int TOTAL_STATE_BITS = LOCATION_BITS + RESOURCE_BITS;

// --- Environment Objects ---
std::vector<std::vector<bool>> obstacles; // true = obstacle
std::set<std::pair<int, int>> resource_locations; // Set of (x,y) pairs

// --- RL Parameters ---
double GAMMA = 0.99;
double EPSILON = 0.2;
double ALPHA = 0.1;
int NUM_EPISODES = 10000; // Increased default
double RESOURCE_REWARD = 50.0; // Reward for collecting a resource
double MOVE_COST = -0.1;
double OBSTACLE_COST = -5.0;
// double TERMINAL_REWARD = 100.0; // Optional: Extra reward for collecting the last resource

// --- Action Definitions ---
enum Action { INVALID_ACTION = -1, UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3 };
const int NUM_ACTIONS = 4;
const double EVAL_ERROR_THRESHOLD = -9990.0; // Different threshold

// --- Metrics Structure (same as grid_world) ---
struct AddQMetrics {
    std::vector<double> avg_q_values;
    std::vector<double> avg_decision_dag_sizes; // Changed to double
    std::vector<double> episode_times;
    std::vector<int> terminal_state_visits; // Renamed from goal_visits
    std::vector<double> bellman_errors;
    std::vector<double> memory_usage;
    // avg_path_lengths might be less meaningful here, maybe track resources collected?
    std::vector<double> avg_resources_collected_per_ep; // New metric
};

// --- Globals ---
DdManager* manager = nullptr;
DdNode* terminal_bdd = nullptr; // BDD representing the terminal state (all resources collected)
std::vector<DdNode*> vars;       // All BDD variables (location + resources)
std::mt19937 env_gen;
AddQMetrics metrics;
const int METRICS_INTERVAL = 100; // Moved global

// --- Forward Declarations ---
void initializeEnvironment();
std::vector<int> createStateVector(int x, int y, const std::vector<bool>& resource_status);
int getCoordValue(const std::vector<int>& state, int coord_type);
bool getResourceStatus(const std::vector<int>& state, int resource_idx);
void setCoordValue(std::vector<int>& state, int coord_type, int value);
void setResourceStatus(std::vector<int>& state, int resource_idx, bool present);
DdNode* createTerminalBDD();
bool isTerminalState(const std::vector<int>& state); // Checks resource bits
std::string stateToString(const std::vector<int>& state);
double evaluateADD(DdManager* mgr, DdNode* add, const std::vector<int>& state);
std::pair<std::vector<int>, double> applyAgentAction(const std::vector<int>& current_state, Action action);
std::map<std::string, Action> symbolicQLearning(bool verbose, int sample_num_states_metrics, bool collect_metrics);
void runSimulation(const std::map<std::string, Action>& policy, int numTrials, bool verbose);
void visualizeGrid(const std::map<std::string, Action>& policy, const std::vector<int>& current_sim_state = {}); // Added state for visualization
void printPolicy(const std::map<std::string, Action>& policy);
void saveMetricsToCSV();
void printHelp();
double calculateAverageQValue(const std::vector<DdNode*>& q_functions, const std::vector<std::vector<int>>& sample_states);
double calculateAverageDAGSize(const std::vector<DdNode*>& q_functions);
double calculateBellmanError(const std::vector<DdNode*>& q_functions, const std::vector<std::vector<int>>& sample_states);
int countCollectedResources(const std::vector<int>& state); // Helper for metrics

// --- Helper Functions ---

void printNodeInfo(const char* name, DdNode* node, bool isAdd = false) {
    // (Same as grid_world.txt)
    if (!manager) return;
    if (!node) { std::cout << (isAdd ? "ADD " : "BDD ") << name << ": NULL node" << std::endl; return; }
    DdNode* regular_node = Cudd_Regular(node);
    std::cout << (isAdd ? "ADD " : "BDD ") << name << ": "
              << (void*)regular_node << (Cudd_IsComplement(node) ? "'" : "")
              << ", index = " << (Cudd_IsConstant(regular_node) ? -1 : Cudd_NodeReadIndex(regular_node))
              << ", val = " << (Cudd_IsConstant(regular_node) ? std::to_string(Cudd_V(regular_node)) : "N/A")
              << ", DagSize = " << Cudd_DagSize(node)
              << std::endl;
}

// Get variable index for location coordinate bit
int getLocVarIndex(int coord_type, int bit_idx) {
    assert(coord_type >= 0 && coord_type < 2);  // 0 for x, 1 for y
    assert(bit_idx >= 0 && bit_idx < BITS_PER_COORD);
    return coord_type * BITS_PER_COORD + bit_idx; // Location bits come first
}

// Get variable index for resource status bit
int getResourceVarIndex(int resource_idx) {
    assert(resource_idx >= 0 && resource_idx < NUM_RESOURCES);
    return LOCATION_BITS + resource_idx; // Resource bits come after location bits
}

// Extract coordinate value from state vector
int getCoordValue(const std::vector<int>& state, int coord_type) {
    if (state.empty() || coord_type < 0 || coord_type >= 2) return -1;
    int value = 0;
    for (int b = 0; b < BITS_PER_COORD; ++b) {
        int var_idx = getLocVarIndex(coord_type, b);
        if (var_idx >= (int)state.size()) return -1;
        if (state[var_idx] == 1) { value |= (1 << b); }
    }
    return value;
}

// Get status of a specific resource from state vector
bool getResourceStatus(const std::vector<int>& state, int resource_idx) {
    if (state.empty() || resource_idx < 0 || resource_idx >= NUM_RESOURCES) return false; // Default to not present if invalid
    int var_idx = getResourceVarIndex(resource_idx);
    if (var_idx >= (int)state.size()) return false;
    return (state[var_idx] == 1); // 1 means present
}

// Set coordinate value in state vector
void setCoordValue(std::vector<int>& state, int coord_type, int value) {
    if (state.empty() || coord_type < 0 || coord_type >= 2) return;
    int max_value = (1 << BITS_PER_COORD) - 1;
    value = std::min(std::max(value, 0), max_value); // Clamp value
    for (int b = 0; b < BITS_PER_COORD; ++b) {
        int var_idx = getLocVarIndex(coord_type, b);
        if (var_idx >= (int)state.size()) return;
        state[var_idx] = (value >> b) & 1;
    }
}

// Set status of a specific resource in state vector
void setResourceStatus(std::vector<int>& state, int resource_idx, bool present) {
    if (state.empty() || resource_idx < 0 || resource_idx >= NUM_RESOURCES) return;
    int var_idx = getResourceVarIndex(resource_idx);
    if (var_idx >= (int)state.size()) return;
    state[var_idx] = present ? 1 : 0;
}

// Create state vector from components
std::vector<int> createStateVector(int x, int y, const std::vector<bool>& resource_status) {
    std::vector<int> state(TOTAL_STATE_BITS);
    setCoordValue(state, 0, x);
    setCoordValue(state, 1, y);
    assert(resource_status.size() == NUM_RESOURCES);
    for (int i = 0; i < NUM_RESOURCES; ++i) {
        setResourceStatus(state, i, resource_status[i]);
    }
    return state;
}

// Convert state vector to readable string
std::string stateToString(const std::vector<int>& state) {
    if (state.size() != TOTAL_STATE_BITS) return "[invalid_state]";
    int x = getCoordValue(state, 0);
    int y = getCoordValue(state, 1);
    std::stringstream ss;
    ss << "(" << x << "," << y << ")|R:";
    for (int i = 0; i < NUM_RESOURCES; ++i) {
        ss << getResourceStatus(state, i);
    }
    return ss.str();
}

// Get action name for display
std::string getActionName(Action action) {
    switch (action) {
        case UP: return "UP";
        case RIGHT: return "RIGHT";
        case DOWN: return "DOWN";
        case LEFT: return "LEFT";
        case INVALID_ACTION: return "Terminal";
        default: return "Unknown";
    }
}

// Check if a location contains an obstacle
bool isObstacle(int x, int y) {
    if (x < 0 || x >= GRID_SIZE_X || y < 0 || y >= GRID_SIZE_Y) return true;
    return obstacles[y][x];
}

// Initialize obstacles and resource locations
void initializeEnvironment() {
    // --- Initialize Obstacles ---
    obstacles.assign(GRID_SIZE_Y, std::vector<bool>(GRID_SIZE_X, false));
    std::random_device rd_env;
    std::mt19937 gen(rd_env());
    std::uniform_int_distribution<> x_dis(0, GRID_SIZE_X - 1);
    std::uniform_int_distribution<> y_dis(0, GRID_SIZE_Y - 1);

    int placed_obstacles = 0;
    while (placed_obstacles < NUM_OBSTACLES) {
        int x = x_dis(gen);
        int y = y_dis(gen);
        if (!obstacles[y][x]) { // Don't overwrite
            obstacles[y][x] = true;
            placed_obstacles++;
        }
    }

    // --- Initialize Resource Locations ---
    resource_locations.clear();
    while (resource_locations.size() < (size_t)NUM_RESOURCES) {
        int x = x_dis(gen);
        int y = y_dis(gen);
        if (!obstacles[y][x]) { // Place resource only on non-obstacle cells
            resource_locations.insert({x, y});
        }
    }

    // Ensure obstacles didn't overwrite resources
    for (const auto& loc : resource_locations) {
        obstacles[loc.second][loc.first] = false;
    }
}

// --- CUDD Related Functions ---

// Create BDD for terminal state (all resource bits are 0)
DdNode* createTerminalBDD() {
    if (!manager || vars.empty() || RESOURCE_BITS == 0) return nullptr; // Need resources

    DdNode* termBdd = Cudd_ReadOne(manager); // Start with True
    Cudd_Ref(termBdd);

    for (int i = 0; i < NUM_RESOURCES; ++i) {
        int var_idx = getResourceVarIndex(i);
        if (var_idx >= (int)vars.size() || !vars[var_idx]) { /* Error */ Cudd_RecursiveDeref(manager, termBdd); return nullptr; }

        // Condition is resource_bit == 0 (i.e., NOT present)
        DdNode* literal = Cudd_Not(vars[var_idx]); // Negated variable
        DdNode* tmp = Cudd_bddAnd(manager, termBdd, literal);
        if (!tmp) { /* Error */ Cudd_RecursiveDeref(manager, termBdd); return nullptr; }
        Cudd_Ref(tmp);
        Cudd_RecursiveDeref(manager, termBdd);
        // literal is managed by CUDD cache relative to vars[var_idx]
        termBdd = tmp;
    }
    return termBdd; // Referenced BDD
}

// Check if state is terminal (all resources collected)
bool isTerminalState(const std::vector<int>& state) {
    if (state.size() != TOTAL_STATE_BITS) return false;
    for (int i = 0; i < NUM_RESOURCES; ++i) {
        if (getResourceStatus(state, i)) { // If any resource is present (bit is 1)
            return false;
        }
    }
    return true; // All resources collected (all bits are 0)
}

// Evaluate ADD for a specific state
double evaluateADD(DdManager* mgr, DdNode* add, const std::vector<int>& state) {
    if (!mgr || !add) return EVAL_ERROR_THRESHOLD;
    if (state.size() != TOTAL_STATE_BITS) return EVAL_ERROR_THRESHOLD;
    if (Cudd_IsConstant(add)) return Cudd_V(add);

    int managerSize = Cudd_ReadSize(mgr);
    if (managerSize < TOTAL_STATE_BITS) return EVAL_ERROR_THRESHOLD;

    // Cudd_Eval expects int* for assignment
    std::vector<int> assignment_vec = state; // Copy state to assignment
    // Ensure size matches manager (pad with 0 if needed, though shouldn't be necessary if init is correct)
    if(assignment_vec.size() < (size_t)managerSize) {
        assignment_vec.resize(managerSize, 0);
    }

    DdNode* evalNode = Cudd_Eval(mgr, add, assignment_vec.data());

    if (evalNode == NULL || !Cudd_IsConstant(evalNode)) {
        // std::cerr << "Warning: Cudd_Eval failed or returned non-constant in evaluateADD." << std::endl;
        return EVAL_ERROR_THRESHOLD;
    }
    return Cudd_V(evalNode);
}

// Create state BDD
DdNode* createStateBDD(DdManager* mgr, const std::vector<int>& state) {
    if (!mgr || state.size() != TOTAL_STATE_BITS) return Cudd_ReadLogicZero(mgr);

    DdNode* bdd = Cudd_ReadOne(mgr);
    Cudd_Ref(bdd);

    try {
        for (size_t i = 0; i < state.size(); ++i) {
            if (i >= vars.size() || !vars[i]) {
                throw std::runtime_error("Invalid BDD variable index or null var");
            }

            bool val = (state[i] == 1);
            DdNode* literal = Cudd_NotCond(vars[i], !val); // var if val=1, !var if val=0
            DdNode* tmp = Cudd_bddAnd(mgr, bdd, literal);
            if (!tmp) {
                throw std::runtime_error("Cudd_bddAnd failed");
            }
            Cudd_Ref(tmp);
            Cudd_RecursiveDeref(mgr, bdd); // Deref previous intermediate
            bdd = tmp; // Update current BDD
        }
    } catch (const std::runtime_error& e) {
        std::cerr << "Error in createStateBDD: " << e.what() << std::endl;
        if(bdd) Cudd_RecursiveDeref(mgr, bdd); // Clean up partially built BDD
        return Cudd_ReadLogicZero(mgr);
    }
    return bdd; // Return referenced BDD
}


// --- Environment Transition Function ---
// Returns <next_state_vector, immediate_reward>
std::pair<std::vector<int>, double> applyAgentAction(const std::vector<int>& current_state, Action action) {
    if (current_state.size() != TOTAL_STATE_BITS) {
        // Handle error: return current state with high penalty?
        return {current_state, OBSTACLE_COST * 10};
    }

    int x = getCoordValue(current_state, 0);
    int y = getCoordValue(current_state, 1);
    int next_x = x, next_y = y;

    // Calculate next potential position
    switch (action) {
        case UP:    next_y = std::max(0, y - 1); break;
        case RIGHT: next_x = std::min(GRID_SIZE_X - 1, x + 1); break;
        case DOWN:  next_y = std::min(GRID_SIZE_Y - 1, y + 1); break;
        case LEFT:  next_x = std::max(0, x - 1); break;
        default: break; // Should not happen
    }

    double immediate_reward = MOVE_COST; // Base cost for moving

    // Check for obstacle collision
    if (isObstacle(next_x, next_y)) {
        immediate_reward += OBSTACLE_COST;
        next_x = x; // Stay in place
        next_y = y;
    }

    // Create the next state vector (initially copy current)
    std::vector<int> next_state = current_state;

    // Update agent position in next state vector
    setCoordValue(next_state, 0, next_x);
    setCoordValue(next_state, 1, next_y);

    // Check for resource collection at the *final* position (next_x, next_y)
    int resource_idx = 0;
    for (const auto& loc : resource_locations) {
        if (next_x == loc.first && next_y == loc.second) {
            // Agent is at resource location
            if (getResourceStatus(current_state, resource_idx)) { // Check if resource WAS present
                immediate_reward += RESOURCE_REWARD;
                setResourceStatus(next_state, resource_idx, false); // Collect resource in next state
                // Optional: Add terminal reward if this was the last one
                // if (isTerminalState(next_state)) {
                //     immediate_reward += TERMINAL_REWARD;
                // }
            }
            // If resource was already collected, no extra reward/penalty
            break; // Agent can only be at one resource location at a time
        }
        resource_idx++;
    }

    return {next_state, immediate_reward};
}

// --- Metrics Calculations (Adapted from grid_world) ---

int countCollectedResources(const std::vector<int>& state) {
    if (state.size() != TOTAL_STATE_BITS) return 0;
    int collected = 0;
    for(int i = 0; i < NUM_RESOURCES; ++i) {
        if (!getResourceStatus(state, i)) { // Status is 0 if collected
            collected++;
        }
    }
    return collected;
}

double calculateAverageQValue(const std::vector<DdNode*>& q_functions,
                             const std::vector<std::vector<int>>& sample_states) {
    if (q_functions.empty() || sample_states.empty() || !manager) return 0.0;
    double sum = 0.0;
    long long count = 0;
    for (const auto& state : sample_states) {
        if (isTerminalState(state)) continue; // Skip terminal states
        for (size_t a = 0; a < q_functions.size(); ++a) {
            if (!q_functions[a]) continue;
            double q_val = evaluateADD(manager, q_functions[a], state);
            if (q_val > EVAL_ERROR_THRESHOLD) {
                sum += q_val;
                count++;
            }
        }
    }
    return (count > 0) ? (sum / count) : 0.0;
}

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

double calculateBellmanError(const std::vector<DdNode*>& q_functions,
                            const std::vector<std::vector<int>>& sample_states) {
    if (q_functions.empty() || sample_states.empty() || !manager) return 0.0;

    double total_error = 0.0;
    long long count = 0;

    for (const auto& state : sample_states) {
        if (isTerminalState(state)) continue; // Skip terminal states

        // Iterate through all possible actions from this state
        for (int action_idx = 0; action_idx < NUM_ACTIONS; ++action_idx) {
            Action action = static_cast<Action>(action_idx);
             if (static_cast<size_t>(action_idx) >= q_functions.size() || !q_functions[action_idx]) continue;

            // 1. Get current Q-value Q(s, a)
            double current_q = evaluateADD(manager, q_functions[action_idx], state);
            if (current_q < EVAL_ERROR_THRESHOLD) continue;

            // 2. Calculate the target value: R(s,a,s') + gamma * max_a' Q(s', a')
            auto [next_state, reward] = applyAgentAction(state, action);

            // 3. Find max_a' Q(s', a')
            double max_next_q = 0.0; // Default to 0 if next state is terminal
            if (!isTerminalState(next_state)) {
                max_next_q = -std::numeric_limits<double>::infinity();
                bool found_valid_next_q = false;
                for (int next_action_idx = 0; next_action_idx < NUM_ACTIONS; ++next_action_idx) {
                     if (static_cast<size_t>(next_action_idx) >= q_functions.size() || !q_functions[next_action_idx]) continue;
                    double q_val = evaluateADD(manager, q_functions[next_action_idx], next_state);
                    if (q_val > EVAL_ERROR_THRESHOLD) {
                        max_next_q = std::max(max_next_q, q_val);
                        found_valid_next_q = true;
                    }
                }
                if (!found_valid_next_q) max_next_q = 0.0; // If all next evals failed
                 else if (max_next_q == -std::numeric_limits<double>::infinity()) max_next_q = 0.0; // Ensure non-infinity if loop ran
            }

            // 4. Calculate target and error
            double target_q_value = reward + GAMMA * max_next_q;
            double error = std::abs(current_q - target_q_value);
            total_error += error;
            count++;
        } // end loop over actions
    } // end loop over sample_states

    return (count > 0) ? (total_error / count) : 0.0;
}


// --- Symbolic Q-Learning (Adapted from grid_world) ---
std::map<std::string, Action> symbolicQLearning(bool verbose = true,
                                                int sample_num_states_metrics = 1000,
                                                bool collect_metrics = true) {
    auto global_start_time = std::chrono::high_resolution_clock::now();
    std::map<std::string, Action> policy;

    // --- Validation & Setup ---
    if (!manager || vars.empty()) { std::cerr << "Manager/vars not ready\n"; return {}; }
    if (verbose) { std::cout << "Starting Symbolic Q-Learning for Resource Collector..." << std::endl; }

    // --- Create Terminal BDD ---
    if (terminal_bdd) Cudd_RecursiveDeref(manager, terminal_bdd);
    terminal_bdd = createTerminalBDD();
    if (!terminal_bdd) { std::cerr << "Error: Failed to create terminal BDD." << std::endl; return {}; }
    if (verbose) printNodeInfo("Terminal BDD (Resources Collected)", terminal_bdd);

    // --- Initialize Q-functions ---
    std::vector<DdNode*> q_functions(NUM_ACTIONS);
    DdNode* initial_q_add = Cudd_addConst(manager, 0.0);
    if (!initial_q_add) { /* Error */ Cudd_RecursiveDeref(manager, terminal_bdd); return {}; }
    Cudd_Ref(initial_q_add);
    for (int a = 0; a < NUM_ACTIONS; ++a) {
        q_functions[a] = initial_q_add;
        Cudd_Ref(q_functions[a]);
    }
    Cudd_RecursiveDeref(manager, initial_q_add);
    if (verbose) std::cout << "Initialized " << NUM_ACTIONS << " Q-functions." << std::endl;

    // --- Generate States for Sampling ---
    std::vector<std::vector<int>> all_possible_states;
    // Iterate through all location and resource combinations
    int max_loc_states = 1 << LOCATION_BITS;
    int max_res_states = 1 << RESOURCE_BITS;
    int valid_state_count = 0;
    all_possible_states.reserve(max_loc_states * max_res_states); // Rough upper bound

    if(verbose) std::cout << "Generating and filtering states..." << std::endl;
    for (int x = 0; x < GRID_SIZE_X; ++x) {
     for (int y = 0; y < GRID_SIZE_Y; ++y) {
         if (isObstacle(x, y)) continue; // Skip obstacle locations
         for (int res_combo = 0; res_combo < max_res_states; ++res_combo) {
              std::vector<bool> res_status(NUM_RESOURCES);
              int temp_combo = res_combo;
              for(int r=0; r<NUM_RESOURCES; ++r) {
                  res_status[r] = (temp_combo % 2 == 1);
                  temp_combo /= 2;
              }
              // Check if resource status matches actual locations for "present"
              bool possible_config = true;
              int r_idx = 0;
              for(const auto& loc : resource_locations) {
                  if (res_status[r_idx] && (x == loc.first && y == loc.second)) {
                      // Cannot be present at a location if agent is also there (would have been picked up)
                      // This constraint is complex. Let's just generate all combos for now.
                      // Simpler: just generate based on bits, don't filter resource locations here.
                  }
                  r_idx++;
              }

              all_possible_states.push_back(createStateVector(x, y, res_status));
              valid_state_count++;
         } // res_combo
     }} // y, x
    if(verbose) std::cout << "Generated " << valid_state_count << " possible valid states." << std::endl;


    // Sample states for metrics
    std::vector<std::vector<int>> sampled_states_for_metrics;
    if (collect_metrics && sample_num_states_metrics > 0 && !all_possible_states.empty()) {
        sampled_states_for_metrics.reserve(sample_num_states_metrics);
        std::vector<int> indices(all_possible_states.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), env_gen);
        for (int i = 0; i < std::min((int)all_possible_states.size(), sample_num_states_metrics); ++i) {
            sampled_states_for_metrics.push_back(all_possible_states[indices[i]]);
        }
        if (verbose) std::cout << "Sampled " << sampled_states_for_metrics.size() << " states for metrics." << std::endl;
    }

    // --- RNG Setup ---
    std::random_device rd_agent;
    std::mt19937 agent_gen(rd_agent());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::uniform_int_distribution<> action_dis(0, NUM_ACTIONS - 1);


    // --- Metrics Initialization ---
    if (collect_metrics) {
        int num_metric_points = NUM_EPISODES / METRICS_INTERVAL + 1;
        metrics.avg_q_values.reserve(num_metric_points);
        metrics.avg_decision_dag_sizes.reserve(num_metric_points);
        metrics.episode_times.reserve(NUM_EPISODES + 1);
        metrics.terminal_state_visits.reserve(NUM_EPISODES + 1);
        metrics.bellman_errors.reserve(num_metric_points);
        metrics.memory_usage.reserve(num_metric_points);
        metrics.avg_resources_collected_per_ep.reserve(NUM_EPISODES + 1); // Reserve for new metric
    }

    // --- Failure Counters ---
    int createStateBDD_failures = 0; int evaluateADD_failures = 0; int addIte_failures = 0;
    int addConst_failures = 0; int bddToAdd_failures = 0;

    // --- Learning Loop ---
    if (verbose) std::cout << "\nStarting learning loop..." << std::endl;
    const int PRINT_INTERVAL = std::max(1, NUM_EPISODES / 10);
    const int MAX_STEPS_TO_PRINT = 5;

    for (int episode = 0; episode < NUM_EPISODES; episode++) {
        auto episode_start = std::chrono::high_resolution_clock::now();
        int terminal_visits_in_episode = 0;
        int resources_collected_in_episode = 0;
        int total_steps_in_episode = 0;

        // --- Initial State ---
        // Start at a random non-obstacle, non-resource location with all resources present
        int start_x, start_y;
        std::vector<bool> initial_res_status(NUM_RESOURCES, true); // All present
        do {
            start_x = std::uniform_int_distribution<>(0, GRID_SIZE_X - 1)(agent_gen);
            start_y = std::uniform_int_distribution<>(0, GRID_SIZE_Y - 1)(agent_gen);
        } while (isObstacle(start_x, start_y) || resource_locations.count({start_x, start_y}));
        std::vector<int> state = createStateVector(start_x, start_y, initial_res_status);
        // --- End Initial State ---


        bool print_this_episode = verbose && (episode < 1 || (episode + 1) % PRINT_INTERVAL == 0);
        if (print_this_episode && episode > 0) std::cout << std::endl;
        if (print_this_episode) std::cout << "--- Episode " << (episode + 1) << " Start: " << stateToString(state) << " ---" << std::endl;


        const int MAX_EPISODE_STEPS = (GRID_SIZE_X * GRID_SIZE_Y) * (NUM_RESOURCES + 1); // Heuristic limit

        while (total_steps_in_episode < MAX_EPISODE_STEPS) {
             if (isTerminalState(state)) {
                 terminal_visits_in_episode++;
                 if (print_this_episode && total_steps_in_episode < MAX_STEPS_TO_PRINT) {
                     std::cout << "  Step " << total_steps_in_episode +1 << ": Reached Terminal State " << stateToString(state) << std::endl;
                 }
                 break; // End episode
             }

            total_steps_in_episode++;
            bool print_this_step = print_this_episode && total_steps_in_episode <= MAX_STEPS_TO_PRINT;

            // --- Action Selection ---
            Action action;
            if (dis(agent_gen) < EPSILON) {
                action = static_cast<Action>(action_dis(agent_gen)); // Explore
                 if (print_this_step) std::cout << "  Step " << total_steps_in_episode << ": S=" << stateToString(state) << " -> Explore A=" << getActionName(action) << std::endl;
            } else {
                // Exploit
                double maxQ = -std::numeric_limits<double>::infinity();
                action = UP; // Default
                bool found_valid_q = false;
                for (int a_idx = 0; a_idx < NUM_ACTIONS; ++a_idx) {
                     if (!q_functions[a_idx]) continue;
                    double q = evaluateADD(manager, q_functions[a_idx], state);
                    if (q > EVAL_ERROR_THRESHOLD) {
                        if (!found_valid_q || q > maxQ) {
                            maxQ = q;
                            action = static_cast<Action>(a_idx);
                            found_valid_q = true;
                        }
                    } else { evaluateADD_failures++; }
                }
                if (!found_valid_q) { action = static_cast<Action>(action_dis(agent_gen)); } // Fallback if all failed
                 if (print_this_step) std::cout << "  Step " << total_steps_in_episode << ": S=" << stateToString(state) << " -> Exploit A=" << getActionName(action) << (found_valid_q ? " (Q=" + std::to_string(maxQ) + ")" : " (Default)") << std::endl;
            }

            // --- Symbolic Update ---
            auto [next_state, reward] = applyAgentAction(state, action);
            if (print_this_step) {
                std::cout << "      -> S'=" << stateToString(next_state) << ", R=" << reward << std::endl;
            }

            // Count if a resource was collected this step
            if (countCollectedResources(next_state) > countCollectedResources(state)) {
                resources_collected_in_episode++;
            }

            // Find max Q for next state
            double max_next_q = 0.0;
            if (!isTerminalState(next_state)) {
                 max_next_q = -std::numeric_limits<double>::infinity();
                 bool found_valid_next_q = false;
                 for (int next_a_idx = 0; next_a_idx < NUM_ACTIONS; ++next_a_idx) {
                      if (!q_functions[next_a_idx]) continue;
                     double q_next = evaluateADD(manager, q_functions[next_a_idx], next_state);
                     if (q_next > EVAL_ERROR_THRESHOLD) {
                          if (!found_valid_next_q || q_next > max_next_q) {
                               max_next_q = q_next;
                               found_valid_next_q = true;
                          }
                     } else { evaluateADD_failures++; }
                 }
                  if (!found_valid_next_q) max_next_q = 0.0;
                  else if (max_next_q == -std::numeric_limits<double>::infinity()) max_next_q = 0.0;
            }
             if (print_this_step) std::cout << "      MaxQ(S') = " << max_next_q << std::endl;

            // Calculate target
            double target_q_value = reward + GAMMA * max_next_q;

            // Create BDD for current state
            DdNode* stateBdd = createStateBDD(manager, state);
             if (!stateBdd || stateBdd == Cudd_ReadLogicZero(manager)) { createStateBDD_failures++; state = next_state; continue; }

            // Convert BDD to ADD
            DdNode* stateAdd = Cudd_BddToAdd(manager, stateBdd);
             if (!stateAdd) { bddToAdd_failures++; Cudd_RecursiveDeref(manager, stateBdd); state = next_state; continue; }
             Cudd_Ref(stateAdd); Cudd_RecursiveDeref(manager, stateBdd);

            // Get current Q-value
            int action_idx = static_cast<int>(action);
            DdNode* current_q_add = q_functions[action_idx];
            double oldQ_value = evaluateADD(manager, current_q_add, state);
             if (oldQ_value < EVAL_ERROR_THRESHOLD) { evaluateADD_failures++; Cudd_RecursiveDeref(manager, stateAdd); state = next_state; continue; }

            // Calculate new scalar Q
            double newQ_s_value = (1.0 - ALPHA) * oldQ_value + ALPHA * target_q_value;
             if (print_this_step) { std::cout << "      OldQ=" << oldQ_value << ", Target=" << target_q_value << ", NewScalarQ=" << newQ_s_value << std::endl; }

            // Create ADD constant
            DdNode* newQ_s_add = Cudd_addConst(manager, newQ_s_value);
             if (!newQ_s_add) { addConst_failures++; Cudd_RecursiveDeref(manager, stateAdd); state = next_state; continue; }
             Cudd_Ref(newQ_s_add);

            // Update Q-function with ITE
            DdNode* updatedQAdd = Cudd_addIte(manager, stateAdd, newQ_s_add, current_q_add);
             if (!updatedQAdd) { addIte_failures++; Cudd_RecursiveDeref(manager, stateAdd); Cudd_RecursiveDeref(manager, newQ_s_add); state = next_state; continue; }
             Cudd_Ref(updatedQAdd);

            // Clean up intermediates and update Q-function
            Cudd_RecursiveDeref(manager, stateAdd);
            Cudd_RecursiveDeref(manager, newQ_s_add);
            Cudd_RecursiveDeref(manager, q_functions[action_idx]);
            q_functions[action_idx] = updatedQAdd;

            // Move to next state
            state = next_state;

        } // End while steps

         if (print_this_episode && total_steps_in_episode >= MAX_EPISODE_STEPS) {
             std::cout << "  Episode " << (episode + 1) << " finished due to MAX_STEPS." << std::endl;
         }

        // --- Track Metrics ---
        if (collect_metrics) {
            metrics.terminal_state_visits.push_back(terminal_visits_in_episode);
            metrics.avg_resources_collected_per_ep.push_back(static_cast<double>(resources_collected_in_episode)); // Store collected count
            auto episode_end = std::chrono::high_resolution_clock::now();
            metrics.episode_times.push_back(std::chrono::duration<double>(episode_end - episode_start).count());

            if ((episode + 1) % METRICS_INTERVAL == 0 && !sampled_states_for_metrics.empty()) {
                double avg_q = calculateAverageQValue(q_functions, sampled_states_for_metrics);
                double avg_dag = calculateAverageDAGSize(q_functions);
                double bellman_err = calculateBellmanError(q_functions, sampled_states_for_metrics);
                double mem_usage = avg_dag * sizeof(DdNode) / (1024.0 * 1024.0);

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
        } // end collect_metrics

    } // End for episodes

    if (verbose) { /* Print failure counts */ }

    // --- Extract Policy ---
    if (verbose) std::cout << "\nExtracting policy..." << std::endl;
    policy.clear();
    if (!all_possible_states.empty()) {
        for (const auto& s : all_possible_states) {
            std::string stateStr = stateToString(s);
            if (isTerminalState(s)) {
                policy[stateStr] = INVALID_ACTION;
            } else {
                double maxQ = -std::numeric_limits<double>::infinity();
                Action bestAction = UP; // Default
                bool found_best = false;
                for (int a_idx = 0; a_idx < NUM_ACTIONS; ++a_idx) {
                     if (!q_functions[a_idx]) continue;
                    double q = evaluateADD(manager, q_functions[a_idx], s);
                    if (q > EVAL_ERROR_THRESHOLD) {
                        if (!found_best || q > maxQ) {
                            maxQ = q;
                            bestAction = static_cast<Action>(a_idx);
                            found_best = true;
                        }
                    }
                }
                policy[stateStr] = found_best ? bestAction : UP; // Assign best or default UP
            }
        }
         if (verbose) std::cout << "Policy extracted for " << policy.size() << " states." << std::endl;
    } else {
         if (verbose) std::cout << "No valid states to extract policy from." << std::endl;
    }

    // --- Clean Up ---
    if (verbose) std::cout << "Cleaning up Q-functions and terminal BDD..." << std::endl;
    for (auto& q_add : q_functions) if (q_add) Cudd_RecursiveDeref(manager, q_add);
    q_functions.clear();
    if (terminal_bdd) Cudd_RecursiveDeref(manager, terminal_bdd);
    terminal_bdd = nullptr;

    auto global_end_time = std::chrono::high_resolution_clock::now();
    if (verbose) std::cout << "Symbolic Q-Learning function finished in " << std::chrono::duration<double>(global_end_time - global_start_time).count() << " seconds." << std::endl;

    return policy;
}


// --- Simulation ---
void runSimulation(const std::map<std::string, Action>& policy, int numTrials, bool verbose = true) {
    if (policy.empty()) { std::cerr << "Cannot run simulation: Policy map is empty." << std::endl; return; }

    long long totalSteps = 0;
    int successCount = 0;
    int timeoutCount = 0;
    int policyErrorCount = 0;
    double totalResourcesCollected = 0;

    std::random_device rd_sim;
    std::mt19937 sim_gen(rd_sim());

    if (verbose) std::cout << "\n--- Starting Simulation (" << numTrials << " trials) ---" << std::endl;

    for (int trial = 0; trial < numTrials; ++trial) {
        // Initial state: random non-obstacle, non-resource start, all resources present
        int start_x, start_y;
        std::vector<bool> initial_res_status(NUM_RESOURCES, true);
        do {
            start_x = std::uniform_int_distribution<>(0, GRID_SIZE_X - 1)(sim_gen);
            start_y = std::uniform_int_distribution<>(0, GRID_SIZE_Y - 1)(sim_gen);
        } while (isObstacle(start_x, start_y) || resource_locations.count({start_x, start_y}));
        std::vector<int> state = createStateVector(start_x, start_y, initial_res_status);

        int steps = 0;
        const int MAX_STEPS = (GRID_SIZE_X * GRID_SIZE_Y) * (NUM_RESOURCES + 1) * 2; // Generous limit
        bool reachedTerminal = false;
        bool errorOccurred = false;

        if (verbose && trial < 3) std::cout << "\n--- Trial " << (trial + 1) << " Start: " << stateToString(state) << " ---" << std::endl;


        while (steps < MAX_STEPS) {
            if (isTerminalState(state)) {
                reachedTerminal = true; successCount++; totalSteps += steps;
                totalResourcesCollected += NUM_RESOURCES; // Reached terminal means all collected
                if (verbose && trial < 3) std::cout << "  Terminal reached in " << steps << " steps!" << std::endl;
                break;
            }
            steps++;

            std::string stateStr = stateToString(state);
            auto policy_it = policy.find(stateStr);

            if (policy_it == policy.end()) {
                if (verbose) std::cerr << "Error: State " << stateStr << " not in policy map (Trial " << trial + 1 << ")." << std::endl;
                policyErrorCount++; errorOccurred = true; break;
            }
            Action action = policy_it->second;
            if (action == INVALID_ACTION) {
                 if (verbose) std::cerr << "Warning: Policy has INVALID_ACTION for non-terminal state " << stateStr << " (Trial " << trial+1 << ")." << std::endl;
                 policyErrorCount++; errorOccurred = true; break;
            }

            if (verbose && trial < 3) std::cout << "  Sim Step " << steps << ": S=" << stateStr << " -> A=" << getActionName(action) << std::endl;

            auto [next_state, reward] = applyAgentAction(state, action);
            state = next_state; // Update state

            if (verbose && trial < 3) std::cout << "      -> S'=" << stateToString(state) << ", R=" << reward << std::endl;

        } // End while steps

        if (!reachedTerminal && !errorOccurred && steps >= MAX_STEPS) {
            timeoutCount++;
            totalResourcesCollected += countCollectedResources(state); // Count how many were collected before timeout
            if (verbose && trial < 3) std::cout << "  Timeout after " << MAX_STEPS << " steps." << std::endl;
        } else if (errorOccurred) {
             totalResourcesCollected += countCollectedResources(state); // Count resources before error
        }
    } // End for trials

    // --- Print Simulation Results ---
    double successRate = (numTrials > 0) ? static_cast<double>(successCount) / numTrials * 100.0 : 0.0;
    double avgSteps = successCount > 0 ? static_cast<double>(totalSteps) / successCount : 0.0;
    double avgResources = numTrials > 0 ? totalResourcesCollected / numTrials : 0.0;

    std::cout << "\n--- Simulation Results (" << numTrials << " trials) ---" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Success Rate (All Resources): " << successCount << "/" << numTrials << " (" << successRate << "%)" << std::endl;
    std::cout << "  Timeouts:                     " << timeoutCount << "/" << numTrials << std::endl;
    std::cout << "  Policy Errors:                " << policyErrorCount << "/" << numTrials << std::endl;
    std::cout << "  Avg Resources Collected:      " << avgResources << "/" << NUM_RESOURCES << std::endl;
    if (successCount > 0) {
        std::cout << "  Avg Steps (success):          " << avgSteps << std::endl;
    } else { std::cout << "  No trials finished successfully." << std::endl; }
    std::cout << "------------------------------------------" << std::endl;
}


// --- Visualization, Policy Printing, CSV Saving ---

void visualizeGrid(const std::map<std::string, Action>& policy, const std::vector<int>& current_sim_state) {
    std::cout << "\n--- Grid World Visualization ---" << std::endl;
    int agent_x = -1, agent_y = -1;
    std::vector<bool> sim_res_status(NUM_RESOURCES, false);
    if (current_sim_state.size() == TOTAL_STATE_BITS) {
        agent_x = getCoordValue(current_sim_state, 0);
        agent_y = getCoordValue(current_sim_state, 1);
        for(int i=0; i<NUM_RESOURCES; ++i) sim_res_status[i] = getResourceStatus(current_sim_state, i);
    }

    // Header
    std::cout << "  "; for (int x = 0; x < GRID_SIZE_X; ++x) std::cout << "  " << x << " "; std::cout << std::endl;
    std::cout << "  +"; for (int x = 0; x < GRID_SIZE_X; ++x) std::cout << "---+"; std::cout << std::endl;

    // Grid content
    for (int y = 0; y < GRID_SIZE_Y; ++y) {
        std::cout << y << " |";
        for (int x = 0; x < GRID_SIZE_X; ++x) {
            char cell = ' '; std::string direction = " ";
            bool is_resource_loc = resource_locations.count({x, y});
            int resource_idx_at_loc = -1;
            if (is_resource_loc) {
                int idx = 0;
                for(const auto& loc : resource_locations) { if (loc.first == x && loc.second == y) { resource_idx_at_loc = idx; break; } idx++; }
            }

            if (x == agent_x && y == agent_y) cell = 'A'; // Agent
            else if (isObstacle(x, y)) cell = '#'; // Obstacle
            else if (is_resource_loc) cell = '$'; // Resource location
            else cell = '.'; // Empty

            // Show policy direction for a representative state (e.g., all resources present)
            std::vector<bool> representative_res_status(NUM_RESOURCES, true);
            // Or use current sim state if available
            if (!current_sim_state.empty()) representative_res_status = sim_res_status;

            std::vector<int> representative_state = createStateVector(x, y, representative_res_status);
            std::string stateStr = stateToString(representative_state);
            auto it = policy.find(stateStr);

            if (!isObstacle(x,y) && it != policy.end()) {
                 Action action = it->second;
                 if (isTerminalState(representative_state)) direction = "*"; // Terminal state
                 else {
                     switch (action) {
                         case UP: direction = "^"; break; case RIGHT: direction = ">"; break;
                         case DOWN: direction = "v"; break; case LEFT: direction = "<"; break;
                         case INVALID_ACTION: direction = "*"; break; default: direction = "?"; break;
                     }
                 }
            }

            // Modify display if resource is collected in sim state
            if (cell == '$' && resource_idx_at_loc != -1 && !sim_res_status[resource_idx_at_loc]) {
                 cell = 'o'; // Indicate collected resource location
            }

            std::cout << cell << direction << (cell == 'A' || cell == '#' ? cell : '.') << "|"; // Center display
        }
        std::cout << std::endl;
        std::cout << "  +"; for (int x = 0; x < GRID_SIZE_X; ++x) std::cout << "---+"; std::cout << std::endl;
    }

    // Legend
    std::cout << "\nLegend: A=Agent, #=Obstacle, $=Resource, o=Collected Loc, .=Empty" << std::endl;
    std::cout << "  Direction arrows (^>v<) show policy for current resource status." << std::endl;
    std::cout << "  * indicates policy for terminal state." << std::endl;
    std::cout << "-----------------------------" << std::endl;
}


void printPolicy(const std::map<std::string, Action>& policy) {
    std::cout << "\n--- Learned Policy ---" << std::endl;
    int stateWidth = 15 + NUM_RESOURCES; // Adjust width based on state string
    std::cout << std::setw(stateWidth) << std::left << "State (Pos)|R:Status"
              << std::setw(10) << std::left << "Action" << std::endl;
    std::cout << std::string(stateWidth + 10, '-') << std::endl;

    std::vector<std::string> stateStrs; stateStrs.reserve(policy.size());
    for (const auto& entry : policy) stateStrs.push_back(entry.first);
    std::sort(stateStrs.begin(), stateStrs.end());

    int printedCount = 0; const int MAX_PRINT_POLICY = 50;
    int terminalCount = 0; int nonTerminalCount = 0;

    for (const auto& stateStr : stateStrs) {
        Action action = policy.at(stateStr);
        if (action == INVALID_ACTION) { terminalCount++; }
        else {
            nonTerminalCount++;
            if (printedCount < MAX_PRINT_POLICY) {
                std::cout << std::setw(stateWidth) << std::left << stateStr
                          << std::setw(10) << std::left << getActionName(action) << std::endl;
                printedCount++;
            }
        }
    }
    if (nonTerminalCount > printedCount) std::cout << "... (omitting " << (nonTerminalCount - printedCount) << " non-terminal states)" << std::endl;
    if (terminalCount > 0) std::cout << std::setw(stateWidth) << std::left << "<Terminal States>" << std::setw(10) << std::left << "Terminal" << std::endl;
    std::cout << std::string(stateWidth + 10, '-') << std::endl;
    std::cout << "Total states in policy: " << policy.size() << " (" << nonTerminalCount << " non-terminal, " << terminalCount << " terminal)" << std::endl;
    std::cout << "---------------------------" << std::endl;
}

void saveMetricsToCSV() {
     if (metrics.avg_q_values.empty() && metrics.avg_resources_collected_per_ep.empty()) { // Check if any metrics were collected
         std::cout << "No metrics collected, CSV files not generated." << std::endl;
         return;
     }
    std::cout << "Saving metrics to CSV files..." << std::endl;

    // Use prefix "resource" for filenames
    const std::string prefix = "add_q_resource_";

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
     // Save avg resources collected
      if (!metrics.avg_resources_collected_per_ep.empty()) {
         std::ofstream file(prefix + "resources_collected.csv"); file << "Episode,AvgResourcesCollected\n";
         for (size_t i = 0; i < metrics.avg_resources_collected_per_ep.size(); ++i) file << i + 1 << "," << metrics.avg_resources_collected_per_ep[i] << "\n";
     }

    std::cout << "Metrics saved successfully." << std::endl;
}

// --- Help and Main Function ---
void printHelp() {
    std::cout << "Usage: add_q_resource_collector [options]\n";
    std::cout << "Options:\n";
    std::cout << "  -gx N         Set grid width (default: 5)\n";
    std::cout << "  -gy N         Set grid height (default: 5)\n";
    std::cout << "  -obs N        Number of obstacles (default: 3)\n";
    std::cout << "  -res N        Number of resources (default: 3)\n";
    std::cout << "  -e N          Episodes (default: 10000)\n";
    std::cout << "  -a F          Alpha (default: 0.1)\n";
    std::cout << "  -g F          Gamma (default: 0.99)\n";
    std::cout << "  -eps F        Epsilon (default: 0.2)\n";
    std::cout << "  -rres F       Resource collection reward (default: 50.0)\n";
    std::cout << "  -pobst F      Obstacle penalty (negative, default: -5.0)\n";
    std::cout << "  -pcost F      Step cost (negative, default: -0.1)\n";
    std::cout << "  -sim N        Simulation trials (default: 0)\n";
    std::cout << "  -metrics N    Collect metrics (sample N states, default 1000)\n";
    std::cout << "  -v            Verbose mode\n";
    std::cout << "  -h            Print this help message\n";
}

int main(int argc, char* argv[]) {
    bool verbose = false;
    int numSimTrials = 0;
    bool collect_and_save_metrics = false;
    int metrics_sample_size = 1000;

    // --- Argument Parsing ---
     for (int i = 1; i < argc; ++i) {
         std::string arg = argv[i];
         try {
             if (arg == "-v") verbose = true;
             else if (arg == "-metrics" && i + 1 < argc) { collect_and_save_metrics = true; metrics_sample_size = std::stoi(argv[++i]); if(metrics_sample_size<0) metrics_sample_size=0;}
             else if (arg == "-metrics") { collect_and_save_metrics = true; }
             else if (arg == "-gx" && i + 1 < argc) { GRID_SIZE_X = std::stoi(argv[++i]); if(GRID_SIZE_X<=1) {std::cerr<<"Grid X must be > 1\n"; return 1;}}
             else if (arg == "-gy" && i + 1 < argc) { GRID_SIZE_Y = std::stoi(argv[++i]); if(GRID_SIZE_Y<=1) {std::cerr<<"Grid Y must be > 1\n"; return 1;}}
             else if (arg == "-obs" && i + 1 < argc) { NUM_OBSTACLES = std::stoi(argv[++i]); if(NUM_OBSTACLES<0) NUM_OBSTACLES=0;}
             else if (arg == "-res" && i + 1 < argc) { NUM_RESOURCES = std::stoi(argv[++i]); if(NUM_RESOURCES<=0) {std::cerr<<"Resources must be > 0\n"; return 1;}}
             else if (arg == "-e" && i + 1 < argc) { NUM_EPISODES = std::stoi(argv[++i]); if(NUM_EPISODES<=0) {std::cerr<<"Episodes must be > 0\n"; return 1;}}
             else if (arg == "-a" && i + 1 < argc) { ALPHA = std::stod(argv[++i]); if(ALPHA<=0 || ALPHA>1) {std::cerr<<"Alpha must be (0,1]\n"; return 1;}}
             else if (arg == "-g" && i + 1 < argc) { GAMMA = std::stod(argv[++i]); if(GAMMA<0 || GAMMA>1) {std::cerr<<"Gamma must be [0,1]\n"; return 1;}}
             else if (arg == "-eps" && i + 1 < argc) { EPSILON = std::stod(argv[++i]); if(EPSILON<0 || EPSILON>1) {std::cerr<<"Epsilon must be [0,1]\n"; return 1;}}
             else if (arg == "-rres" && i + 1 < argc) { RESOURCE_REWARD = std::stod(argv[++i]); }
             else if (arg == "-pobst" && i + 1 < argc) { OBSTACLE_COST = std::stod(argv[++i]); } // Note: This is added to reward, so should be negative
             else if (arg == "-pcost" && i + 1 < argc) { MOVE_COST = std::stod(argv[++i]); } // Note: This is added to reward, so should be negative
             else if (arg == "-sim" && i + 1 < argc) { numSimTrials = std::stoi(argv[++i]); if(numSimTrials<0) numSimTrials=0;}
             else if (arg == "-h") { printHelp(); return 0; }
             else { std::cerr << "Unknown option: " << arg << "\n"; printHelp(); return 1; }
         } catch (const std::exception& e) { std::cerr << "Error parsing argument for " << arg << ": " << e.what() << std::endl; return 1;}
     }

    // --- Calculate Derived Parameters ---
    BITS_PER_COORD = static_cast<int>(std::ceil(std::log2(std::max(GRID_SIZE_X, GRID_SIZE_Y))));
     if (BITS_PER_COORD == 0 && std::max(GRID_SIZE_X, GRID_SIZE_Y) > 0) BITS_PER_COORD = 1;
    LOCATION_BITS = BITS_PER_COORD * 2;
    RESOURCE_BITS = NUM_RESOURCES;
    TOTAL_STATE_BITS = LOCATION_BITS + RESOURCE_BITS;
     if (TOTAL_STATE_BITS <= 0) { std::cerr << "Error: TOTAL_STATE_BITS is not positive." << std::endl; return 1;}
     if (TOTAL_STATE_BITS > Cudd_ReadMaxIndex()) { std::cerr << "Error: Required BDD variables (" << TOTAL_STATE_BITS << ") exceed CUDD limit." << std::endl; return 1; }
     // Ensure NUM_OBSTACLES is not excessive
     if (NUM_OBSTACLES >= GRID_SIZE_X * GRID_SIZE_Y - NUM_RESOURCES - 1) {
          std::cerr << "Warning: High number of obstacles may make the environment unsolvable." << std::endl;
          NUM_OBSTACLES = std::max(0, GRID_SIZE_X * GRID_SIZE_Y - NUM_RESOURCES - 1); // Cap obstacles
     }


    // --- Print Parameters ---
    std::cout << "--- Resource Collector Parameters ---" << std::endl;
    std::cout << "  Grid Size: " << GRID_SIZE_X << "x" << GRID_SIZE_Y << std::endl;
    std::cout << "  Obstacles: " << NUM_OBSTACLES << ", Resources: " << NUM_RESOURCES << std::endl;
    std::cout << "  Bits/Coord: " << BITS_PER_COORD << ", Location Bits: " << LOCATION_BITS << ", Resource Bits: " << RESOURCE_BITS << ", Total Bits: " << TOTAL_STATE_BITS << std::endl;
    std::cout << "  RL Params: Episodes=" << NUM_EPISODES << ", Alpha=" << ALPHA << ", Gamma=" << GAMMA << ", Epsilon=" << EPSILON << std::endl;
    std::cout << "  Rewards: Resource=" << RESOURCE_REWARD << ", Obstacle=" << OBSTACLE_COST << ", Step=" << MOVE_COST << std::endl;
    std::cout << "------------------------------------" << std::endl;

    // --- Initialize Environment ---
    initializeEnvironment();
     if(verbose) {
          std::cout << "Resource Locations: ";
          for(const auto& loc : resource_locations) std::cout << "(" << loc.first << "," << loc.second << ") ";
          std::cout << std::endl;
     }


    // --- Initialize CUDD ---
    manager = Cudd_Init(TOTAL_STATE_BITS, 0, CUDD_UNIQUE_SLOTS, CUDD_CACHE_SLOTS, 0);
    if (!manager) { std::cerr << "CUDD initialization failed.\n"; return 1; }

    // --- Create BDD Variables ---
    vars.resize(TOTAL_STATE_BITS);
    bool vars_ok = true;
    for (int i = 0; i < TOTAL_STATE_BITS; ++i) {
        vars[i] = Cudd_bddIthVar(manager, i);
        if (!vars[i]) { vars_ok = false; std::cerr << "BDD variable creation failed for index " << i << std::endl; break; }
    }
    if (!vars_ok) { std::cerr << "Aborting due to BDD variable failure.\n"; Cudd_Quit(manager); return 1; }
    if (verbose) std::cout << "Created " << TOTAL_STATE_BITS << " BDD variables." << std::endl;

    // --- Run Learning ---
    std::map<std::string, Action> policy = symbolicQLearning(verbose, metrics_sample_size, collect_and_save_metrics);

    // --- Post-Learning ---
     if (policy.empty() && TOTAL_STATE_BITS > 0) { // Check if policy is empty but state space exists
         std::cerr << "Learning failed or resulted in an empty policy for a non-trivial state space." << std::endl;
         Cudd_Quit(manager);
         return 1;
     } else if (policy.empty()) {
          std::cout << "Learning resulted in empty policy (likely trivial state space)." << std::endl;
     }


    printPolicy(policy);

    // Visualize initial grid (optional)
    if(verbose && !policy.empty()) {
         // Create a representative initial state for visualization
          int start_x, start_y;
          std::vector<bool> initial_res_status(NUM_RESOURCES, true);
          do {
              start_x = std::uniform_int_distribution<>(0, GRID_SIZE_X - 1)(env_gen); // Use global gen
              start_y = std::uniform_int_distribution<>(0, GRID_SIZE_Y - 1)(env_gen);
          } while (isObstacle(start_x, start_y) || resource_locations.count({start_x, start_y}));
          std::vector<int> vis_state = createStateVector(start_x, start_y, initial_res_status);
          visualizeGrid(policy, vis_state);
    }


    // --- Run Simulation ---
    if (numSimTrials > 0 && !policy.empty()) {
        runSimulation(policy, numSimTrials, verbose);
    } else if (numSimTrials > 0 && policy.empty()) {
         std::cout << "Skipping simulation as policy is empty." << std::endl;
    } else {
        std::cout << "\nRun with '-sim N' to simulate the learned policy." << std::endl;
    }

    // --- Save Metrics ---
    if (collect_and_save_metrics) {
        saveMetricsToCSV();
    }

    // --- Clean Up CUDD ---
    if (verbose) std::cout << "Cleaning up CUDD manager..." << std::endl;
    vars.clear();
    int check = Cudd_CheckZeroRef(manager);
    if (check > 0 && verbose) std::cerr << "Warning: " << check << " CUDD nodes dangling." << std::endl;
    Cudd_Quit(manager);
    manager = nullptr;

    return 0;
}