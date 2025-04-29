//Author - Aadithya Srinivasan Anand

// Efficient ADD-Q algorithm implementation for grid world navigation
// Demonstrates the benefits of symbolic representation and compressed Q-functions

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
#include <fstream>  // Added for file I/O
#include <limits>
#include <algorithm>
#include <chrono>
#include <random>
#include <numeric>
#include <functional>
#include <memory>

// Include CUDD header
#include "cudd.h"
#include "cuddInt.h"

// --- Grid World Parameters ---
int GRID_SIZE_X = 8;        // Width of grid
int GRID_SIZE_Y = 8;        // Height of grid
int NUM_OBSTACLES = 10;     // Number of obstacles

// Agent location encoding - log2 encoding for efficiency
const int BITS_PER_COORD = 3;  // Enough for 8x8 grid (needs to be adjusted for larger grids)
const int TOTAL_LOCATION_BITS = BITS_PER_COORD * 2;  // x and y coordinates
const size_t ESTIMATED_BYTES_PER_NODE = 32; // Informed estimate for CUDD internal node size

// Define obstacle patterns (true = obstacle, false = clear)
std::vector<std::vector<bool>> obstacles;

// Location of goal state
int GOAL_X = GRID_SIZE_X - 1;
int GOAL_Y = GRID_SIZE_Y - 1;

// RL Parameters
double GAMMA = 0.99;          // Higher discount factor for grid world
double EPSILON = 0.2;
double ALPHA = 0.1;
int NUM_EPISODES = 5000;
double GOAL_REWARD = 100.0;
double MOVE_COST = -0.1;
double OBSTACLE_COST = -5.0;

// Action Definitions
enum Action { INVALID_ACTION = -1, UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3 };
const int NUM_ACTIONS = 4;
const double EVAL_ERROR_THRESHOLD = -9999.0;

// Metrics for ADD-Q analysis
struct AddQMetrics {
    std::vector<double> avg_q_values;          // Average Q-value during learning
    std::vector<int> avg_decision_dag_sizes;   // Average BDD/ADD node count 
    std::vector<double> episode_times;         // Time per episode
    std::vector<int> goal_visits;              // Goal state visits per episode
    std::vector<double> bellman_errors;        // Bellman error per episode
    std::vector<double> memory_usage;          // Memory usage per episode
    std::vector<double> avg_path_lengths;      // Average path length to goal
};

// Globals
DdManager* manager = nullptr;
DdNode* goal_bdd = nullptr;
std::vector<DdNode*> vars;
std::mt19937 env_gen;
AddQMetrics metrics;

// Forward declarations
class StateCache;
DdNode* createGoalBDD();
bool isGoalState(const std::vector<int>& state);
void printPolicy(const std::map<std::string, Action>& policy);
void initializeObstacles();

// --- Helper Functions ---

// Print detailed information about a CUDD node
void printNodeInfo(const char* name, DdNode* node, bool isAdd = false) {
    if (!manager) return;
    if (!node) { 
        std::cout << (isAdd ? "ADD " : "BDD ") << name << ": NULL node" << std::endl; 
        return; 
    }
    
    DdNode* regular_node = Cudd_Regular(node);
    std::cout << (isAdd ? "ADD " : "BDD ") << name << ": " << (void*)regular_node 
              << (Cudd_IsComplement(node) ? "'" : "")
              << ", index = " << (Cudd_IsConstant(regular_node) ? -1 : Cudd_NodeReadIndex(regular_node))
              << ", val = " << (Cudd_IsConstant(regular_node) ? std::to_string(Cudd_V(regular_node)) : "N/A")
              << ", DagSize = " << Cudd_DagSize(node) << std::endl;
}

// Get variable index in the CUDD manager
int getVarIndex(int coord_type, int bit_idx) {
    assert(coord_type >= 0 && coord_type < 2);  // 0 for x, 1 for y
    assert(bit_idx >= 0 && bit_idx < BITS_PER_COORD);
    return coord_type * BITS_PER_COORD + bit_idx;
}

// Extract coordinate value from state vector
int getCoordValue(const std::vector<int>& state, int coord_type) {
    if (state.empty() || coord_type < 0 || coord_type >= 2) return -1;
    
    int value = 0;
    int base_idx = coord_type * BITS_PER_COORD;
    
    for (int b = 0; b < BITS_PER_COORD; ++b) {
        int var_idx = base_idx + b;
        if (var_idx >= (int)state.size()) return -1;
        if (state[var_idx] == 1) { value |= (1 << b); }
    }
    
    return value;
}

// Set coordinate value in state vector
void setCoordValue(std::vector<int>& state, int coord_type, int value) {
    if (state.empty() || coord_type < 0 || coord_type >= 2) return;
    
    int base_idx = coord_type * BITS_PER_COORD;
    int max_value = (1 << BITS_PER_COORD) - 1;
    
    // Ensure value is within bounds
    value = std::min(value, max_value);
    value = std::max(value, 0);
    
    int temp_value = value;
    for (int b = 0; b < BITS_PER_COORD; ++b) {
        int var_idx = base_idx + b;
        if (var_idx >= (int)state.size()) return;
        state[var_idx] = (temp_value % 2);
        temp_value /= 2;
    }
}

// Convert coordinates to state vector
std::vector<int> coordsToState(int x, int y) {
    std::vector<int> state(TOTAL_LOCATION_BITS, 0);
    setCoordValue(state, 0, x);
    setCoordValue(state, 1, y);
    return state;
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
    if (x < 0 || x >= GRID_SIZE_X || y < 0 || y >= GRID_SIZE_Y) return true;  // Grid boundaries as obstacles
    return obstacles[y][x];
}

// Apply an agent action to the state and return the result
std::pair<std::vector<int>, bool> applyAgentAction(const std::vector<int>& state, Action action) {
    int x = getCoordValue(state, 0);
    int y = getCoordValue(state, 1);
    
    if (x < 0 || y < 0 || x >= GRID_SIZE_X || y >= GRID_SIZE_Y) {
        return {state, false};  // Invalid state
    }
    
    // Calculate new position based on action
    int new_x = x;
    int new_y = y;
    
    switch(action) {
        case UP:    new_y = std::max(0, y - 1); break;
        case RIGHT: new_x = std::min(GRID_SIZE_X - 1, x + 1); break;
        case DOWN:  new_y = std::min(GRID_SIZE_Y - 1, y + 1); break;
        case LEFT:  new_x = std::max(0, x - 1); break;
        default: break;
    }
    
    // Check if the new position has an obstacle
    bool hit_obstacle = isObstacle(new_x, new_y);
    if (hit_obstacle) {
        new_x = x;  // Stay in place if hitting an obstacle
        new_y = y;
    }
    
    std::vector<int> new_state = coordsToState(new_x, new_y);
    return {new_state, !hit_obstacle};
}

// State caching for BDD operations to improve performance
class StateCache {
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
    
    std::string stateToString(const std::vector<int>& state) {
        std::string result;
        result.reserve(state.size());
        for (int bit : state) {
            result.push_back('0' + bit);
        }
        return result;
    }
    
    DdNode* createStateBDD(const std::vector<int>& state) {
        std::string key = stateToString(state);
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

// Create BDD pattern for a specific coordinate value
DdNode* createCoordValueBDD(int coord_type, int target_value) {
    if (!manager) return nullptr;
    
    DdNode* valueBdd = Cudd_ReadOne(manager);
    Cudd_Ref(valueBdd);
    
    int temp_val = target_value;
    for (int b = 0; b < BITS_PER_COORD; ++b) {
        int var_idx = getVarIndex(coord_type, b);
        bool bit_is_set = (temp_val % 2 != 0);
        temp_val /= 2;
        
        if (var_idx < 0 || var_idx >= (int)vars.size() || !vars[var_idx]) {
            Cudd_RecursiveDeref(manager, valueBdd);
            return nullptr;
        }
        
        DdNode* literal = Cudd_NotCond(vars[var_idx], !bit_is_set);
        DdNode* tmp = Cudd_bddAnd(manager, valueBdd, literal);
        
        if (!tmp) {
            Cudd_RecursiveDeref(manager, valueBdd);
            return nullptr;
        }
        
        Cudd_Ref(tmp);
        Cudd_RecursiveDeref(manager, valueBdd);
        valueBdd = tmp;
    }
    
    return valueBdd;
}

// Create goal BDD for the specific goal state
DdNode* createGoalBDD() {
    if (!manager || vars.empty()) return nullptr;
    
    // Create BDD for x = GOAL_X
    DdNode* goal_x_bdd = createCoordValueBDD(0, GOAL_X);
    if (!goal_x_bdd) return nullptr;
    
    // Create BDD for y = GOAL_Y
    DdNode* goal_y_bdd = createCoordValueBDD(1, GOAL_Y);
    if (!goal_y_bdd) {
        Cudd_RecursiveDeref(manager, goal_x_bdd);
        return nullptr;
    }
    
    // Combine: goal = (x == GOAL_X) AND (y == GOAL_Y)
    DdNode* goal_bdd = Cudd_bddAnd(manager, goal_x_bdd, goal_y_bdd);
    if (!goal_bdd) {
        Cudd_RecursiveDeref(manager, goal_x_bdd);
        Cudd_RecursiveDeref(manager, goal_y_bdd);
        return nullptr;
    }
    Cudd_Ref(goal_bdd);
    
    Cudd_RecursiveDeref(manager, goal_x_bdd);
    Cudd_RecursiveDeref(manager, goal_y_bdd);
    
    return goal_bdd;
}

// Create obstacle BDD for states that contain obstacles
DdNode* createObstacleBDD() {
    if (!manager || vars.empty()) return nullptr;
    
    DdNode* obstacle_bdd = Cudd_ReadLogicZero(manager);
    Cudd_Ref(obstacle_bdd);
    
    for (int y = 0; y < GRID_SIZE_Y; ++y) {
        for (int x = 0; x < GRID_SIZE_X; ++x) {
            if (isObstacle(x, y)) {
                // Create BDD for this obstacle location
                DdNode* x_bdd = createCoordValueBDD(0, x);
                if (!x_bdd) continue;
                
                DdNode* y_bdd = createCoordValueBDD(1, y);
                if (!y_bdd) {
                    Cudd_RecursiveDeref(manager, x_bdd);
                    continue;
                }
                
                // Combine x and y for this obstacle
                DdNode* loc_bdd = Cudd_bddAnd(manager, x_bdd, y_bdd);
                if (!loc_bdd) {
                    Cudd_RecursiveDeref(manager, x_bdd);
                    Cudd_RecursiveDeref(manager, y_bdd);
                    continue;
                }
                Cudd_Ref(loc_bdd);
                Cudd_RecursiveDeref(manager, x_bdd);
                Cudd_RecursiveDeref(manager, y_bdd);
                
                // OR with the main obstacle BDD
                DdNode* tmp = Cudd_bddOr(manager, obstacle_bdd, loc_bdd);
                if (!tmp) {
                    Cudd_RecursiveDeref(manager, loc_bdd);
                    continue;
                }
                Cudd_Ref(tmp);
                Cudd_RecursiveDeref(manager, obstacle_bdd);
                Cudd_RecursiveDeref(manager, loc_bdd);
                obstacle_bdd = tmp;
            }
        }
    }
    
    return obstacle_bdd;
}

// Check if a state is a goal state
bool isGoalState(const std::vector<int>& state) {
    if (!manager || !goal_bdd) { return false; }
    if (Cudd_IsConstant(goal_bdd)) { return (goal_bdd == Cudd_ReadOne(manager)); }
    if ((int)state.size() != TOTAL_LOCATION_BITS) return false;
    
    int* assignment = new int[state.size()];
    for (size_t i = 0; i < state.size(); ++i) assignment[i] = state[i];
    
    DdNode* evalNode = Cudd_Eval(manager, goal_bdd, assignment);
    delete[] assignment;
    
    return (evalNode != nullptr && evalNode == Cudd_ReadOne(manager));
}

// Evaluate an ADD for a specific state
double evaluateADD(DdManager* mgr, DdNode* add, const std::vector<int>& state) {
    if (!mgr || !add) return EVAL_ERROR_THRESHOLD;
    
    int* assignment = new int[state.size()];
    for (size_t i = 0; i < state.size(); ++i) {
        assignment[i] = state[i];
    }
    
    DdNode* evalNode = Cudd_Eval(mgr, add, assignment);
    delete[] assignment;
    
    if (evalNode == NULL || !Cudd_IsConstant(evalNode)) {
        return EVAL_ERROR_THRESHOLD;
    }
    
    return Cudd_V(evalNode);
}

// Generate a random valid state for the grid world
std::vector<int> generateRandomState(std::mt19937& gen) {
    std::uniform_int_distribution<> x_dis(0, GRID_SIZE_X - 1);
    std::uniform_int_distribution<> y_dis(0, GRID_SIZE_Y - 1);
    
    int x, y;
    do {
        x = x_dis(gen);
        y = y_dis(gen);
    } while (isObstacle(x, y) || (x == GOAL_X && y == GOAL_Y));
    
    return coordsToState(x, y);
}

// Initialize obstacles in the grid
void initializeObstacles() {
    // Resize obstacles grid
    obstacles.resize(GRID_SIZE_Y, std::vector<bool>(GRID_SIZE_X, false));
    
    // Random obstacles
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> x_dis(0, GRID_SIZE_X - 1);
    std::uniform_int_distribution<> y_dis(0, GRID_SIZE_Y - 1);
    
    int placed = 0;
    while (placed < NUM_OBSTACLES) {
        int x = x_dis(gen);
        int y = y_dis(gen);
        
        // Don't place obstacle at start or goal
        if ((x == 0 && y == 0) || (x == GOAL_X && y == GOAL_Y)) {
            continue;
        }
        
        // Place obstacle if not already one there
        if (!obstacles[y][x]) {
            obstacles[y][x] = true;
            placed++;
        }
    }
    
    // Optionally add some structured obstacles like walls
    // For example, a wall in the middle with a gap
    if (GRID_SIZE_X >= 4 && GRID_SIZE_Y >= 4) {
        int wall_x = GRID_SIZE_X / 2;
        int gap_y = GRID_SIZE_Y / 2;
        
        for (int y = 0; y < GRID_SIZE_Y; ++y) {
            if (y != gap_y) {
                obstacles[y][wall_x] = true;
            }
        }
    }
}

// Convert state to a readable string
std::string stateToString(const std::vector<int>& state) {
    int x = getCoordValue(state, 0);
    int y = getCoordValue(state, 1);
    
    std::stringstream ss;
    ss << "(" << x << "," << y << ")";
    return ss.str();
}

// Calculate average Q-values across all Q-functions and all states
double calculateAverageQValue(const std::vector<DdNode*>& q_functions, 
                             const std::vector<std::vector<int>>& sample_states) {
    if (q_functions.empty() || sample_states.empty()) return 0.0;
    
    double sum = 0.0;
    int count = 0;
    
    for (const auto& state : sample_states) {
        for (const auto& q_add : q_functions) {
            double q_val = evaluateADD(manager, q_add, state);
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
    if (q_functions.empty()) return 0.0;
    
    long long total_size = 0;
    for (const auto& q_add : q_functions) {
        if (q_add) total_size += Cudd_DagSize(q_add);
    }
    
    return static_cast<double>(total_size) / q_functions.size();
}

// Calculate Bellman error for a sample of states
double calculateBellmanError(const std::vector<DdNode*>& q_functions, 
                            const std::vector<std::vector<int>>& sample_states) {
    if (q_functions.empty() || sample_states.empty()) return 0.0;
    
    double total_error = 0.0;
    int count = 0;
    
    for (const auto& state : sample_states) {
        if (isGoalState(state)) continue;
        
        // Find max Q value for this state
        double max_q = -std::numeric_limits<double>::infinity();
        int best_action = -1;
        
        for (int a = 0; a < NUM_ACTIONS; ++a) {
            double q_val = evaluateADD(manager, q_functions[a], state);
            if (q_val > EVAL_ERROR_THRESHOLD && q_val > max_q) {
                max_q = q_val;
                best_action = a;
            }
        }
        
        if (best_action < 0) continue;
        
        // Apply best action and get next state
        std::pair<std::vector<int>, bool> result = applyAgentAction(state, static_cast<Action>(best_action));
        std::vector<int> next_state = result.first;
        
        // Calculate reward
        double reward = MOVE_COST;
        if (!result.second) {
            reward = OBSTACLE_COST;  // Hit obstacle
        }
        
        if (isGoalState(next_state)) {
            reward += GOAL_REWARD;
        }
        
        // Calculate max Q' for next state
        double max_next_q = 0.0;
        if (!isGoalState(next_state)) {
            max_next_q = -std::numeric_limits<double>::infinity();
            for (int a = 0; a < NUM_ACTIONS; ++a) {
                double q_val = evaluateADD(manager, q_functions[a], next_state);
                if (q_val > EVAL_ERROR_THRESHOLD) {
                    max_next_q = std::max(max_next_q, q_val);
                }
            }
            if (max_next_q == -std::numeric_limits<double>::infinity()) {
                max_next_q = 0.0;
            }
        }
        
        // Calculate target value
        double target = reward + GAMMA * max_next_q;
        
        // Calculate Bellman error
        double error = std::abs(max_q - target);
        
        total_error += error;
        count++;
    }
    
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
// Symbolic Q-Learning with ADD compression benefits
std::map<std::string, Action> symbolicQLearning(bool verbose = true,
                                               int sample_num_states = 1000,
                                               int cache_cleanup_interval = 1000,
                                               bool collect_metrics = true) {
    auto start = std::chrono::high_resolution_clock::now();
    std::map<std::string, Action> policy;
    
    if (!manager || vars.empty()) {
        std::cerr << "Err: Manager/vars not ready\n";
        return {};
    }
    
    // Setup random generators
    std::random_device rd_env;
    env_gen.seed(rd_env());
    
    // Create goal BDD
    if (goal_bdd) Cudd_RecursiveDeref(manager, goal_bdd);
    goal_bdd = createGoalBDD();
    if (!goal_bdd) {
        std::cerr << "Err: Goal BDD creation failed\n";
        return {};
    }
    
    if (verbose) {
        std::cout << "Goal BDD created." << std::endl;
        printNodeInfo("goal_bdd", goal_bdd);
    }
    
    // Initialize Q-functions
    std::vector<DdNode*> q_functions(NUM_ACTIONS);
    DdNode* initial_q_add = Cudd_addConst(manager, 0.0);
    if (!initial_q_add) {
        std::cerr << "Err: Failed to create initial Q-value ADD\n";
        return {};
    }
    
    Cudd_Ref(initial_q_add);
    for (int a = 0; a < NUM_ACTIONS; ++a) {
        q_functions[a] = initial_q_add;
        Cudd_Ref(q_functions[a]);
    }
    Cudd_RecursiveDeref(manager, initial_q_add);
    
    if (verbose) std::cout << "Initialized " << NUM_ACTIONS << " Q-functions" << std::endl;
    
    StateCache state_cache(manager);
    
    std::random_device rd_agent;
    std::mt19937 agent_gen(rd_agent());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::uniform_int_distribution<> action_dis(0, NUM_ACTIONS - 1);
    
    // Generate sample states for metrics
    std::vector<std::vector<int>> sampled_states;
    if (sample_num_states > 0) {
        for (int i = 0; i < sample_num_states; ++i) {
            sampled_states.push_back(generateRandomState(agent_gen));
        }
        if (verbose) std::cout << "Generated " << sampled_states.size() << " sample states" << std::endl;
    }
    
    // Reserve space for metrics
    if (collect_metrics) {
        metrics.avg_q_values.reserve(NUM_EPISODES / 100 + 1);
        metrics.avg_decision_dag_sizes.reserve(NUM_EPISODES / 100 + 1);
        metrics.episode_times.reserve(NUM_EPISODES + 1);
        metrics.goal_visits.reserve(NUM_EPISODES + 1);
        metrics.bellman_errors.reserve(NUM_EPISODES / 100 + 1);
        metrics.memory_usage.reserve(NUM_EPISODES / 100 + 1);
        metrics.avg_path_lengths.reserve(NUM_EPISODES / 100 + 1);
    }
    
    if (verbose) {
        std::cout << "\nPerforming Symbolic Q-Learning for " << NUM_EPISODES << " episodes..." << std::endl;
    }
    
    int eval_fail = 0, statebdd_fail = 0, bddadd_fail = 0, addconst_fail = 0, ite_fail = 0;
    const int PRINT_INTERVAL = std::max(1, NUM_EPISODES / 10);
    const int METRICS_INTERVAL = 100;
    const int MAX_STEPS_TO_PRINT = 3;
    
    auto global_start_time = std::chrono::high_resolution_clock::now();
    
    for (int episode = 0; episode < NUM_EPISODES; ++episode) {
        auto episode_start = std::chrono::high_resolution_clock::now();
        int goal_visits_in_episode = 0;
        int steps_to_goal = 0;
        bool reached_goal = false;
        
        bool print_this_episode = verbose && (episode < 1 || (episode + 1) % PRINT_INTERVAL == 0);
        if (print_this_episode && episode > 0) std::cout << std::endl;
        if (print_this_episode) std::cout << "--- Episode " << (episode + 1) << " ---" << std::endl;
        
        // Generate random initial state, avoiding goal states
        std::vector<int> state = generateRandomState(agent_gen);
        
        int steps_in_episode = 0;
        const int MAX_EPISODE_STEPS = 100;
        
        while (steps_in_episode < MAX_EPISODE_STEPS) {
            steps_in_episode++;
            bool print_this_step = print_this_episode && steps_in_episode <= MAX_STEPS_TO_PRINT;
            
            // Check if we're in a goal state
            if (isGoalState(state)) {
                goal_visits_in_episode++;
                if (!reached_goal) {
                    reached_goal = true;
                    steps_to_goal = steps_in_episode;
                }
                
                if (print_this_step) {
                    std::cout << "  Step " << steps_in_episode << ": State=" << stateToString(state)
                              << " (Goal state)" << std::endl;
                }
                
                // Generate a new state to continue learning
                state = generateRandomState(agent_gen);
                continue;
            }
            
            // Choose action (epsilon-greedy)
            Action action;
            if (dis(agent_gen) < EPSILON) {
                action = static_cast<Action>(action_dis(agent_gen));
            } else {
                double maxQ = -std::numeric_limits<double>::infinity();
                action = UP; // Default action
                bool ok = true;
                
                for (int a = 0; a < NUM_ACTIONS; ++a) {
                    double q = evaluateADD(manager, q_functions[a], state);
                    if (q < EVAL_ERROR_THRESHOLD) {
                        eval_fail++;
                        ok = false;
                        break;
                    }
                    
                    if (q > maxQ) {
                        maxQ = q;
                        action = static_cast<Action>(a);
                    }
                }
                
                if (!ok) {
                    action = static_cast<Action>(action_dis(agent_gen));
                }
            }
            
            if (print_this_step) {
                std::cout << "  Step " << steps_in_episode << ": State=" << stateToString(state)
                          << " -> Action=" << getActionName(action);
            }
            
            // Apply action and get next state
            std::pair<std::vector<int>, bool> agent_result = applyAgentAction(state, action);
            std::vector<int> nextState = agent_result.first;
            bool move_succeeded = agent_result.second;
            
            if (print_this_step) {
                std::cout << " -> NextState=" << stateToString(nextState)
                          << (move_succeeded ? "" : " (Hit obstacle)") << std::endl;
            }
            
            // Calculate reward
            double reward = MOVE_COST;
            if (!move_succeeded) {
                reward = OBSTACLE_COST;
            }
            
            if (isGoalState(nextState)) {
                reward += GOAL_REWARD;
            }
            
            // Find max Q-value for next state
            double max_next_q = 0.0;
            if (!isGoalState(nextState)) {
                max_next_q = -std::numeric_limits<double>::infinity();
                bool nextOk = true;
                
                for (int a = 0; a < NUM_ACTIONS; ++a) {
                    double q = evaluateADD(manager, q_functions[a], nextState);
                    if (q < EVAL_ERROR_THRESHOLD) {
                        eval_fail++;
                        nextOk = false;
                        break;
                    }
                    max_next_q = std::max(max_next_q, q);
                }
                
                if (!nextOk || max_next_q == -std::numeric_limits<double>::infinity()) {
                    max_next_q = 0.0;
                }
            }
            
            double target_q_value = reward + GAMMA * max_next_q;
            
            // Create BDD for current state
            DdNode* stateBdd = state_cache.createStateBDD(state);
            if (!stateBdd) {
                statebdd_fail++;
                state = nextState;
                continue;
            }
            
            // Convert to ADD
            DdNode* stateAdd = Cudd_BddToAdd(manager, stateBdd);
            if (!stateAdd) {
                bddadd_fail++;
                Cudd_RecursiveDeref(manager, stateBdd);
                state = nextState;
                continue;
            }
            Cudd_Ref(stateAdd);
            Cudd_RecursiveDeref(manager, stateBdd);
            
            // Get current Q-value
            DdNode* current_q_add = q_functions[action];
            double oldQ = evaluateADD(manager, current_q_add, state);
            if (oldQ < EVAL_ERROR_THRESHOLD) {
                eval_fail++;
                Cudd_RecursiveDeref(manager, stateAdd);
                state = nextState;
                continue;
            }
            
            // Calculate new Q-value
            double newQ = (1.0 - ALPHA) * oldQ + ALPHA * target_q_value;
            if (print_this_step) {
                std::cout << "      R=" << reward << ", maxQ'=" << max_next_q
                          << ", oldQ=" << oldQ << ", Target=" << target_q_value
                          << ", newQ=" << newQ << std::endl;
            }
            
            // Create constant for new value
            DdNode* newQ_s_add = Cudd_addConst(manager, newQ);
            if (!newQ_s_add) {
                addconst_fail++;
                Cudd_RecursiveDeref(manager, stateAdd);
                state = nextState;
                continue;
            }
            Cudd_Ref(newQ_s_add);
            
            // Update Q-function using ITE (If-Then-Else)
            DdNode* updatedQAdd = Cudd_addIte(manager, stateAdd, newQ_s_add, current_q_add);
            if (!updatedQAdd) {
                ite_fail++;
                Cudd_RecursiveDeref(manager, stateAdd);
                Cudd_RecursiveDeref(manager, newQ_s_add);
                state = nextState;
                continue;
            }
            Cudd_Ref(updatedQAdd);
            
            // Clean up
            Cudd_RecursiveDeref(manager, stateAdd);
            Cudd_RecursiveDeref(manager, newQ_s_add);
            Cudd_RecursiveDeref(manager, q_functions[action]);
            q_functions[action] = updatedQAdd;
            
            // Move to next state
            state = nextState;
        }
        
        if (print_this_episode && steps_in_episode >= MAX_EPISODE_STEPS) {
            std::cout << "  Episode finished due to MAX_STEPS." << std::endl;
        }
        
        // Track metrics for this episode
        if (collect_metrics) {
            metrics.goal_visits.push_back(goal_visits_in_episode);
            
            // Collect timing info
            auto episode_end = std::chrono::high_resolution_clock::now();
            double episode_duration = std::chrono::duration<double>(episode_end - episode_start).count();
            metrics.episode_times.push_back(episode_duration);
            
            // Collect other metrics periodically
            if ((episode + 1) % METRICS_INTERVAL == 0) {
                // Average Q-value
                double avg_q = calculateAverageQValue(q_functions, sampled_states);
                metrics.avg_q_values.push_back(avg_q);
                
                // Average DAG size
                double avg_dag_size = calculateAverageDAGSize(q_functions);
                metrics.avg_decision_dag_sizes.push_back(static_cast<int>(avg_dag_size));
                
                // Bellman error
                double bellman_error = calculateBellmanError(q_functions, sampled_states);
                metrics.bellman_errors.push_back(bellman_error);
                
                // Memory usage
                double mem_usage = calculateActualADDQMemory(q_functions);
                metrics.memory_usage.push_back(mem_usage);

                // Total memeory
                double total_memory_mb = (double)Cudd_ReadMemoryInUse(manager) / (1024.0 * 1024.0);

                
                // Path length
                if (reached_goal) {
                    metrics.avg_path_lengths.push_back(steps_to_goal);
                } else if (!metrics.avg_path_lengths.empty()) {
                    metrics.avg_path_lengths.push_back(metrics.avg_path_lengths.back());
                } else {
                    metrics.avg_path_lengths.push_back(MAX_EPISODE_STEPS);
                }
                
                if (verbose && (episode + 1) % PRINT_INTERVAL == 0) {
                    std::cout << "  Metrics at episode " << (episode + 1) << ":" << std::endl;
                    std::cout << "    Avg Q-value: " << avg_q << std::endl;
                    std::cout << "    Avg DAG size: " << avg_dag_size << " nodes" << std::endl;
                    std::cout << "    Bellman error: " << bellman_error << std::endl;
                    std::cout << "    Est. memory: " << mem_usage << " MB" << std::endl;
                    std::cout << "    TotalCuddMem:" << total_memory_mb << "MB" << std::endl;// Show total actual"
                    if (reached_goal) {
                        std::cout << "    Steps to goal: " << steps_to_goal << std::endl;
                    }
                }
            }
        }
        
        // Periodically clean up cache to avoid memory explosion
        if ((episode + 1) % cache_cleanup_interval == 0) {
            if (verbose) std::cout << "  Cleaning state cache at episode " << (episode + 1) << std::endl;
            // StateCache destructor will handle cleanup when we create a new one
            state_cache = StateCache(manager);
            
            // Force garbage collection in CUDD
            Cudd_ReduceHeap(manager, CUDD_REORDER_SAME, 0);
        }
    }
    
    if (verbose) {
        std::cout << "\nLearn complete. Fails: eval=" << eval_fail 
                  << " stateBdd=" << statebdd_fail 
                  << " bddAdd=" << bddadd_fail 
                  << " addConst=" << addconst_fail 
                  << " ITE=" << ite_fail << std::endl;
    }
    
    // Extract policy from all possible valid states
    if (verbose) std::cout << "Extracting policy from grid world states..." << std::endl;
    for (int y = 0; y < GRID_SIZE_Y; ++y) {
        for (int x = 0; x < GRID_SIZE_X; ++x) {
            if (isObstacle(x, y)) {
                continue; // Skip obstacles
            }
            
            std::vector<int> s = coordsToState(x, y);
            std::string sStr = stateToString(s);
            
            if (isGoalState(s)) {
                policy[sStr] = INVALID_ACTION; // Terminal state
            } else {
                Action bestA = UP; // Default action
                double maxQ = -std::numeric_limits<double>::infinity();
                bool ok = true;
                
                for (int a = 0; a < NUM_ACTIONS; ++a) {
                    double q = evaluateADD(manager, q_functions[a], s);
                    if (q < EVAL_ERROR_THRESHOLD) {
                        ok = false;
                        break;
                    }
                    
                    if (q > maxQ) {
                        maxQ = q;
                        bestA = static_cast<Action>(a);
                    }
                }
                
                if (ok) {
                    policy[sStr] = bestA;
                } else {
                    policy[sStr] = UP; // Default to UP if evaluation failed
                }
            }
        }
    }
    
    // Clean up Q-functions
    if (verbose) std::cout << "Cleaning up Q-functions..." << std::endl;
    for (auto& q : q_functions) {
        if (q) Cudd_RecursiveDeref(manager, q);
    }
    q_functions.clear();
    
    auto end = std::chrono::high_resolution_clock::now();
    double total_duration = std::chrono::duration<double>(end - global_start_time).count();
    
    if (verbose) {
        std::cout << "Learning completed in " << total_duration << " seconds." << std::endl;
    }
    
    return policy;
}

// Run simulation with learned policy and collect statistics
void runSimulation(const std::map<std::string, Action>& policy, int numTrials, bool verbose = true) {
    if (policy.empty() || !manager || !goal_bdd) {
        std::cerr << "Sim Err: Policy/Mgr/Goal missing\n";
        return;
    }
    
    long long totalSteps = 0;
    int successCount = 0;
    int timeoutCount = 0;
    long long successSteps = 0;
    
    std::random_device rd_sim;
    std::mt19937 sim_gen(rd_sim());
    
    for (int trial = 0; trial < numTrials; ++trial) {
        std::vector<int> state = generateRandomState(sim_gen);
        int steps = 0;
        const int MAX_STEPS = 100;
        bool reachedGoal = false;
        
        if (verbose && trial < 3) {
            std::cout << "\n--- Trial " << (trial + 1) << " Start: " << stateToString(state) << " ---" << std::endl;
        }
        
        while (steps < MAX_STEPS) {
            steps++;
            totalSteps++;
            
            // Check if we're in a goal state
            if (isGoalState(state)) {
                reachedGoal = true;
                successCount++;
                successSteps += steps;
                if (verbose && trial < 3) {
                    std::cout << "  Goal reached in " << steps << " steps!" << std::endl;
                }
                break;
            }
            
            std::string stateStr = stateToString(state);
            auto policy_it = policy.find(stateStr);
            Action action;
            
            if (policy_it == policy.end()) {
                // State not in policy - use default action
                action = UP;
                if (verbose && trial < 3) {
                    std::cerr << "  Sim: State not in policy. Using default action." << std::endl;
                }
            } else {
                action = policy_it->second;
                if (action == INVALID_ACTION) {
                    action = UP; // Shouldn't happen but just in case
                }
            }
            
            if (verbose && trial < 3) {
                std::cout << "  Sim Step " << steps << ": S=" << stateStr 
                          << " A=" << getActionName(action);
            }
            
            std::pair<std::vector<int>, bool> agent_res = applyAgentAction(state, action);
            std::vector<int> new_state = agent_res.first;
            bool move_succeeded = agent_res.second;
            
            state = new_state; // Update state
            
            if (verbose && trial < 3) {
                std::cout << " -> S'=" << stateToString(state) 
                          << (move_succeeded ? "" : " (Hit obstacle)") << std::endl;
            }
        }
        
        if (!reachedGoal) {
            timeoutCount++;
            if (verbose && trial < 3) {
                std::cout << "  Timeout after " << MAX_STEPS << " steps." << std::endl;
            }
        }
    }
    
    double successRate = (numTrials > 0) ? static_cast<double>(successCount) / numTrials * 100.0 : 0.0;
    double avgSteps = (successCount > 0) ? static_cast<double>(successSteps) / successCount : 0.0;
    
    std::cout << "\n--- Simulation Results (" << numTrials << " trials, " << totalSteps << " total steps) ---" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Success Rate:          " << successCount << "/" << numTrials
              << " (" << successRate << "%)" << std::endl;
    std::cout << "  Timeouts:              " << timeoutCount << "/" << numTrials << std::endl;
    
    if (successCount > 0) {
        std::cout << "  Avg Steps to Goal:     " << avgSteps << std::endl;
    }
    
    std::cout << "-------------------------------------------------------------" << std::endl;
}

// Visualization of the grid and the policy
void visualizeGrid(const std::map<std::string, Action>& policy) {
    std::cout << "\n--- Grid World Visualization ---" << std::endl;
    
    // Header
    std::cout << "  ";
    for (int x = 0; x < GRID_SIZE_X; ++x) {
        std::cout << "  " << x << " ";
    }
    std::cout << std::endl;
    
    // Horizontal line
    std::cout << "  +";
    for (int x = 0; x < GRID_SIZE_X; ++x) {
        std::cout << "---+";
    }
    std::cout << std::endl;
    
    // Grid content
    for (int y = 0; y < GRID_SIZE_Y; ++y) {
        std::cout << y << " |";
        
        for (int x = 0; x < GRID_SIZE_X; ++x) {
            char cell = ' ';
            std::string direction = " ";
            
            if (x == GOAL_X && y == GOAL_Y) {
                cell = 'G'; // Goal
            } else if (isObstacle(x, y)) {
                cell = '#'; // Obstacle
            } else {
                cell = '.'; // Empty
                
                // Show policy direction
                std::vector<int> state = coordsToState(x, y);
                std::string stateStr = stateToString(state);
                
                auto it = policy.find(stateStr);
                if (it != policy.end()) {
                    Action action = it->second;
                    switch (action) {
                        case UP: direction = "^"; break;
                        case RIGHT: direction = ">"; break;
                        case DOWN: direction = "v"; break;
                        case LEFT: direction = "<"; break;
                        case INVALID_ACTION: direction = "*"; break;
                        default: direction = "?"; break;
                    }
                }
            }
            
            std::cout << cell << direction << cell << "|";
        }
        
        std::cout << std::endl;
        
        // Horizontal line
        std::cout << "  +";
        for (int x = 0; x < GRID_SIZE_X; ++x) {
            std::cout << "---+";
        }
        std::cout << std::endl;
    }
    
    // Legend
    std::cout << "\nLegend:" << std::endl;
    std::cout << "  G   - Goal" << std::endl;
    std::cout << "  #   - Obstacle" << std::endl;
    std::cout << "  .^. - Empty cell with UP policy" << std::endl;
    std::cout << "  .>. - Empty cell with RIGHT policy" << std::endl;
    std::cout << "  .v. - Empty cell with DOWN policy" << std::endl;
    std::cout << "  .<. - Empty cell with LEFT policy" << std::endl;
    std::cout << "  .*. - Terminal state" << std::endl;
    
    std::cout << "-----------------------------" << std::endl;
}

// Print the learned policy
void printPolicy(const std::map<std::string, Action>& policy) {
    std::cout << "\n--- Learned Policy ---" << std::endl;
    int stateWidth = 10;
    std::cout << std::setw(stateWidth) << std::left << "State" 
              << std::setw(10) << std::left << "Action" << std::endl;
    std::cout << std::string(stateWidth + 10, '-') << std::endl;
    
    std::vector<std::string> stateStrs;
    stateStrs.reserve(policy.size());
    for (const auto& entry : policy) stateStrs.push_back(entry.first);
    std::sort(stateStrs.begin(), stateStrs.end());
    
    int printedCount = 0;
    const int MAX_PRINT_POLICY = 30;
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
                          << std::setw(10) << std::left << getActionName(action) << std::endl;
                printedCount++;
            }
        }
    }
    
    if (nonTerminalCount > printedCount) {
        std::cout << "... (omitting " << (nonTerminalCount - printedCount) << " non-terminal states)" << std::endl;
    }
    
    if (terminalCount > 0) {
        std::cout << std::setw(stateWidth) << std::left << "<Goal States>" 
                  << std::setw(10) << std::left << "Terminal" << std::endl;
    }
    
    std::cout << "Total states in policy: " << policy.size() 
              << " (" << nonTerminalCount << " non-terminal, " 
              << terminalCount << " terminal)" << std::endl;
    std::cout << "---------------------------" << std::endl;
}

// Save metrics to CSV files for easy plotting
void saveMetricsToCSV() {
    // Save average Q-values
    {
        std::ofstream file("add_q_grid_avg_values.csv");
        file << "Episode,AvgQValue" << std::endl;
        for (size_t i = 0; i < metrics.avg_q_values.size(); ++i) {
            file << (i+1) * 100 << "," << metrics.avg_q_values[i] << std::endl;
        }
    }
    
    // Save DAG sizes
    {
        std::ofstream file("add_q_grid_dag_sizes.csv");
        file << "Episode,NodeCount" << std::endl;
        for (size_t i = 0; i < metrics.avg_decision_dag_sizes.size(); ++i) {
            file << (i+1) * 100 << "," << metrics.avg_decision_dag_sizes[i] << std::endl;
        }
    }
    
    // Save Bellman errors
    {
        std::ofstream file("add_q_grid_bellman_errors.csv");
        file << "Episode,Error" << std::endl;
        for (size_t i = 0; i < metrics.bellman_errors.size(); ++i) {
            file << (i+1) * 100 << "," << metrics.bellman_errors[i] << std::endl;
        }
    }
    
    // Save memory usage
    {
        std::ofstream file("add_q_grid_memory_usage.csv");
        file << "Episode,MemoryMB" << std::endl;
        for (size_t i = 0; i < metrics.memory_usage.size(); ++i) {
            file << (i+1) * 100 << "," << metrics.memory_usage[i] << std::endl;
        }
    }
    
    // Save goal visits
    {
        std::ofstream file("add_q_grid_goal_visits.csv");
        file << "Episode,GoalVisits" << std::endl;
        for (size_t i = 0; i < metrics.goal_visits.size(); ++i) {
            file << i+1 << "," << metrics.goal_visits[i] << std::endl;
        }
    }
    
    // Save path lengths
    {
        std::ofstream file("add_q_grid_path_lengths.csv");
        file << "Episode,PathLength" << std::endl;
        for (size_t i = 0; i < metrics.avg_path_lengths.size(); ++i) {
            file << (i+1) * 100 << "," << metrics.avg_path_lengths[i] << std::endl;
        }
    }
    
    // Save episode times
    {
        std::ofstream file("add_q_grid_episode_times.csv");
        file << "Episode,Duration" << std::endl;
        for (size_t i = 0; i < metrics.episode_times.size(); ++i) {
            file << i+1 << "," << metrics.episode_times[i] << std::endl;
        }
    }
    
    std::cout << "Metrics saved to CSV files for visualization." << std::endl;
}

// Print help message
void printHelp() {
    std::cout << "Usage: program [options]\n";
    std::cout << "Options:\n";
    std::cout << "  -gx N         Set grid width to N (default: 8)\n";
    std::cout << "  -gy N         Set grid height to N (default: 8)\n";
    std::cout << "  -obs N        Set number of obstacles to N (default: 10)\n";
    std::cout << "  -e N          Set number of episodes to N (default: 5000)\n";
    std::cout << "  -a F          Set alpha learning rate to F (default: 0.1)\n";
    std::cout << "  -g F          Set gamma discount factor to F (default: 0.99)\n";
    std::cout << "  -eps F        Set epsilon for exploration to F (default: 0.2)\n";
    std::cout << "  -s N          Set policy sampling size to N (default: 1000)\n";
    std::cout << "  -sim N        Run N simulation trials after learning\n";
    std::cout << "  -v            Enable verbose mode\n";
    std::cout << "  -metrics      Generate CSV files with metrics for plotting\n";
    std::cout << "  -h            Print this help message\n";
}

int main(int argc, char* argv[]) {
    bool verbose = false;
    int numSimTrials = 0;
    int policy_sample_size = 1000;
    int cache_cleanup_interval = 1000;
    bool save_metrics = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-v") {
            verbose = true;
        } else if (arg == "-metrics") {
            save_metrics = true;
        } else if (arg == "-gx" && i + 1 < argc) {
            try {
                GRID_SIZE_X = std::stoi(argv[++i]);
                if (GRID_SIZE_X <= 0) {
                    std::cerr << "GRID_SIZE_X must be positive.\n";
                    return 1;
                }
            } catch (...) {
                std::cerr << "Invalid number for -gx.\n";
                return 1;
            }
        } else if (arg == "-gy" && i + 1 < argc) {
            try {
                GRID_SIZE_Y = std::stoi(argv[++i]);
                if (GRID_SIZE_Y <= 0) {
                    std::cerr << "GRID_SIZE_Y must be positive.\n";
                    return 1;
                }
            } catch (...) {
                std::cerr << "Invalid number for -gy.\n";
                return 1;
            }
        } else if (arg == "-obs" && i + 1 < argc) {
            try {
                NUM_OBSTACLES = std::stoi(argv[++i]);
                if (NUM_OBSTACLES < 0) {
                    std::cerr << "NUM_OBSTACLES must be non-negative.\n";
                    return 1;
                }
            } catch (...) {
                std::cerr << "Invalid number for -obs.\n";
                return 1;
            }
        } else if (arg == "-e" && i + 1 < argc) {
            try {
                NUM_EPISODES = std::stoi(argv[++i]);
                if (NUM_EPISODES <= 0) {
                    std::cerr << "NUM_EPISODES must be positive.\n";
                    return 1;
                }
            } catch (...) {
                std::cerr << "Invalid number for -e.\n";
                return 1;
            }
        } else if (arg == "-a" && i + 1 < argc) {
            try {
                ALPHA = std::stod(argv[++i]);
                if (ALPHA <= 0 || ALPHA > 1) {
                    std::cerr << "ALPHA must be in (0,1].\n";
                    return 1;
                }
            } catch (...) {
                std::cerr << "Invalid number for -a.\n";
                return 1;
            }
        } else if (arg == "-g" && i + 1 < argc) {
            try {
                GAMMA = std::stod(argv[++i]);
                if (GAMMA < 0 || GAMMA > 1) {
                    std::cerr << "GAMMA must be in [0,1].\n";
                    return 1;
                }
            } catch (...) {
                std::cerr << "Invalid number for -g.\n";
                return 1;
            }
        } else if (arg == "-eps" && i + 1 < argc) {
            try {
                EPSILON = std::stod(argv[++i]);
                if (EPSILON < 0 || EPSILON > 1) {
                    std::cerr << "EPSILON must be in [0,1].\n";
                    return 1;
                }
            } catch (...) {
                std::cerr << "Invalid number for -eps.\n";
                return 1;
            }
        } else if (arg == "-s" && i + 1 < argc) {
            try {
                policy_sample_size = std::stoi(argv[++i]);
                if (policy_sample_size < 0) {
                    policy_sample_size = 0;
                }
            } catch (...) {
                std::cerr << "Invalid number for -s.\n";
                return 1;
            }
        } else if (arg == "-sim" && i + 1 < argc) {
            try {
                numSimTrials = std::stoi(argv[++i]);
                if (numSimTrials < 0) numSimTrials = 0;
            } catch (...) {
                std::cerr << "Invalid number for -sim.\n";
                return 1;
            }
        } else if (arg == "-h") {
            printHelp();
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            printHelp();
            return 1;
        }
    }
    
    // Set goal coordinates to be at the bottom right of the grid
    GOAL_X = GRID_SIZE_X - 1;
    GOAL_Y = GRID_SIZE_Y - 1;
    
    // Print parameters
    std::cout << "--- Grid World Parameters ---" << std::endl;
    std::cout << "  Grid Size: " << GRID_SIZE_X << "x" << GRID_SIZE_Y << std::endl;
    std::cout << "  Number of Obstacles: " << NUM_OBSTACLES << std::endl;
    std::cout << "  Goal Position: (" << GOAL_X << "," << GOAL_Y << ")" << std::endl;
    std::cout << "  Bits Per Coordinate: " << BITS_PER_COORD << std::endl;
    std::cout << "  Total Location Bits: " << TOTAL_LOCATION_BITS << std::endl;
    std::cout << "  Gamma: " << GAMMA << ", Epsilon: " << EPSILON << ", Alpha: " << ALPHA << std::endl;
    std::cout << "  Episodes: " << NUM_EPISODES << std::endl;
    std::cout << "  Policy Sampling: " << policy_sample_size << " states" << std::endl;
    std::cout << "------------------------------" << std::endl;
    
    // Initialize obstacles
    initializeObstacles();
    
    // Initialize CUDD manager
    manager = Cudd_Init(TOTAL_LOCATION_BITS, 0, CUDD_UNIQUE_SLOTS, CUDD_CACHE_SLOTS, 0);
    if (!manager) {
        std::cerr << "Failed to initialize CUDD manager.\n";
        return 1;
    }
    
    // Set variable reordering for optimal performance
    Cudd_AutodynEnable(manager, CUDD_REORDER_SIFT);
    
    // Create variables
    vars.resize(TOTAL_LOCATION_BITS);
    bool vars_ok = true;
    for (int i = 0; i < TOTAL_LOCATION_BITS; ++i) {
        vars[i] = Cudd_bddIthVar(manager, i);
        if (!vars[i]) {
            vars_ok = false;
            break;
        }
    }
    
    if (!vars_ok) {
        std::cerr << "Failed to create BDD variables.\n";
        Cudd_Quit(manager);
        return 1;
    }
    
    // Perform Q-learning with ADD representation
    std::map<std::string, Action> policy = symbolicQLearning(verbose, policy_sample_size, cache_cleanup_interval, save_metrics);
    
    // Visualize the grid and policy
    if (!policy.empty()) {
        visualizeGrid(policy);
        printPolicy(policy);
    } else {
        std::cerr << "Learning failed (empty policy).\n";
        Cudd_Quit(manager);
        return 1;
    }
    
    // Run simulation if requested
    if (numSimTrials > 0) {
        if (!goal_bdd) {
            std::cerr << "Warning: Goal BDD missing for simulation.\n";
            goal_bdd = createGoalBDD();
        }
        
        if (goal_bdd) {
            runSimulation(policy, numSimTrials, verbose);
        } else {
            std::cerr << "Error: Could not create goal BDD for simulation.\n";
        }
    } else {
        std::cout << "\nRun with '-sim N' to simulate the learned policy.\n";
    }
    
    // Save metrics to CSV if requested
    if (save_metrics) {
        saveMetricsToCSV();
    }
    
    // Clean up
    if (manager) {
        if (goal_bdd) {
            Cudd_RecursiveDeref(manager, goal_bdd);
            goal_bdd = nullptr;
        }
        
        vars.clear();
        Cudd_Quit(manager);
        manager = nullptr;
    }
    
    return 0;
}
