// Efficient ADD-Q algorithm implementation for a dice game (e.g., Pig variations)
// Demonstrates symbolic representation for dice states and expected value updates.
// Includes metrics collection and CSV output similar to the grid world example.
// Author - Aadithya Srinivasn Anand

#include <iostream>
#include <vector>
#include <map>
#include <unordered_map> // Keep for potential future cache, though not used now
#include <cmath>
#include <cassert>
#include <iomanip>
#include <string>
#include <sstream>
#include <fstream> // Added for file I/O
#include <limits>
#include <algorithm>
#include <chrono>
#include <random>
#include <numeric> // std::iota
#include <functional> // Keep for potential future use
#include <memory> // Keep for potential future use, just in case 

// Include CUDD header
#include "cudd.h"
#include "cuddInt.h" // Keep for potential detailed debugging - generally it is better to path it to cudd when you download it

// --- Dice Game Parameters ---
int NUM_DICE = 3;
int NUM_FACES = 6;      // Default to standard dice
double GAMMA = 0.9;
double EPSILON = 0.2;
double ALPHA = 0.1;
int NUM_EPISODES = 20000;
double TERMINAL_REWARD = 1.0; // Reward for reaching a terminal state (all dice same)
double STEP_COST = 0.0;       // Cost per non-terminal step (can be non-zero)

// --- Encoding Setup ---
int VARS_PER_DIE = 0; // Calculated in main
int TOTAL_BDD_VARS = 0; // Calculated in main

// Action Definition: Action 'k' means "keep all dice showing face value k"
// Terminal states have no valid actions.
using Action = int;
const Action INVALID_ACTION = -1; // Represents terminal state or invalid action choice
int NUM_ACTIONS = 0; // Calculated in main
const double EVAL_ERROR_THRESHOLD = -9990.0; // Error check threshold
const int METRICS_INTERVAL = 100; 

// Metrics for ADD-Q analysis (adapted from grid_world)
struct AddQMetrics {
    std::vector<double> avg_q_values;          // Average Q-value during learning
    std::vector<double> avg_decision_dag_sizes; // Average ADD node count (double for consistency)
    std::vector<double> episode_times;         // Time per episode
    std::vector<int> terminal_state_visits;    // Visits to terminal states per episode
    std::vector<double> bellman_errors;        // Bellman error per episode
    std::vector<double> memory_usage;          // Estimated memory usage (MB) per episode
    // std::vector<double> avg_path_lengths; // Not directly applicable like grid world, add later if required
};

// Globals
DdManager* manager = nullptr;
std::vector<DdNode*> vars; // BDD variables
DdNode* terminal_bdd = nullptr; // BDD representing all terminal states (refer CuDD manual for more details)
std::mt19937 env_gen; // Random generator for environment aspects (if any needed beyond simulation)
AddQMetrics metrics;

// Forward declarations
double calculateAverageQValue(const std::vector<DdNode*>& q_functions, const std::vector<std::vector<int>>& sample_states);
double calculateAverageDAGSize(const std::vector<DdNode*>& q_functions);
double calculateBellmanError(const std::vector<DdNode*>& q_functions, const std::vector<std::vector<int>>& sample_states, DdNode* terminalBDD);
void saveMetricsToCSV();
void printPolicy(const std::map<std::string, Action>& policy); // Keep printPolicy declaration
int calculateVarsPerDie(int numFaces); // Declaration for calculation function


/********************************************************************************************************************************************************/
// ---------------------------------- Helper Functions (Printing, State Conversion, CUDD Interaction, etc.) --------------------------------------------
/********************************************************************************************************************************************************/



void printNodeInfo(const char* name, DdNode* node, bool isAdd = false) {
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

void printDiceState(const std::vector<int>& state) {
    std::cout << "[";
    for (size_t i = 0; i < state.size(); i++) { std::cout << state[i]; if (i < state.size() - 1) std::cout << ","; }
    std::cout << "]";
}

std::string stateToString(const std::vector<int>& state) {
    std::string result = "[";
    for (size_t i = 0; i < state.size(); i++) { result += std::to_string(state[i]); if (i < state.size() - 1) result += ","; }
    result += "]"; return result;
}

std::vector<int> stringToState(const std::string& stateStr) {
     std::vector<int> state; if (stateStr.length() <= 2) return state;
    std::string clean = stateStr.substr(1, stateStr.size() - 2); std::stringstream ss(clean); std::string token;
    while (std::getline(ss, token, ',')) { try { state.push_back(std::stoi(token)); } catch (...) {} }
    return state;
}

// Generate all possible dice states
std::vector<std::vector<int>> generateAllStates(int numDice, int numFaces) {
     std::vector<std::vector<int>> states; if (numDice <= 0 || numFaces <= 0) return states;
    long long totalStates = 1;
    // Check for potential overflow before multiplication
    for(int i=0; i<numDice; ++i) {
        if (numFaces > 0 && totalStates > std::numeric_limits<long long>::max() / numFaces) {
             std::cerr << "Warning: Number of states exceeds representable limit (" << numFaces << "^" << numDice << "). Returning empty state list." << std::endl;
             return {}; // Return empty vector to indicate failure
        }
        totalStates *= numFaces;
    }

    if (totalStates > 10000000) { // Increased limit slightly
         std::cerr << "Warning: Number of states (" << totalStates << ") is very large. State generation might be slow or consume significant memory." << std::endl;
    } else if (totalStates == 0 && numFaces > 0 && numDice > 0) {
         std::cerr << "Warning: Calculated total states is 0, which is unexpected for positive dice/faces." << std::endl;
         return {};
    }

    try {
        states.reserve(static_cast<size_t>(totalStates));
    } catch (const std::bad_alloc& e) {
        std::cerr << "Error: Failed to reserve memory for " << totalStates << " states. " << e.what() << std::endl;
        return {};
    }

    for (long long i = 0; i < totalStates; i++) {
        std::vector<int> state(numDice); long long temp = i;
        for (int j = numDice - 1; j >= 0; j--) { state[j] = temp % numFaces; temp /= numFaces; } states.push_back(std::move(state)); // Use move for efficiency
    } return states;
}

// Check if all dice show the same face value
bool allSameValue(const std::vector<int>& state) {
     if (state.empty() || state.size() == 1) return true; int val = state[0];
    for (size_t i = 1; i < state.size(); i++) { if (state[i] != val) return false; } return true;
}

// Get a readable name for an action
std::string getActionName(Action action) {
    if (action == INVALID_ACTION) return "Terminal"; if (action >= 0 && action < NUM_FACES) return "Keep " + std::to_string(action) + "s";
    return "Unknown Action";
}

// Get the list of valid actions for a non-terminal state
std::vector<Action> getAvailableActions(const std::vector<int>& state) {
    // Terminal states have no actions
    if (allSameValue(state)) return {};
    // In this game, any face value present can be chosen to keep.
    // A simpler version (used here) is you can always choose to keep any face value 0..NUM_FACES-1
    std::vector<Action> actions(NUM_ACTIONS);
    std::iota(actions.begin(), actions.end(), 0); // Actions are 0 to NUM_FACES-1
    return actions;
}

// Structure to represent a possible transition
struct Transition { std::vector<int> nextState; double probability; };

// Generate possible next states and their probabilities for a given state and action
std::vector<Transition> generateTransitions(const std::vector<int>& state, Action action) {
     std::vector<Transition> transitions;
     // If already in a terminal state, the only transition is to itself with probability 1
     if (allSameValue(state)) { transitions.push_back({state, 1.0}); return transitions; }

     // Action dictates which face value to keep
     int valueToKeep = action;
     if (valueToKeep < 0 || valueToKeep >= NUM_FACES) {
         std::cerr << "Error: Invalid action " << action << " passed to generateTransitions." << std::endl;
         return {}; // Return empty vector for invalid action
     }

    std::vector<bool> keepDie(state.size(), false); int rerolledDice = 0;
    for (size_t i = 0; i < state.size(); i++) {
        if (state[i] == valueToKeep) keepDie[i] = true;
        else rerolledDice++;
    }

    // If no dice are rerolled (e.g., all dice already match the value to keep),
    // the state transitions to itself with probability 1.
    // This state might become terminal if all dice match.
    if (rerolledDice == 0) {
        transitions.push_back({state, 1.0});
        return transitions;
    }

    // Compute the number of possible outcomes for the rerolled dice
    long long possibleOutcomes = 1;
    if (NUM_FACES <= 0) return {}; // Avoid issues with zero faces
    for(int i=0; i<rerolledDice; ++i) {
        if (possibleOutcomes > std::numeric_limits<long long>::max() / NUM_FACES) {
            std::cerr << "Warning: Number of transition outcomes exceeds limit in generateTransitions." << std::endl;
            return {}; // Too many outcomes to handle
        }
        possibleOutcomes *= NUM_FACES;
    }

     // Avoid division by zero if NUM_FACES is somehow invalid or possibleOutcomes is zero
     if (possibleOutcomes <= 0) return {};

    double probability = 1.0 / static_cast<double>(possibleOutcomes);
    transitions.reserve(static_cast<size_t>(possibleOutcomes));

    // Iterate through all possible outcomes of the reroll
    for (long long outcome_idx = 0; outcome_idx < possibleOutcomes; outcome_idx++) {
        std::vector<int> nextState = state; long long tempOutcome = outcome_idx;
        int current_reroll_die = 0; // Track which die is being assigned a value
        for (size_t i = 0; i < state.size(); i++) {
            if (!keepDie[i]) {
                 // Assign the next face value from the outcome index
                 nextState[i] = tempOutcome % NUM_FACES;
                 tempOutcome /= NUM_FACES;
                 current_reroll_die++;
            }
        }
        // Ensure we used the correct number of rerolled dice values
         assert(current_reroll_die == rerolledDice);

        transitions.push_back({nextState, probability});
    } return transitions;
}


// --- CUDD Interaction ---

// Calculate the number of BDD variables needed per die based on log2 encoding
int calculateVarsPerDie(int numFaces) {
    if (numFaces <= 1) return 0; // No variables needed if only 0 or 1 face
    // Calculate ceil(log2(numFaces))
    return static_cast<int>(std::ceil(std::log2(static_cast<double>(numFaces))));
}

// Get the global BDD variable index for a specific bit of a specific die
int getVarIndex(int die_index, int bit_index) {
    assert(die_index >= 0 && die_index < NUM_DICE);
    assert(bit_index >= 0 && bit_index < VARS_PER_DIE);
    return die_index * VARS_PER_DIE + bit_index;
}

// Create a BDD representing a specific die showing a specific face value
DdNode* createFaceValueBDD(DdManager* mgr, int die_idx, int face_value, const std::vector<DdNode*>& bddVars) {
     if (face_value < 0 || face_value >= NUM_FACES) {
         // std::cerr << "Warning: Invalid face value " << face_value << " requested in createFaceValueBDD." << std::endl;
         return Cudd_ReadLogicZero(mgr); // Return logical zero for invalid face values
     }
     if (VARS_PER_DIE == 0) {
         // If NUM_FACES is 1, any valid face_value (which must be 0) is represented by TRUE
         return (face_value == 0) ? Cudd_ReadOne(mgr) : Cudd_ReadLogicZero(mgr);
     }

    DdNode* faceBdd = Cudd_ReadOne(mgr); Cudd_Ref(faceBdd);
    int temp_value = face_value;

    for (int bit = 0; bit < VARS_PER_DIE; ++bit) {
        int var_idx = getVarIndex(die_idx, bit);
        bool expected_bit_value = (temp_value % 2 != 0); // Is the current lowest bit 1?
        temp_value /= 2; // Move to the next bit

        // Basic sanity check
        if (var_idx >= static_cast<int>(bddVars.size()) || !bddVars[var_idx]) {
             std::cerr << "Error: Invalid var_idx " << var_idx << " or null BDD var in createFaceValueBDD." << std::endl;
             Cudd_RecursiveDeref(mgr, faceBdd);
             return Cudd_ReadLogicZero(mgr); // Indicate failure
        }

        DdNode* bddVar = bddVars[var_idx];
        DdNode* literal = Cudd_NotCond(bddVar, !expected_bit_value); // var if bit is 1, !var if bit is 0
        DdNode* tmp = Cudd_bddAnd(mgr, faceBdd, literal);

        if (!tmp) {
            std::cerr << "Error: Cudd_bddAnd failed in createFaceValueBDD." << std::endl;
            Cudd_RecursiveDeref(mgr, faceBdd);
            // Note: 'literal' is derived from bddVar and doesn't need separate deref here
            return Cudd_ReadLogicZero(mgr); // Indicate failure
        }
        Cudd_Ref(tmp);
        Cudd_RecursiveDeref(mgr, faceBdd);
        faceBdd = tmp; // Move to the next conjunction
    }
    // The final faceBdd holds the conjunction of literals representing the face value
    return faceBdd; // Return the referenced BDD
}

// Create a BDD representing a specific complete dice state (conjunction of face values)
DdNode* createStateBDD(DdManager* mgr, const std::vector<int>& state, const std::vector<DdNode*>& bddVars) {
     if (static_cast<int>(state.size()) != NUM_DICE) {
         std::cerr << "Error: State size mismatch in createStateBDD. Expected " << NUM_DICE << ", got " << state.size() << "." << std::endl;
         return Cudd_ReadLogicZero(mgr); // Return logical zero for invalid input state
     }

    DdNode* stateBdd = Cudd_ReadOne(mgr); Cudd_Ref(stateBdd); // Start with logical TRUE

    for (int i = 0; i < NUM_DICE; ++i) {
        // Create BDD for the i-th die having the value state[i]
        DdNode* dieValueBdd = createFaceValueBDD(mgr, i, state[i], bddVars); // Gets a referenced BDD or logicZero

        // Check if dieValueBdd creation failed (returned logicZero)
        if (dieValueBdd == Cudd_ReadLogicZero(mgr)) {
            Cudd_RecursiveDeref(mgr, stateBdd); // Clean up the accumulating state BDD
            // dieValueBdd is logicZero, managed by CUDD, no deref needed
            return Cudd_ReadLogicZero(mgr); // Propagate failure
        }

        // AND the current state BDD with the BDD for the specific die's value
        DdNode* tmp = Cudd_bddAnd(mgr, stateBdd, dieValueBdd);

        if (!tmp) {
             std::cerr << "Error: Cudd_bddAnd failed in createStateBDD." << std::endl;
             Cudd_RecursiveDeref(mgr, stateBdd);
             Cudd_RecursiveDeref(mgr, dieValueBdd); // Deref the successfully created dieValueBdd
             return Cudd_ReadLogicZero(mgr); // Propagate failure
        }
        Cudd_Ref(tmp);

        // Clean up the previous stateBdd and the dieValueBdd (they are now part of tmp)
        Cudd_RecursiveDeref(mgr, stateBdd);
        Cudd_RecursiveDeref(mgr, dieValueBdd);

        stateBdd = tmp; // Update stateBdd to the new conjunction
    }
    // Final stateBdd represents the complete state
    return stateBdd; // Return the referenced BDD for the state
}

// Evaluate the value of an ADD for a specific concrete dice state
double evaluateADD(DdManager* mgr, DdNode* add, const std::vector<int>& state, const std::vector<DdNode*>& /*bddVars - according to me it is not actually necessary for Eval*/) {
     // --- Input Validation ---
     if (!mgr) { std::cerr << "Error: Null manager passed to evaluateADD." << std::endl; return EVAL_ERROR_THRESHOLD; }
     if (!add) { /*std::cerr << "Warning: Null ADD passed to evaluateADD." << std::endl;*/ return EVAL_ERROR_THRESHOLD; } // Null ADD might be valid (e.g., before learning)
     if (static_cast<int>(state.size()) != NUM_DICE) { std::cerr << "Error: State size mismatch in evaluateADD. Expected " << NUM_DICE << ", got " << state.size() << "." << std::endl; return EVAL_ERROR_THRESHOLD; }
     if (VARS_PER_DIE * NUM_DICE != TOTAL_BDD_VARS) { std::cerr << "Error: Variable count mismatch in evaluateADD." << std::endl; return EVAL_ERROR_THRESHOLD; }

     // Check if the ADD is constant - optimization
     if (Cudd_IsConstant(add)) { return Cudd_V(add); }

     // --- Create Assignment Array ---
     int managerSize = Cudd_ReadSize(mgr); // Get total variables in the manager
     // Ensure manager has at least the variables we need
     if (managerSize < TOTAL_BDD_VARS) { std::cerr << "Error: CUDD manager size (" << managerSize << ") is less than required BDD variables (" << TOTAL_BDD_VARS << ")." << std::endl; return EVAL_ERROR_THRESHOLD; }

     std::vector<int> assignment(managerSize, 0); // Initialize assignment array (0 is typical default)

     // --- Populate Assignment based on State ---
     for (int i = 0; i < NUM_DICE; ++i) { // Iterate through each die
         int face_value = state[i];
         // Validate face value against game rules
         if (face_value < 0 || face_value >= NUM_FACES) { std::cerr << "Error: Invalid face value " << face_value << " for die " << i << " in evaluateADD." << std::endl; return EVAL_ERROR_THRESHOLD; }

         int temp_value = face_value;
         for (int bit = 0; bit < VARS_PER_DIE; ++bit) { // Iterate through bits for this die
             int var_idx = getVarIndex(i, bit);
             // Check index bounds against the assignment array size
             if (var_idx < 0 || var_idx >= managerSize) { std::cerr << "Error: Calculated variable index " << var_idx << " is out of bounds for manager size " << managerSize << "." << std::endl; return EVAL_ERROR_THRESHOLD; }

             assignment[var_idx] = (temp_value % 2 != 0) ? 1 : 0; // Assign 1 if bit is set, 0 otherwise
             temp_value /= 2;
         }
     }

     // --- Evaluate ---
     DdNode* evalNode = Cudd_Eval(mgr, add, assignment.data()); // Pass pointer to vector data

     // --- Result Handling ---
     if (evalNode == NULL) { std::cerr << "Error: Cudd_Eval returned NULL in evaluateADD." << std::endl; return EVAL_ERROR_THRESHOLD; }
     if (!Cudd_IsConstant(evalNode)) { std::cerr << "Error: Cudd_Eval did not return a constant node in evaluateADD." << std::endl; // This shouldn't happen with a valid assignment
         return EVAL_ERROR_THRESHOLD; }

     return Cudd_V(evalNode); // Return the constant value of the evaluated node
}

// Check if a state is terminal using the precomputed terminal BDD
bool isTerminalState(DdManager* mgr, DdNode* termBdd, const std::vector<int>& state, const std::vector<DdNode*>& bddVars) {
    if (!mgr || !termBdd) return false; // Cannot check without manager or terminal BDD
    if (Cudd_IsConstant(termBdd)) { // Check if the terminal condition itself is constant
        return (termBdd == Cudd_ReadOne(mgr)); // Terminal if it's constant TRUE
    }
    if (static_cast<int>(state.size()) != NUM_DICE) return false; // Invalid state size

    // Create a temporary BDD for the specific state
    DdNode* stateBdd = createStateBDD(mgr, state, bddVars); // Referenced
    if (stateBdd == Cudd_ReadLogicZero(mgr)) {
        return false; // Failed to create state BDD, cannot evaluate
    }

    // Check if the state BDD is subsumed by the terminal BDD
    // state => terminal is equivalent to (state AND terminal) == state
    DdNode* intersection = Cudd_bddAnd(mgr, stateBdd, termBdd);
    if (!intersection) {
        Cudd_RecursiveDeref(mgr, stateBdd);
        return false; // Error during AND operation
    }
    Cudd_Ref(intersection);

    bool isTerminal = (intersection == stateBdd);

    Cudd_RecursiveDeref(mgr, stateBdd);
    Cudd_RecursiveDeref(mgr, intersection);

    return isTerminal;

    /* Alternative using Cudd_Eval (something is bugging out - need to kook into it):
    int managerSize = Cudd_ReadSize(mgr);
    if (managerSize < TOTAL_BDD_VARS) return false;
    std::vector<int> assignment(managerSize, 0);
    // Populate assignment (same logic as in evaluateADD)
     for (int i = 0; i < NUM_DICE; ++i) {
         int face_value = state[i];
         if (face_value < 0 || face_value >= NUM_FACES) return false;
         int temp_value = face_value;
         for (int bit = 0; bit < VARS_PER_DIE; ++bit) {
             int var_idx = getVarIndex(i, bit);
             if (var_idx < 0 || var_idx >= managerSize) return false;
             assignment[var_idx] = (temp_value % 2 != 0) ? 1 : 0;
             temp_value /= 2;
         }
     }
    DdNode* evalNode = Cudd_Eval(mgr, termBdd, assignment.data());
    return (evalNode != nullptr && evalNode == Cudd_ReadOne(mgr));
    */
}
/************************************************************************************************************************************************************ */
// --------------------------------Metrics Calculation Functions ---------------------------------------------------------------------------------------
/************************************************************************************************************************************************************* */


// Calculate average Q-values across all Q-functions and sampled states 
double calculateAverageQValue(const std::vector<DdNode*>& q_functions,
                             const std::vector<std::vector<int>>& sample_states) {
    if (q_functions.empty() || sample_states.empty() || !manager) return 0.0;

    double sum = 0.0;
    long long count = 0; // Use long long for potentially many states/actions

    for (const auto& state : sample_states) {
        // Skip terminal states for average Q calculation, as they don't have future value
        if (allSameValue(state)) continue;

        for (size_t a = 0; a < q_functions.size(); ++a) {
            if (!q_functions[a]) continue; // Skip if Q-function ADD is null

            double q_val = evaluateADD(manager, q_functions[a], state, vars);
            if (q_val > EVAL_ERROR_THRESHOLD) { // Check if evaluation was successful
                sum += q_val;
                count++;
            } else {
                 // Optional: Log evaluation error during metric calculation
                 // std::cerr << "Warning: evaluateADD failed during calculateAverageQValue for state " << stateToString(state) << ", action " << a << std::endl;
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

// Calculate Bellman error for a sample of states (comment out if necessarry)
double calculateBellmanError(const std::vector<DdNode*>& q_functions,
                            const std::vector<std::vector<int>>& sample_states,
                            DdNode* terminalBDD) {
    if (q_functions.empty() || sample_states.empty() || !manager || !terminalBDD) return 0.0;

    double total_error = 0.0;
    long long count = 0; // Use long long

    for (const auto& state : sample_states) {
        // Skip terminal states, no updates happen there
        if (isTerminalState(manager, terminalBDD, state, vars)) continue;

        // For each possible action from this state
        std::vector<Action> actions = getAvailableActions(state);
        if (actions.empty()) continue; // Should not happen for non-terminal, but check

        for (Action action : actions) {
             if (action < 0 || static_cast<size_t>(action) >= q_functions.size() || !q_functions[action]) {
                 continue; // Skip invalid actions or null Q-functions
             }

            // 1. Get current Q-value Q(s, a)
            double current_q = evaluateADD(manager, q_functions[action], state, vars);
            if (current_q < EVAL_ERROR_THRESHOLD) {
                // std::cerr << "Warning: Eval error for Q(s,a) in Bellman calculation. State: " << stateToString(state) << ", Action: " << action << std::endl;
                continue; // Skip if current Q value is invalid
            }

            // 2. Calculate the expected target value: E[R + gamma * max_a' Q(s', a')]
            double expected_target_q_value = 0.0;
            std::vector<Transition> transitions = generateTransitions(state, action);
            if (transitions.empty()) continue; // Skip if no transitions generated

            bool expectation_calc_ok = true;
            for (const auto& transition : transitions) {
                const std::vector<int>& next_state = transition.nextState;
                double probability = transition.probability;
                bool next_is_terminal = isTerminalState(manager, terminalBDD, next_state, vars);

                // Determine reward R(s,a,s')
                double reward = next_is_terminal ? TERMINAL_REWARD : STEP_COST;

                // Determine max_a' Q(s', a')
                double max_next_q = 0.0; // Value is 0 if next state is terminal
                if (!next_is_terminal) {
                    max_next_q = -std::numeric_limits<double>::infinity();
                    std::vector<Action> next_actions = getAvailableActions(next_state);
                    if(next_actions.empty() && !next_is_terminal) {
                        // This case should ideally not happen if logic is correct
                        // std::cerr << "Warning: Non-terminal state has no actions in Bellman calc: " << stateToString(next_state) << std::endl;
                        max_next_q = 0.0; // Assign a default value
                    } else {
                        bool any_next_q_valid = false;
                        for (Action next_action : next_actions) {
                            if (next_action < 0 || static_cast<size_t>(next_action) >= q_functions.size() || !q_functions[next_action]) continue;
                            double q_val = evaluateADD(manager, q_functions[next_action], next_state, vars);
                            if (q_val > EVAL_ERROR_THRESHOLD) {
                                max_next_q = std::max(max_next_q, q_val);
                                any_next_q_valid = true;
                            } else {
                               // Error evaluating Q(s', a')
                            }
                        }
                        // If no valid Q-values found for next state actions, default max_next_q
                         if (!any_next_q_valid) {
                            // This might indicate learning hasn't progressed far, or evaluation issues
                            // std::cerr << "Warning: No valid next Q-values found for state " << stateToString(next_state) << " in Bellman calculation." << std::endl;
                            max_next_q = 0.0; // Default to 0 if all evaluations failed or state has no valid actions explored yet
                         } else if (max_next_q == -std::numeric_limits<double>::infinity()){
                             // If loop ran but maxQ is still -inf, it implies eval errors occurred
                             // Or potentially all valid Q values were extremely negative, keep it as 0.0
                             max_next_q = 0.0;
                         }
                    }
                } // end if !next_is_terminal

                expected_target_q_value += probability * (reward + GAMMA * max_next_q);

            } // end loop over transitions

            // 3. Calculate Bellman error for this (s, a) pair: |Q(s, a) - E[target]|
            double error = std::abs(current_q - expected_target_q_value);
            total_error += error;
            count++;

        } // end loop over actions
    } // end loop over sample_states

    return (count > 0) ? (total_error / count) : 0.0;
}


/************************************************************************************************************************************************************ */
// ---------------------------------------------- Symbolic Q-Learning (Core logic from n_val_q_learn_symbolic, metrics added) ---------------------------------
/************************************************************************************************************************************************************* */



std::map<std::string, Action> symbolicQLearning(bool verbose = true,
                                               int sample_num_states_metrics = 1000, // How many states to sample for periodic metrics
                                               bool collect_metrics = true) {
    auto global_start_time = std::chrono::high_resolution_clock::now();
    std::map<std::string, Action> policy;

    // --- Basic Validation & Setup ---
    if (!manager || vars.empty()) {
        std::cerr << "Error: CUDD Manager or variables not initialized before symbolicQLearning." << std::endl;
        return {};
    }
    if (NUM_FACES <= 0 || NUM_DICE <= 0 || NUM_ACTIONS <= 0) {
        std::cerr << "Error: Invalid game parameters (Faces, Dice, or Actions <= 0)." << std::endl;
        return {};
    }
    if (verbose) {
        std::cout << "Starting Symbolic Q-Learning..." << std::endl;
        std::cout << "  Episodes: " << NUM_EPISODES << ", Alpha: " << ALPHA << ", Gamma: " << GAMMA << ", Epsilon: " << EPSILON << std::endl;
        std::cout << "  Terminal Reward: " << TERMINAL_REWARD << ", Step Cost: " << STEP_COST << std::endl;
    }

    // --- Re-create Terminal BDD ---
    // Deref existing one if it exists from a previous run (e.g., in testing) -- in doubt look through cudd folder files
    if (terminal_bdd) {
        Cudd_RecursiveDeref(manager, terminal_bdd);
        terminal_bdd = nullptr;
    }
    terminal_bdd = Cudd_ReadLogicZero(manager); Cudd_Ref(terminal_bdd);
    bool terminal_creation_error = false;
    for (int face_value = 0; face_value < NUM_FACES; ++face_value) {
        DdNode* all_same_val_bdd = Cudd_ReadOne(manager); Cudd_Ref(all_same_val_bdd); bool inner_error = false;
        for (int d = 0; d < NUM_DICE; ++d) {
            DdNode* die_val_bdd = createFaceValueBDD(manager, d, face_value, vars); // Referenced
            if (die_val_bdd == Cudd_ReadLogicZero(manager)) { 
                inner_error = true; 
                Cudd_RecursiveDeref(manager, all_same_val_bdd); break; } // Specific check
            DdNode* tmp_and = Cudd_bddAnd(manager, all_same_val_bdd, die_val_bdd);
            if (!tmp_and) { inner_error = true; Cudd_RecursiveDeref(manager, all_same_val_bdd); 
                Cudd_RecursiveDeref(manager, die_val_bdd); 
                break; }
            Cudd_Ref(tmp_and); 
            Cudd_RecursiveDeref(manager, all_same_val_bdd); 
            Cudd_RecursiveDeref(manager, die_val_bdd); 
            all_same_val_bdd = tmp_and;
        }
        if (inner_error) {
             terminal_creation_error = true; std::cerr << "Error creating 'all_same_val_bdd' for value " << face_value << std::endl;
             break; }
        DdNode* tmp_or = Cudd_bddOr(manager, terminal_bdd, all_same_val_bdd);
        if (!tmp_or) { 
            terminal_creation_error = true; Cudd_RecursiveDeref(manager, all_same_val_bdd); std::cerr << "Error: Cudd_bddOr failed." << std::endl; 
            break; }
        Cudd_Ref(tmp_or); Cudd_RecursiveDeref(manager, terminal_bdd); 
        Cudd_RecursiveDeref(manager, all_same_val_bdd); 
        terminal_bdd = tmp_or;
    }
    if (terminal_creation_error) { 
        std::cerr << "Aborting: terminal BDD creation error." << std::endl; if (terminal_bdd) Cudd_RecursiveDeref(manager, terminal_bdd); 
        return {}; }
    if (verbose) printNodeInfo("terminal_states_bdd (learning)", terminal_bdd);


    // --- Initialize Q-functions ---
    std::vector<DdNode*> q_functions(NUM_ACTIONS);
    DdNode* initial_q_add = Cudd_addConst(manager, 0.0);
     if (!initial_q_add) { 
        std::cerr << "Error: Failed to create initial Q-value ADD." << std::endl; Cudd_RecursiveDeref(manager, terminal_bdd); return {}; }
    Cudd_Ref(initial_q_add);
    for (int a = 0; a < NUM_ACTIONS; ++a) {
        q_functions[a] = initial_q_add;
        Cudd_Ref(q_functions[a]); // Reference for each action's Q-function
    }
    Cudd_RecursiveDeref(manager, initial_q_add); // Dereference the initial temporary ADD
    if (verbose) std::cout << "Initialized " << NUM_ACTIONS << " Q-functions to 0.0" << std::endl;


    // --- Setup for Learning Loop ---
    std::vector<std::vector<int>> allStates = generateAllStates(NUM_DICE, NUM_FACES);
    if (allStates.empty() && (NUM_DICE > 0 && NUM_FACES > 0)) {
        std::cerr << "Error: State generation failed. Aborting learning." << std::endl;
         // Clean up Q functions before returning
         for (auto& q : q_functions) if (q) Cudd_RecursiveDeref(manager, q);
         Cudd_RecursiveDeref(manager, terminal_bdd);
         return {};
    }
     if (verbose) std::cout << "Generated " << allStates.size() << " states for simulation and metrics." << std::endl;

    // Filter out terminal states for sampling starting states and metrics
    std::vector<std::vector<int>> non_terminal_states;
    non_terminal_states.reserve(allStates.size());
    for(const auto& s : allStates) {
        if (!isTerminalState(manager, terminal_bdd, s, vars)) {
            non_terminal_states.push_back(s);
        }
    }
     if (verbose) std::cout << "Filtered to " << non_terminal_states.size() << " non-terminal states." << std::endl;


    // RNG for action selection and state sampling
    std::random_device rd_agent;
    std::mt19937 agent_gen(rd_agent());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::uniform_int_distribution<> action_dis(0, NUM_ACTIONS - 1); // Action indices 0 to NUM_ACTIONS-1
    // Distribution for sampling non-terminal states (indices)
    std::uniform_int_distribution<> non_terminal_state_idx_dis;
     if (!non_terminal_states.empty()) {
         non_terminal_state_idx_dis = std::uniform_int_distribution<>(0, non_terminal_states.size() - 1);
     } else if (!allStates.empty()) {
          std::cout << "Warning: No non-terminal states found. Learning loop may not run." << std::endl;
     } else {
         std::cerr << "Error: No states available for learning." << std::endl;
         // Clean up Q functions before returning
         for (auto& q : q_functions) if (q) Cudd_RecursiveDeref(manager, q);
         Cudd_RecursiveDeref(manager, terminal_bdd);
         return {};
     }


    // Generate sample states for metrics calculation
    std::vector<std::vector<int>> sampled_states_for_metrics;
    if (collect_metrics && sample_num_states_metrics > 0 && !allStates.empty()) {
         sampled_states_for_metrics.reserve(sample_num_states_metrics);
         std::uniform_int_distribution<> metrics_state_idx_dis(0, allStates.size() - 1);
         std::vector<int> indices(allStates.size());
         std::iota(indices.begin(), indices.end(), 0);
         std::shuffle(indices.begin(), indices.end(), agent_gen);
         for (int i = 0; i < std::min((int)allStates.size(), sample_num_states_metrics); ++i) {
             sampled_states_for_metrics.push_back(allStates[indices[i]]);
         }
         if (verbose) std::cout << "Sampled " << sampled_states_for_metrics.size() << " states for periodic metrics." << std::endl;
     } else if (collect_metrics && sample_num_states_metrics > 0) {
          if (verbose) std::cout << "No states generated, cannot sample for metrics." << std::endl;
     }


    // Reserve space for metrics
    if (collect_metrics) {
        int num_metric_points = NUM_EPISODES / 100 + 1; // Estimate based on interval
        metrics.avg_q_values.reserve(num_metric_points);
        metrics.avg_decision_dag_sizes.reserve(num_metric_points);
        metrics.episode_times.reserve(NUM_EPISODES + 1);
        metrics.terminal_state_visits.reserve(NUM_EPISODES + 1); // Changed name
        metrics.bellman_errors.reserve(num_metric_points);
        metrics.memory_usage.reserve(num_metric_points);
        // metrics.avg_path_lengths removed
    }

    // --- Failure Counters for Debugging ---
    int createStateBDD_failures = 0; int evaluateADD_failures = 0; int addIte_failures = 0;
    int addConst_failures = 0; int bddToAdd_failures = 0; int terminal_check_failures = 0;

    // --- Logging setup ---
    const int PRINT_INTERVAL = std::max(1, NUM_EPISODES / 10);
    const int METRICS_INTERVAL = 100; // How often to calculate and store detailed metrics
    const int MAX_STEPS_TO_PRINT = 3; // Print details for first few steps of specific episodes


    // --- Main Learning Loop ---
    if (verbose) std::cout << "\nStarting learning loop..." << std::endl;
    for (int episode = 0; episode < NUM_EPISODES; episode++) {
        auto episode_start = std::chrono::high_resolution_clock::now();
        int terminal_visits_in_episode = 0; // Renamed from goal_visits

        bool print_this_episode = verbose && (episode < 1 || (episode + 1) % PRINT_INTERVAL == 0);
        if (print_this_episode && episode > 0) std::cout << std::endl;
        if (print_this_episode) std::cout << "--- Episode " << (episode + 1) << " ---" << std::endl;

        // Initialize state: Sample a non-terminal starting state
        std::vector<int> state;
         if (!non_terminal_states.empty()) {
             state = non_terminal_states[non_terminal_state_idx_dis(agent_gen)];
         } else {
             if (verbose) std::cout << "No non-terminal states to start episode, ending learning." << std::endl;
             break; // Stop if no non-terminal states exist
         }
         if (print_this_episode) std::cout << "  Start State: " << stateToString(state) << std::endl;


        int steps_in_episode = 0;
        const int MAX_EPISODE_STEPS = NUM_DICE * 30; // Limit episode length

        // Loop within episode until terminal state or max steps
        while (steps_in_episode < MAX_EPISODE_STEPS) {
            // Check for terminal state *before* taking action
            if (isTerminalState(manager, terminal_bdd, state, vars)) {
                terminal_visits_in_episode++;
                 if (print_this_episode && steps_in_episode < MAX_STEPS_TO_PRINT) {
                     std::cout << "  Step " << steps_in_episode +1 << ": Reached Terminal State " << stateToString(state) << std::endl;
                 }
                 // Episode ends when a terminal state is reached
                 break;
            }

            steps_in_episode++;
            bool print_this_step = print_this_episode && steps_in_episode <= MAX_STEPS_TO_PRINT;

            // --- Action Selection (Epsilon-Greedy) ---
            Action action;
            if (dis(agent_gen) < EPSILON) {
                action = action_dis(agent_gen); // Explore: random action
                 if (print_this_step) std::cout << "  Step " << steps_in_episode << ": State=" << stateToString(state) << " -> Explore Action=" << getActionName(action);
            } else {
                // Exploit: choose action with highest Q-value
                double maxQ = -std::numeric_limits<double>::infinity();
                action = 0; // Default action if all evaluations fail
                bool exploitSuccess = false; // Track if we found a valid Q value
                std::vector<Action> currentActions = getAvailableActions(state); // Should always be non-empty here

                for (Action a : currentActions) {
                    if (a < 0 || static_cast<size_t>(a) >= q_functions.size() || !q_functions[a]) continue;
                    double currentQ = evaluateADD(manager, q_functions[a], state, vars);

                    if (currentQ > EVAL_ERROR_THRESHOLD) { // Check for successful evaluation
                        if (!exploitSuccess || currentQ > maxQ) { // If first success or better Q
                            maxQ = currentQ;
                            action = a;
                            exploitSuccess = true;
                        }
                    } else {
                        evaluateADD_failures++; // Count evaluation failures during exploitation
                    }
                }
                // If exploitation failed (no valid Q values found), choose randomly
                if (!exploitSuccess) {
                    action = action_dis(agent_gen);
                     if (print_this_step) std::cout << "  Step " << steps_in_episode << ": State=" << stateToString(state) << " -> Exploit FAILED, Random Action=" << getActionName(action);
                 } else {
                     if (print_this_step) std::cout << "  Step " << steps_in_episode << ": State=" << stateToString(state) << " -> Exploit Action=" << getActionName(action) << " (Q=" << maxQ << ")";
                 }
            }
            // Add newline if action was printed
             if (print_this_step) std::cout << std::endl;


            // --- Symbolic Update (Using Expected Value Target) ---

            // 1. Calculate Expected Target Value: E[R + gamma * max_a' Q(s', a')]
            std::vector<Transition> transitions = generateTransitions(state, action);
            if (transitions.empty()) {
                 std::cerr << "Warning: No transitions generated for state " << stateToString(state) << ", action " << action << ". Skipping update." << std::endl;
                 // Decide how to proceed: break episode, or just skip step? Let's skip step for now.
                 // Need a strategy to pick a next state if transitions fail. Sample randomly?
                 if (!non_terminal_states.empty()) state = non_terminal_states[non_terminal_state_idx_dis(agent_gen)];
                 else break; // No non-terminal states left
                 continue;
            }

            double expected_target_q_value = 0.0;
            bool any_eval_error_in_expectation = false;

            for (const auto& transition : transitions) {
                const std::vector<int>& nextState = transition.nextState; // Use different name 'nextState'
                double probability = transition.probability;
                bool next_is_terminal = isTerminalState(manager, terminal_bdd, nextState, vars);

                // Determine reward R(s, a, s')
                double reward = next_is_terminal ? TERMINAL_REWARD : STEP_COST;

                // Determine max_a' Q(s', a')
                double maxNextQ = 0.0; // Value is 0 if next state is terminal
                if (!next_is_terminal) {
                    maxNextQ = -std::numeric_limits<double>::infinity();
                    bool evalSuccessNext = false;
                    std::vector<Action> nextActions = getAvailableActions(nextState);

                    for (Action next_a : nextActions) {
                        if (next_a < 0 || static_cast<size_t>(next_a) >= q_functions.size() || !q_functions[next_a]) continue;
                        double qVal = evaluateADD(manager, q_functions[next_a], nextState, vars);
                        if (qVal > EVAL_ERROR_THRESHOLD) {
                            if (!evalSuccessNext || qVal > maxNextQ) {
                                maxNextQ = qVal;
                                evalSuccessNext = true;
                            }
                        } else {
                            evaluateADD_failures++; // Count errors during expectation calc
                            // Don't set any_eval_error_in_expectation = true here,
                            // just proceed without this Q value if others are valid.
                        }
                    }
                    // If no valid Q-values were found for the next state, default maxNextQ to 0
                    if (!evalSuccessNext) {
                         maxNextQ = 0.0;
                         // Potentially set any_eval_error_in_expectation = true if this is critical
                         // For now, let's assume 0 if none are valid yet.
                         // any_eval_error_in_expectation = true; // Uncomment if failed eval should halt expectation
                    }
                } // end if !next_is_terminal

                expected_target_q_value += probability * (reward + GAMMA * maxNextQ);

            } // End loop over transitions for expectation

            // If critical evaluation failed during expectation, maybe skip update
            if (any_eval_error_in_expectation) {
                 if (print_this_step) std::cerr << "      Skipping update due to critical eval error during expectation calc." << std::endl;
                 // How to choose next state if skipping update? Use a random transition?
                 state = transitions[agent_gen() % transitions.size()].nextState; // Simple random pick
                 continue;
            }

            // Use the calculated expected value as the final target
            double target_q_value = expected_target_q_value;

            // 2. Create BDD for the current state 's'
            DdNode* stateBdd = createStateBDD(manager, state, vars); // Referenced BDD
            if (stateBdd == Cudd_ReadLogicZero(manager)) {
                createStateBDD_failures++;
                // Need a strategy for next state if BDD fails
                 state = transitions[agent_gen() % transitions.size()].nextState;
                 continue;
            }

            // 3. Convert state BDD to a 0-1 ADD (where state is true, value is 1)
            DdNode* stateAdd = Cudd_BddToAdd(manager, stateBdd);
            if (!stateAdd) {
                bddToAdd_failures++;
                Cudd_RecursiveDeref(manager, stateBdd);
                state = transitions[agent_gen() % transitions.size()].nextState;
                continue;
            }
            Cudd_Ref(stateAdd);
            Cudd_RecursiveDeref(manager, stateBdd); // Deref the BDD now that we have the ADD

            // 4. Get current Q-add for the chosen action and evaluate old Q value
            DdNode* current_q_add = q_functions[action];
            double oldQ_value = evaluateADD(manager, current_q_add, state, vars);
            if (oldQ_value < EVAL_ERROR_THRESHOLD) {
                evaluateADD_failures++;
                Cudd_RecursiveDeref(manager, stateAdd);
                state = transitions[agent_gen() % transitions.size()].nextState;
                continue;
            }

            // 5. Calculate the new scalar Q value for this specific state 's'
            double newQ_s_value = (1.0 - ALPHA) * oldQ_value + ALPHA * target_q_value;
            if (print_this_step) {
                 std::cout << "      OldQ=" << oldQ_value << ", ExpTarget=" << target_q_value
                           << ", NewScalarQ=" << newQ_s_value << std::endl;
             }

            // 6. Create an ADD constant representing this new scalar value
            DdNode* newQ_s_add = Cudd_addConst(manager, newQ_s_value);
            if (!newQ_s_add) {
                addConst_failures++;
                Cudd_RecursiveDeref(manager, stateAdd);
                state = transitions[agent_gen() % transitions.size()].nextState;
                continue;
            }
            Cudd_Ref(newQ_s_add);

            // 7. Update the Q-function ADD using ITE (If-Then-Else)
            // updatedQ = ITE(stateADD, newQ_s_add, current_q_add)
            // Meaning: IF in state 's' (stateAdd is 1), THEN use newQ_s_add, ELSE keep current_q_add
            DdNode* updatedQAdd = Cudd_addIte(manager, stateAdd, newQ_s_add, current_q_add);
            if (!updatedQAdd) {
                addIte_failures++;
                Cudd_RecursiveDeref(manager, stateAdd);
                Cudd_RecursiveDeref(manager, newQ_s_add);
                state = transitions[agent_gen() % transitions.size()].nextState;
                continue;
            }
            Cudd_Ref(updatedQAdd);

            // 8. Dereference intermediate ADDs used in ITE
            Cudd_RecursiveDeref(manager, stateAdd);   // stateAdd was the condition
            Cudd_RecursiveDeref(manager, newQ_s_add); // newQ_s_add was the 'then' branch

            // 9. Replace the old Q-function ADD with the new one
            Cudd_RecursiveDeref(manager, q_functions[action]); // Deref the old Q-function ADD
            q_functions[action] = updatedQAdd; // Assign the new referenced ADD

            // --- Transition to Next State ---
            // Standard Q-learning samples ONE transition to determine the next state
            // Even though the update used expected value over all transitions.
            double rand_val = dis(agent_gen);
            double cumulative_prob = 0.0;
            std::vector<int> sampledNextState;
            if (!transitions.empty()) {
                sampledNextState = transitions.back().nextState; // Default if probabilities don't sum perfectly
                for (const auto& t : transitions) {
                    cumulative_prob += t.probability;
                    if (rand_val <= cumulative_prob || &t == &transitions.back()) {
                        sampledNextState = t.nextState;
                        break;
                    }
                }
            } else {
                 // Should not happen if check at start of update passed, but handle defensively
                 if (!non_terminal_states.empty()) sampledNextState = non_terminal_states[non_terminal_state_idx_dis(agent_gen)];
                 else break; // No non-terminal states left
            }
            state = sampledNextState; // Update state for the next iteration


        } // End while steps < MAX_EPISODE_STEPS

        if (print_this_episode && steps_in_episode >= MAX_EPISODE_STEPS) {
            std::cout << "  Episode " << (episode + 1) << " finished due to MAX_STEPS." << std::endl;
        }

        // --- Track Metrics for this Episode ---
        if (collect_metrics) {
            metrics.terminal_state_visits.push_back(terminal_visits_in_episode);

            // Collect timing info
            auto episode_end = std::chrono::high_resolution_clock::now();
            double episode_duration = std::chrono::duration<double>(episode_end - episode_start).count();
            metrics.episode_times.push_back(episode_duration);

            // Collect other metrics periodically
            if ((episode + 1) % METRICS_INTERVAL == 0 && !sampled_states_for_metrics.empty()) {
                // Average Q-value
                double avg_q = calculateAverageQValue(q_functions, sampled_states_for_metrics);
                metrics.avg_q_values.push_back(avg_q);

                // Average DAG size
                double avg_dag_size = calculateAverageDAGSize(q_functions);
                metrics.avg_decision_dag_sizes.push_back(avg_dag_size); // Store as double

                // Bellman error
                double bellman_error = calculateBellmanError(q_functions, sampled_states_for_metrics, terminal_bdd);
                metrics.bellman_errors.push_back(bellman_error);

                // Memory usage (estimate based on node count)
                // Size of DdNode is internal to CUDD, using a rough estimate (e.g., 32-40 bytes)
                // Let's use sizeof(DdNode) if available, otherwise a guess. CuddNode is typically around 32 bytes.
                // We use the internal CuddNode structure size for a better estimate.
                double mem_usage = avg_dag_size * sizeof(DdNode) / (1024.0 * 1024.0); // MB
                metrics.memory_usage.push_back(mem_usage);

                if (verbose && print_this_episode) { // Also print metrics if it's a print interval episode
                    std::cout << "  Metrics at episode " << (episode + 1) << ":" << std::endl;
                    std::cout << "    Avg Q-value: " << avg_q << std::endl;
                    std::cout << "    Avg DAG size: " << avg_dag_size << " nodes" << std::endl;
                    std::cout << "    Bellman error: " << bellman_error << std::endl;
                    std::cout << "    Est. ADD memory: " << mem_usage << " MB" << std::endl;
                }
            } else if ((episode + 1) % METRICS_INTERVAL == 0) {
                 // Push default values if metrics couldn't be calculated (e.g., no sampled states)
                 metrics.avg_q_values.push_back(0.0);
                 metrics.avg_decision_dag_sizes.push_back(calculateAverageDAGSize(q_functions)); // Still calc DAG
                 metrics.bellman_errors.push_back(0.0);
                 metrics.memory_usage.push_back(calculateAverageDAGSize(q_functions) * sizeof(DdNode) / (1024.0 * 1024.0));
                 if (verbose && print_this_episode) {
                      std::cout << "  Metrics skipped at episode " << (episode + 1) << " (no sampled states)." << std::endl;
                 }
            }
        }

        // Optional: Add periodic garbage collection if memory becomes an issue
        // if ((episode + 1) % 1000 == 0) {
        //     Cudd_ReduceHeap(manager, CUDD_REORDER_SAME, 0); // Reordering might be slow
        //     if (verbose) std::cout << "  Performed CUDD garbage collection at episode " << episode + 1 << std::endl;
        // }

    } // End for episodes

    if (verbose) {
         std::cout << "\nSymbolic Q-Learning completed." << std::endl;
         std::cout << "  Failure Counts:" << std::endl;
         std::cout << "    createStateBDD: " << createStateBDD_failures << std::endl;
         std::cout << "    evaluateADD: " << evaluateADD_failures << std::endl;
         std::cout << "    Cudd_BddToAdd: " << bddToAdd_failures << std::endl;
         std::cout << "    Cudd_addConst: " << addConst_failures << std::endl;
         std::cout << "    Cudd_addIte: " << addIte_failures << std::endl;
         std::cout << "    isTerminalState (during learning): " << terminal_check_failures << std::endl; // Might be 0 if check is robust
    }

    // Step 8: Extract policy from the learned Q-functions
    if (verbose) std::cout << "Extracting policy..." << std::endl;
    policy.clear(); // Clear any previous policy
    if (!allStates.empty()) {
        for (const auto& s : allStates) {
            std::string stateStr = stateToString(s);
            if (isTerminalState(manager, terminal_bdd, s, vars)) {
                policy[stateStr] = INVALID_ACTION; // Mark terminal states
            } else {
                // Find the best action for non-terminal states
                Action bestAction = 0; // Default to action 0
                double maxQ = -std::numeric_limits<double>::infinity();
                bool policyEvalSuccess = false; // Track if any valid Q found for this state

                std::vector<Action> currentActions = getAvailableActions(s);
                if (currentActions.empty() && !isTerminalState(manager, terminal_bdd, s, vars)) {
                     std::cerr << "Warning: Non-terminal state " << stateStr << " has no actions during policy extraction." << std::endl;
                     policy[stateStr] = 0; // Assign default action
                     continue;
                }

                for (Action a : currentActions) {
                    if (a < 0 || static_cast<size_t>(a) >= q_functions.size() || !q_functions[a]) continue;

                    double qVal = evaluateADD(manager, q_functions[a], s, vars);
                    if (qVal > EVAL_ERROR_THRESHOLD) { // Check for successful evaluation
                        if (!policyEvalSuccess || qVal > maxQ) {
                            maxQ = qVal;
                            bestAction = a;
                            policyEvalSuccess = true;
                        }
                    }
                    // No need to count failures here, just find the best valid one
                }

                // Assign the best action found, or default if none were valid
                policy[stateStr] = policyEvalSuccess ? bestAction : 0;

                // Optional: Print some extracted policy entries for debugging
                // if (verbose && policy.size() < 20 && policyEvalSuccess) {
                //     std::cout << "  Policy Extracted: State " << stateStr << " -> Action "
                //               << getActionName(bestAction) << " (Q=" << maxQ << ")" << std::endl;
                // } else if (verbose && policy.size() < 20 && !policyEvalSuccess) {
                //      std::cout << "  Policy Extracted: State " << stateStr << " -> Action 0 (Default - Eval Failed)" << std::endl;
                // }
            }
        }
         if (verbose) std::cout << "Policy extracted for " << policy.size() << " states." << std::endl;
    } else {
        if (verbose) std::cout << "Warning: No states generated, policy map will be empty." << std::endl;
    }


    // Step 9: Clean up Q-functions and terminal BDD
    if (verbose) std::cout << "Cleaning up Q-functions and terminal BDD..." << std::endl;
    for (auto& q_add : q_functions) {
        if (q_add) Cudd_RecursiveDeref(manager, q_add);
    }
    q_functions.clear(); // Clear the vector itself

    if (terminal_bdd) {
        Cudd_RecursiveDeref(manager, terminal_bdd);
        terminal_bdd = nullptr; // Set to null after deref
    }

    auto global_end_time = std::chrono::high_resolution_clock::now();
    double total_duration = std::chrono::duration<double>(global_end_time - global_start_time).count();
    if (verbose) {
        std::cout << "Symbolic Q-Learning function completed in " << total_duration << " seconds." << std::endl;
    }

    // Final check before returning
    if (policy.empty() && !(NUM_DICE > 0 && NUM_FACES > 0 && pow(NUM_FACES, NUM_DICE) <= 1)) {
         std::cerr << "Warning: Returning empty policy map from symbolicQLearning, but state space expected > 1." << std::endl;
     }
    return policy;
}


// --- Simulation and Main Function ---

// Run simulation using the learned policy
void runSimulation(const std::map<std::string, Action>& policy, int numTrials, bool verbose = true) {
     if (policy.empty()) { std::cerr << "Cannot run simulation: Policy map is empty." << std::endl; return; }
     if (NUM_FACES <= 0) { std::cerr << "Cannot run simulation: NUM_FACES <= 0." << std::endl; return; }

     // Generate states needed for simulation start
     std::vector<std::vector<int>> allStates = generateAllStates(NUM_DICE, NUM_FACES);
     std::vector<std::vector<int>> non_terminal_states_sim;
      if (!allStates.empty()) {
            // Recreate terminal BDD if necessary (should exist if learning ran)
             if (!terminal_bdd) {
                  std::cerr << "Warning: Terminal BDD not found for simulation, attempting to recreate." << std::endl;
                  // Recreate logic (simplified, assumes manager and vars are still valid conceptually)
                  // This is risky if manager/vars were cleaned up elsewhere. Best practice is to pass it.
                  // Let's assume it's available or simulation might fail.
                  // For safety, maybe just rely on allSameValue if BDD isn't passed.
             }

             non_terminal_states_sim.reserve(allStates.size());
             for(const auto& s : allStates) {
                 // Use direct check if BDD is unreliable here
                  if (!allSameValue(s)) {
                      non_terminal_states_sim.push_back(s);
                  }
             }
      }

     if (non_terminal_states_sim.empty() && !allStates.empty()) {
         std::cerr << "Cannot run simulation: No non-terminal states found to start from." << std::endl;
          return;
     }
      if (allStates.empty() && (NUM_DICE > 0 && NUM_FACES > 0)){
          std::cerr << "Cannot run simulation: Failed to generate states." << std::endl;
          return;
      }
       if (non_terminal_states_sim.empty() && allStates.empty()){
            // Trivial case (e.g., 0 dice), simulation is meaningless.
             std::cout << "Simulation skipped: No states in the system." << std::endl;
             return;
       }


    std::random_device rd_sim; std::mt19937 sim_gen(rd_sim());
    std::uniform_int_distribution<> face_dis(0, NUM_FACES - 1);
    std::uniform_int_distribution<> start_state_idx_dis(0, non_terminal_states_sim.size() - 1);


    long long totalSteps = 0;
    int minSteps = std::numeric_limits<int>::max();
    int maxSteps = 0;
    int successCount = 0;
    int timeoutCount = 0;
    int policyErrorCount = 0; // Count states not found in policy

    if (verbose) std::cout << "\n--- Starting Simulation (" << numTrials << " trials) ---" << std::endl;

    for (int trial = 0; trial < numTrials; trial++) {
        // Sample a non-terminal starting state
        std::vector<int> state = non_terminal_states_sim[start_state_idx_dis(sim_gen)];
        int steps = 0;
        const int MAX_STEPS = NUM_DICE * 50; // Simulation step limit (can be higher than learning)
        bool reachedTerminal = false;
        bool errorOccurred = false;

        if (verbose && trial < 3) {
            std::cout << "\n--- Trial " << (trial + 1) << " Start: " << stateToString(state) << " ---" << std::endl;
        }

        while (steps < MAX_STEPS) {
            // Check for terminal state using direct method (safer if BDD state is uncertain)
             if (allSameValue(state)) {
                 reachedTerminal = true;
                 successCount++;
                 totalSteps += steps;
                 minSteps = std::min(minSteps, steps);
                 maxSteps = std::max(maxSteps, steps);
                 if (verbose && trial < 3) {
                     std::cout << "  Terminal state reached in " << steps << " steps!" << std::endl;
                 }
                 break; // End this trial
             }

            steps++; // Increment step count for non-terminal states

            std::string stateStr = stateToString(state);
            auto policy_it = policy.find(stateStr);

            // Check if state exists in the policy map
            if (policy_it == policy.end()) {
                std::cerr << "Error: State " << stateStr << " not found in policy map during simulation trial " << trial + 1 << "." << std::endl;
                policyErrorCount++;
                errorOccurred = true;
                break; // End this trial due to error
            }

            Action action = policy_it->second;

            // Check if the action itself is valid (should be INVALID_ACTION only for terminals)
            if (action == INVALID_ACTION) {
                 // This should only happen if a state marked terminal in policy isn't caught by allSameValue
                 std::cerr << "Warning: Policy has INVALID_ACTION for non-terminal state " << stateStr << " in trial " << trial + 1 << "." << std::endl;
                 // Treat as error or timeout? Let's count as error for now.
                 policyErrorCount++;
                 errorOccurred = true;
                 break;
            }
            if (action < 0 || action >= NUM_FACES) {
                 std::cerr << "Error: Policy contains invalid action " << action << " for state " << stateStr << " in trial " << trial + 1 << "." << std::endl;
                 policyErrorCount++;
                 errorOccurred = true;
                 break;
            }


             if (verbose && trial < 3) {
                  std::cout << "  Sim Step " << steps << ": S=" << stateStr
                            << " -> A=" << getActionName(action);
              }

            // Simulate the environment transition based on the chosen action
            int valueToKeep = action;
            std::vector<int> nextStateVec = state; // Start with current state
            bool diceWereRerolled = false;
            for (size_t i = 0; i < state.size(); i++) {
                if (state[i] != valueToKeep) {
                    nextStateVec[i] = face_dis(sim_gen); // Reroll this die
                    diceWereRerolled = true;
                }
            }

            // If no dice were rerolled, the state technically stays the same.
            // This could lead to infinite loops if the policy always chooses to keep the current state.
            // The MAX_STEPS limit handles this.
            if (!diceWereRerolled && !allSameValue(state)) {
                 // This case means the policy chose to keep a value, and all dice already had that value.
                 // This should lead to a terminal state check in the next iteration.
                 if (verbose && trial < 3) {
                      std::cout << " (No dice rerolled)" << std::endl;
                 }
            } else {
                 if (verbose && trial < 3) {
                      std::cout << " -> S'=" << stateToString(nextStateVec) << std::endl;
                 }
            }

            state = nextStateVec; // Update the state

        } // End while steps < MAX_STEPS

        // Handle timeout
        if (!reachedTerminal && !errorOccurred && steps >= MAX_STEPS) {
            timeoutCount++;
            if (verbose && trial < 3) {
                std::cout << "  Timeout after " << MAX_STEPS << " steps." << std::endl;
            }
        }
    } // End for trials

    // Print Simulation Results Summary
    double avgSteps = successCount > 0 ? static_cast<double>(totalSteps) / successCount : 0;
    double successRate = (numTrials > 0) ? static_cast<double>(successCount) / numTrials * 100.0 : 0.0;

    std::cout << "\n--- Simulation Results (" << numTrials << " trials) ---" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Success Rate:        " << successCount << "/" << numTrials
              << " (" << successRate << "%)" << std::endl;
    std::cout << "  Timeouts:            " << timeoutCount << "/" << numTrials << std::endl;
    std::cout << "  Policy Errors:       " << policyErrorCount << "/" << numTrials << std::endl; // States not found or invalid action

    if (successCount > 0) {
        std::cout << "  Avg Steps (success): " << avgSteps << std::endl;
        std::cout << "  Min Steps (success): " << minSteps << std::endl;
        std::cout << "  Max Steps (success): " << maxSteps << std::endl;
    } else {
        std::cout << "  No trials finished successfully." << std::endl;
    }
    std::cout << "------------------------------------------" << std::endl;
}

// Print the learned policy (similar to grid world version)
void printPolicy(const std::map<std::string, Action>& policy) {
    std::cout << "\n--- Learned Policy ---" << std::endl;
    // Adjust width based on state string length
    int stateWidth = std::max(12, NUM_DICE * 2 + 4); // e.g., "[x,y,z] " + buffer
    std::cout << std::setw(stateWidth) << std::left << "State"
              << std::setw(18) << std::left << "Action (Keep Value)" << std::endl;
    std::cout << std::string(stateWidth + 18, '-') << std::endl;

    // Sort states for consistent output order
    std::vector<std::string> stateStrs;
    stateStrs.reserve(policy.size());
    for (const auto& entry : policy) stateStrs.push_back(entry.first);
    std::sort(stateStrs.begin(), stateStrs.end());

    int printedCount = 0;
    const int MAX_PRINT_POLICY = 50; // Limit how many policy entries to print
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
                          << std::setw(18) << std::left << getActionName(action) << std::endl;
                printedCount++;
            }
        }
    }

    if (nonTerminalCount > printedCount) {
        std::cout << "... (omitting " << (nonTerminalCount - printedCount) << " non-terminal states)" << std::endl;
    }

    if (terminalCount > 0) {
        std::cout << std::setw(stateWidth) << std::left << "<Terminal States>"
                  << std::setw(18) << std::left << "Terminal" << std::endl;
    }

    std::cout << std::string(stateWidth + 18, '-') << std::endl;
    std::cout << "Total states in policy: " << policy.size()
              << " (" << nonTerminalCount << " non-terminal, "
              << terminalCount << " terminal)" << std::endl;
    std::cout << "---------------------------" << std::endl;
}

// Save metrics to CSV files (adapted from grid world)
void saveMetricsToCSV() {
    if (!metrics.avg_q_values.empty()) { // Check if metrics were collected
        std::cout << "Saving metrics to CSV files..." << std::endl;

        // Save average Q-values
        {
            std::ofstream file("add_q_dice_avg_values.csv");
            file << "Episode,AvgQValue" << std::endl;
            for (size_t i = 0; i < metrics.avg_q_values.size(); ++i) {
                file << (i + 1) * METRICS_INTERVAL << "," << metrics.avg_q_values[i] << std::endl;
            }
        }

        // Save DAG sizes
        {
            std::ofstream file("add_q_dice_dag_sizes.csv");
            file << "Episode,AvgNodeCount" << std::endl;
            for (size_t i = 0; i < metrics.avg_decision_dag_sizes.size(); ++i) {
                file << (i + 1) * METRICS_INTERVAL << "," << metrics.avg_decision_dag_sizes[i] << std::endl;
            }
        }

        // Save Bellman errors
        {
            std::ofstream file("add_q_dice_bellman_errors.csv");
            file << "Episode,AvgError" << std::endl;
            for (size_t i = 0; i < metrics.bellman_errors.size(); ++i) {
                file << (i + 1) * METRICS_INTERVAL << "," << metrics.bellman_errors[i] << std::endl;
            }
        }

        // Save memory usage
        {
            std::ofstream file("add_q_dice_memory_usage.csv");
            file << "Episode,MemoryMB" << std::endl;
            for (size_t i = 0; i < metrics.memory_usage.size(); ++i) {
                file << (i + 1) * METRICS_INTERVAL << "," << metrics.memory_usage[i] << std::endl;
            }
        }

        // Save terminal state visits
        {
            std::ofstream file("add_q_dice_terminal_visits.csv");
            file << "Episode,TerminalVisits" << std::endl;
            for (size_t i = 0; i < metrics.terminal_state_visits.size(); ++i) {
                file << i + 1 << "," << metrics.terminal_state_visits[i] << std::endl;
            }
        }

        // Save episode times
        {
            std::ofstream file("add_q_dice_episode_times.csv");
            file << "Episode,DurationSeconds" << std::endl;
            for (size_t i = 0; i < metrics.episode_times.size(); ++i) {
                file << i + 1 << "," << metrics.episode_times[i] << std::endl;
            }
        }

        std::cout << "Metrics saved successfully." << std::endl;
    } else {
        std::cout << "No metrics collected, CSV files not generated." << std::endl;
    }
}


// Print help message (adapted from grid world)
void printHelp() {
    std::cout << "Usage: add_q_dice_game [options]\n";
    std::cout << "Options:\n";
    std::cout << "  -d N          Set number of dice to N (default: " << NUM_DICE << ")\n";
    std::cout << "  -f N          Set number of faces per die to N (default: " << NUM_FACES << ")\n";
    std::cout << "  -e N          Set number of episodes to N (default: " << NUM_EPISODES << ")\n";
    std::cout << "  -a F          Set alpha learning rate to F (default: " << ALPHA << ")\n";
    std::cout << "  -g F          Set gamma discount factor to F (default: " << GAMMA << ")\n";
    std::cout << "  -eps F        Set epsilon for exploration to F (default: " << EPSILON << ")\n";
    std::cout << "  -r F          Set terminal reward to F (default: " << TERMINAL_REWARD << ")\n";
    std::cout << "  -cost F       Set step cost (penalty) to F (default: " << STEP_COST << ")\n";
    std::cout << "  -sim N        Run N simulation trials after learning (default: 0)\n";
    std::cout << "  -metrics N    Collect metrics and save to CSV files (sample N states for calc, default 1000)\n";
    std::cout << "  -v            Enable verbose mode\n";
    std::cout << "  -h            Print this help message\n";
}

int main(int argc, char* argv[]) {
    bool verbose = false;
    int numSimTrials = 0;
    bool collect_and_save_metrics = false;
    int metrics_sample_size = 1000; // Default sample size for metric calculation

    // --- Argument Parsing (Adapted from Grid World) ---
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        try {
            if (arg == "-v") {
                verbose = true;
            } else if (arg == "-metrics" && i + 1 < argc) {
                 collect_and_save_metrics = true;
                 metrics_sample_size = std::stoi(argv[++i]);
                 if (metrics_sample_size < 0) metrics_sample_size = 0;
            } else if (arg == "-metrics") { // Handle -metrics without value
                 collect_and_save_metrics = true;
                 // Keep default metrics_sample_size
            } else if (arg == "-d" && i + 1 < argc) {
                NUM_DICE = std::stoi(argv[++i]);
                if (NUM_DICE <= 0) { std::cerr << "Number of dice must be positive.\n"; return 1; }
            } else if (arg == "-f" && i + 1 < argc) {
                NUM_FACES = std::stoi(argv[++i]);
                 if (NUM_FACES <= 0) { std::cerr << "Number of faces must be positive.\n"; return 1; }
            } else if (arg == "-e" && i + 1 < argc) {
                NUM_EPISODES = std::stoi(argv[++i]);
                if (NUM_EPISODES <= 0) { std::cerr << "Number of episodes must be positive.\n"; return 1; }
            } else if (arg == "-a" && i + 1 < argc) {
                ALPHA = std::stod(argv[++i]);
                if (ALPHA <= 0 || ALPHA > 1) { std::cerr << "ALPHA must be in (0, 1].\n"; return 1; }
            } else if (arg == "-g" && i + 1 < argc) {
                GAMMA = std::stod(argv[++i]);
                if (GAMMA < 0 || GAMMA > 1) { std::cerr << "GAMMA must be in [0, 1].\n"; return 1; }
            } else if (arg == "-eps" && i + 1 < argc) {
                EPSILON = std::stod(argv[++i]);
                if (EPSILON < 0 || EPSILON > 1) { std::cerr << "EPSILON must be in [0, 1].\n"; return 1; }
            } else if (arg == "-r" && i + 1 < argc) {
                 TERMINAL_REWARD = std::stod(argv[++i]);
             } else if (arg == "-cost" && i + 1 < argc) {
                  STEP_COST = std::stod(argv[++i]);
              }else if (arg == "-sim" && i + 1 < argc) {
                numSimTrials = std::stoi(argv[++i]);
                if (numSimTrials < 0) numSimTrials = 0;
            } else if (arg == "-h" || arg == "--help") {
                printHelp();
                return 0;
            } else {
                std::cerr << "Unknown or incomplete option: " << arg << "\n";
                printHelp();
                return 1;
            }
        } catch (const std::invalid_argument& e) {
            std::cerr << "Invalid number format for option " << arg << ": " << argv[i] << " (" << e.what() << ")" << std::endl;
            return 1;
        } catch (const std::out_of_range& e) {
            std::cerr << "Number out of range for option " << arg << ": " << argv[i] << " (" << e.what() << ")" << std::endl;
            return 1;
        }
    }

    // --- Calculate Derived Parameters ---
    VARS_PER_DIE = calculateVarsPerDie(NUM_FACES);
    TOTAL_BDD_VARS = NUM_DICE * VARS_PER_DIE;
    NUM_ACTIONS = NUM_FACES; // Action is choosing which face value to keep

    // Check if parameters are feasible
    if (TOTAL_BDD_VARS > 30) { // Arbitrary limit, CUDD handles more but gets slow
         std::cerr << "Warning: TOTAL_BDD_VARS (" << TOTAL_BDD_VARS << ") is large. Execution might be very slow or fail." << std::endl;
    }
     if (NUM_FACES == 1 && NUM_DICE > 0) {
         std::cout << "Info: Only one face per die. The game is trivial (always terminal)." << std::endl;
         // Learning and simulation are pointless here, but allow execution for consistency.
     } else if (NUM_ACTIONS <= 0 && NUM_FACES > 0) {
          std::cerr << "Error: Calculated NUM_ACTIONS is " << NUM_ACTIONS << " which is invalid." << std::endl;
          return 1;
     }

    // --- Print Parameters ---
    std::cout << "--- Dice Game Parameters ---" << std::endl;
    std::cout << "  Number of Dice: " << NUM_DICE << std::endl;
    std::cout << "  Number of Faces: " << NUM_FACES << std::endl;
    std::cout << "  BDD Vars per Die (Log2 Enc): " << VARS_PER_DIE << std::endl;
    std::cout << "  Total BDD Variables: " << TOTAL_BDD_VARS << std::endl;
    std::cout << "  Number of Actions (Keep Face): " << NUM_ACTIONS << std::endl;
    std::cout << "  Gamma: " << GAMMA << ", Epsilon: " << EPSILON << ", Alpha: " << ALPHA << std::endl;
    std::cout << "  Episodes: " << NUM_EPISODES << std::endl;
    std::cout << "  Terminal Reward: " << TERMINAL_REWARD << ", Step Cost: " << STEP_COST << std::endl;
    std::cout << "  Collect Metrics: " << (collect_and_save_metrics ? "Yes" : "No") << std::endl;
    if(collect_and_save_metrics) std::cout << "  Metrics Sample Size: " << metrics_sample_size << std::endl;
    std::cout << "--------------------------" << std::endl;


    // --- Initialize CUDD Manager ---
    // Using default slots, can be tuned if needed
    manager = Cudd_Init(TOTAL_BDD_VARS, 0, CUDD_UNIQUE_SLOTS, CUDD_CACHE_SLOTS, 0);
    if (!manager) {
        std::cerr << "Failed to initialize CUDD manager.\n";
        return 1;
    }

    // Optional: Enable automatic dynamic reordering (can sometimes help, sometimes hurt)
    // Cudd_AutodynEnable(manager, CUDD_REORDER_SIFT);

    // --- Create BDD Variables ---
    vars.resize(TOTAL_BDD_VARS);
    bool vars_ok = true;
    for (int i = 0; i < TOTAL_BDD_VARS; ++i) {
        vars[i] = Cudd_bddIthVar(manager, i);
        // Cudd_bddIthVar returns a non-referenced pointer managed by CUDD. No Ref needed.
        if (!vars[i]) {
            vars_ok = false;
            std::cerr << "Failed to create BDD variable " << i << ".\n";
            break;
        }
    }

    if (!vars_ok) {
        std::cerr << "Aborting due to BDD variable creation failure.\n";
        Cudd_Quit(manager);
        return 1;
    }
    if (verbose) std::cout << "Created " << TOTAL_BDD_VARS << " BDD variables." << std::endl;


    // --- Perform Symbolic Q-learning ---
    std::map<std::string, Action> policy = symbolicQLearning(verbose, metrics_sample_size, collect_and_save_metrics);

    // --- Post-Learning ---
    if (policy.empty() && !(NUM_DICE > 0 && NUM_FACES > 0 && pow(NUM_FACES, NUM_DICE) <= 1)) {
         std::cerr << "Learning failed (returned empty policy for non-trivial game). Aborting." << std::endl;
         // BDD variables are managed by CUDD, manager quit will handle them.
         Cudd_Quit(manager);
         return 1;
     } else if (policy.empty()) {
          std::cout << "Learning resulted in an empty policy (likely trivial game)." << std::endl;
     }


    // Print the learned policy
    printPolicy(policy);

    // Run simulation if requested
    if (numSimTrials > 0) {
        // We need the terminal BDD for simulation checks ideally,
        // but symbolicQLearning cleans it up. We'll rely on allSameValue in simulation.
        runSimulation(policy, numSimTrials, verbose);
    } else {
        std::cout << "\nRun with '-sim N' to simulate the learned policy." << std::endl;
    }

    // Save metrics to CSV if requested
    if (collect_and_save_metrics) {
        saveMetricsToCSV();
    }

    // --- Clean up CUDD ---
    if (verbose) std::cout << "Cleaning up CUDD manager..." << std::endl;
    // BDD variables in `vars` are managed by CUDD, no need to dereference.
    vars.clear(); // Clear the vector of pointers

    int check = Cudd_CheckZeroRef(manager);
    if (check != 0 && verbose) {
        std::cerr << "Warning: " << check << " CUDD nodes still referenced after cleanup." << std::endl;
    }
    Cudd_Quit(manager);
    manager = nullptr; // Set manager to null after quitting

    if (verbose) std::cout << "Execution finished." << std::endl;
    return 0;
}