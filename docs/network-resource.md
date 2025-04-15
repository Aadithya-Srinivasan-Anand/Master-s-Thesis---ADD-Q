# Network Resource Balancing ADD-Q Implementation

## Overview

The Network Resource Balancing implementation demonstrates the ADD-Q algorithm in a complex resource allocation environment. It models the challenge of managing multiple network sessions with limited bandwidth, showcasing how symbolic reinforcement learning can handle complex state transitions and constraints.

## Environment Description

1. The system manages N network sessions, each in one of four states: IDLE, REQUESTING, TRANSFERRING, or BLOCKED.
2. Each session requires a specific amount of bandwidth when TRANSFERRING.
3. There is a maximum total bandwidth constraint for all sessions combined.
4. Sessions transition probabilistically between states based on environment dynamics.
5. The agent can either WAIT or try to UNBLOCK a specific blocked session.
6. The goal is to maintain a minimum number of TRANSFERRING sessions at all times.

## State Space

The state space consists of all possible combinations of session states. With N sessions and 4 possible states per session, the state space size is 4^N. For example, with 8 sessions, there are 4^8 = 65,536 possible states.

## Task Explanation

Imagine you're managing internet connections for a company. Multiple users want to download files, but your network has limited bandwidth. When users request downloads, they can be either granted bandwidth (TRANSFERRING) or put on hold (BLOCKED) if there's not enough bandwidth available. Users may also finish their downloads (returning to IDLE) or make new requests over time.

As the network manager, you have two options on each turn:
1. WAIT and see how the network changes naturally
2. Try to UNBLOCK a specific user who's waiting, if there's now enough bandwidth available

Your goal is to keep at least a certain number of users actively transferring data at all times, maintaining efficient network utilization.

The challenge is that there are thousands of possible network states with different combinations of users being idle, requesting, transferring, or blocked. The ADD-Q algorithm learns which actions are best in each situation while cleverly grouping similar network states together to make the problem manageable.

## Key Components

### State Representation

The code uses a binary encoding for session states:

- **getVarIndex**: Maps from session index and bit index to BDD variable index
- **getSessionStateValue**: Extracts session state from the state vector
- **setSessionStateValue**: Sets session state in the state vector

### Environment Dynamics

- **calculateBandwidthUsage**: Computes total bandwidth used by all transferring sessions
- **applyAgentAction**: Applies agent actions (waiting or unblocking)
- **environment_step**: Simulates the probabilistic evolution of session states

### Symbolic Operations

- **is_session_in_state_bdd**: Creates a BDD for a session in a specific state
- **createGoalBDD**: Creates a BDD representing the goal (minimum transferring sessions)
- **evaluateADD**: Evaluates an ADD for a specific network state
- **StateCache**: Caches BDDs for efficiency

### Learning Algorithm

- **symbolicQLearning**: Main function implementing the ADD-Q algorithm
- **generateRandomState**: Creates random valid states for exploration
- **calculateAverageQValue**: Computes average Q-values
- **calculateAverageDAGSize**: Measures decision diagram sizes
- **calculateBellmanError**: Computes Bellman error to assess learning

### Visualization and Testing

- **printPolicy**: Displays the learned policy
- **runSimulation**: Tests the policy through simulations
- **saveMetricsToCSV**: Exports metrics to CSV files

## Usage Examples

```bash
# Basic run with 8 sessions, bandwidth limit of 15, goal of 2+ transferring sessions
./add_q_network -n 8 -bw 15 -goal 2 -e 35000 -v

# With metrics collection
./add_q_network -n 8 -bw 15 -goal 2 -e 35000 -metrics 1000 -v

# Run with simulation to test policy
./add_q_network -n 8 -bw 15 -goal 2 -e 35000 -sim 100 -v

# Modify environment dynamics
./add_q_network -n 8 -bw 15 -goal 2 -e 35000 -pf 0.2 -pr 0.3 -sim 100 -v
```

## Implementation Details

The code efficiently handles the complex network resource allocation problem:

- **Binary Encoding**: Each session state requires 2 bits (4 possible states)
- **Dynamic Bandwidth Requirements**: Different sessions can have different bandwidth needs
- **Probabilistic Transitions**: Models natural session behavior with finish and request probabilities
- **Goal State Representation**: Uses "AtLeastK" BDD construction for minimum transferring sessions
- **Policy Sampling**: Uses sampling for policy extraction in large state spaces

## Complex Components Explained

### Session State Transitions

Sessions follow a specific lifecycle:
1. IDLE → REQUESTING (with probability P_NEW_REQUEST)
2. REQUESTING → TRANSFERRING (if bandwidth available) or BLOCKED (if not)
3. TRANSFERRING → IDLE (with probability P_FINISH)
4. BLOCKED → TRANSFERRING (only via agent UNBLOCK action, if bandwidth available)

### Goal BDD Construction

The goal BDD represents states with at least MIN_TRANSFERRING_FOR_GOAL sessions in the TRANSFERRING state. This is constructed using a dynamic programming approach similar to a binary adder circuit, tracking the count of transferring sessions.

### Bandwidth Constraint Handling

When resolving requests or unblocking sessions, the code checks if there's sufficient bandwidth:
1. Calculate current bandwidth usage
2. Check if adding the session's bandwidth requirement exceeds the maximum
3. Allow transition to TRANSFERRING only if constraints are satisfied

## Metrics and Evaluation

The implementation collects several metrics:

- **Average Q-Values**: Shows convergence of learned values
- **Decision Diagram Sizes**: Measures memory efficiency
- **Bellman Error**: Indicates learning progress
- **Terminal State Visits**: Tracks how often the goal state is reached
- **Episode Times**: Measures computational efficiency
- **Memory Usage**: Estimates ADD memory requirements
- **Simulation Goal Hold Rate**: Percentage of time spent in goal states during simulation

## Conclusion

The Network Resource Balancing implementation demonstrates how ADD-Q can handle complex resource allocation problems with:
1. Large state spaces (4^N possible states)
2. Stochastic transitions
3. Resource constraints
4. Binary goal conditions (minimum transferring sessions)

It showcases the power of symbolic reinforcement learning for optimization problems that would be intractable with tabular methods, providing insight into efficient management of limited resources in a dynamic environment.
