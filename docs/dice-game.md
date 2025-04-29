# Dice Game ADD-Q Implementation

## Overview

The Dice Game implementation demonstrates the ADD-Q algorithm in a probabilistic environment where the objective is to make all dice show the same face value. This example showcases how ADD-Q handles stochastic transitions and complex state spaces efficiently.

## Game Rules

1. We have a set of N dice, each with F faces (e.g., 3 dice with 6 faces each).
2. On each turn, the agent chooses an action: "Keep all dice showing face value k" where k is a value from 1 to F.
3. Dice that don't match the chosen value are rerolled.
4. The game terminates when all dice show the same face value.
5. The agent receives a positive reward when reaching a terminal state and a small negative cost for each action.

## State Space

The state space consists of all possible combinations of dice values. For N dice with F faces each, the state space size is F^N. For example, with 3 standard dice (6 faces), there are 6^3 = 216 possible states.

## Task Explanation

Imagine you have several dice in front of you. Your goal is to get all the dice to show the same number using as few rolls as possible. On each turn, you decide which number you want to keep, and you reroll all dice that don't show that number. The game ends when all dice show the same number.

This might sound simple, but determining the optimal strategy can be complex. Should you keep the most frequent number? Or perhaps a different number has a better chance of getting all dice to match in fewer steps?

The ADD-Q algorithm learns the optimal strategy by essentially figuring out:
1. Which number to keep in any given situation
2. How valuable each possible dice arrangement is toward the goal

The clever part is that it doesn't need to store values for every possible dice arrangement separately. Instead, it identifies patterns and stores information efficiently using decision diagrams.

## Key Components

### State Representation

The code uses a binary encoding for each die, where the number of bits per die depends on the number of faces. For standard 6-sided dice, 3 bits per die are used.

### Symbolic Operations

The implementation leverages the CUDD library to create and manipulate Binary Decision Diagrams (BDDs) and Algebraic Decision Diagrams (ADDs). Key functions include:

- **calculateVarsPerDie**: Determines how many binary variables are needed to represent each die
- **createFaceValueBDD**: Creates a BDD representing a specific die showing a specific face value
- **createStateBDD**: Creates a BDD for a complete dice state
- **createTerminalBDD**: Creates a BDD representing all terminal states (all dice showing the same value)
- **evaluateADD**: Evaluates an ADD for a specific dice state

### Environment Dynamics

The code simulates the probabilistic outcomes of actions:

- **generateTransitions**: Given a state and action, generates all possible next states and their probabilities
- **isTerminalState**: Checks if a state is terminal (all dice showing the same value)
- **getAvailableActions**: Determines valid actions for a given state

### Learning Algorithm

The symbolic Q-learning algorithm is implemented with these key steps:

- **symbolicQLearning**: The main function implementing the ADD-Q algorithm
- **applyAgentAction**: Simulates agent actions in the environment
- **runSimulation**: Tests the learned policy by simulating games

### Metrics Collection

The implementation collects various metrics to evaluate performance:

- **calculateAverageQValue**: Computes average Q-values across sampled states
- **calculateAverageDAGSize**: Measures the size of the decision diagrams
- **calculateBellmanError**: Calculates the Bellman error to evaluate learning progress
- **saveMetricsToCSV**: Exports collected metrics to CSV files for analysis

## Usage Examples

```bash
# Basic run with 3 dice, 6 faces, 20,000 episodes
./ADD-Q_dice_game -d 3 -f 6 -e 20000 -v

# Run with metrics collection, sampling 1000 states for calculations
./ADD-Q_dice_game -d 3 -f 6 -e 20000 -metrics 1000 -v

# Run with simulation to test the learned policy
./ADD-Q_dice_game -d 3 -f 6 -e 20000 -sim 100 -v

# Challenging configuration with 4 dice
./ADD-Q_dice_game -d 4 -f 6 -e 30000 -metrics 1000 -sim 100 -v
```

## Implementation Details

The code efficiently handles the exponential state space of the dice game using symbolic techniques:

- **Binary Encoding**: Each die value is encoded in log₂(F) bits
- **Symbolic Updates**: Q-value updates are performed on entire regions of the state space simultaneously
- **Caching**: State BDDs are cached to improve performance
- **Error Handling**: Robust error checking and recovery mechanisms

## Metrics and Evaluation

The implementation collects several metrics:

- **Average Q-Values**: Shows the convergence of the learned value function
- **Decision Diagram Sizes**: Measures memory efficiency
- **Bellman Error**: Indicates learning progress
- **Simulation Success Rate**: Percentage of simulations where all dice match
- **Steps to Terminal**: Distribution of steps needed to reach the goal

## Conclusion

The Dice Game implementation demonstrates the effectiveness of ADD-Q in handling stochastic environments with large state spaces. The symbolic approach allows efficient representation and updates of the Q-function, making it possible to solve problems that would be intractable with tabular methods.
