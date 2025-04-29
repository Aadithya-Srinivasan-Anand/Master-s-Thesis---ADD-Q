# Grid World Navigation ADD-Q Implementation

## Overview

The Grid World navigation implementation demonstrates the ADD-Q algorithm in a classic reinforcement learning environment. This is a deterministic navigation problem where an agent must find an optimal path from any position to a goal location while avoiding obstacles.

## Environment Description

1. The environment is an NxM grid with obstacles placed at random positions.
2. The agent can move in four directions: UP, RIGHT, DOWN, and LEFT.
3. If the agent attempts to move into an obstacle or wall, it remains in its current position.
4. The agent receives a large positive reward when reaching the goal state, a small negative cost for each move, and a larger negative reward for hitting obstacles.
5. The episode terminates when the agent reaches the goal state.

## State Space

The state space consists of all possible agent positions in the grid. For an NxM grid, there are NxM possible states (minus obstacle locations). The state is represented as the (x,y) coordinates of the agent.

## Task Explanation

Imagine a robot navigating a room with obstacles, trying to find the fastest path to a destination. The robot can move up, down, left, or right on each step, but it needs to avoid obstacles like furniture or walls. 

The challenge is teaching the robot the best way to navigate from any starting point. The ADD-Q algorithm helps by learning the "value" of being in any position and which direction is best to move from there.

Instead of storing this information separately for every single position (which could be thousands or millions of positions in a complex environment), the algorithm identifies patterns and common structures. For example, it might recognize that "all positions with a clear path to the goal in one direction have similar values" and store that efficiently.

As the robot explores the environment, it learns which movements lead to the goal most efficiently. Eventually, it builds an optimal navigation policy that works from any starting position, guiding it to the goal while avoiding obstacles.

## Key Components

### State Representation

The code uses a binary encoding for the agent's (x,y) position, with log₂ bits per coordinate:

- **getVarIndex**: Maps from coordinate type and bit index to BDD variable index
- **getCoordValue**: Extracts coordinate values from state vectors
- **setCoordValue**: Sets coordinate values in state vectors
- **coordsToState**: Converts (x,y) coordinates to state vectors

### Environment Interaction

- **isObstacle**: Checks if a location contains an obstacle
- **applyAgentAction**: Applies agent actions and returns the resulting state
- **initializeObstacles**: Initializes obstacle locations randomly

### Symbolic Operations

The implementation uses the CUDD library for BDD and ADD operations:

- **createCoordValueBDD**: Creates a BDD pattern for a specific coordinate value
- **createGoalBDD**: Creates a BDD representing the goal state
- **createObstacleBDD**: Creates a BDD for all obstacle locations
- **evaluateADD**: Evaluates an ADD for a specific state
- **StateCache**: Class for caching state BDDs to improve performance

### Learning Algorithm

The Q-learning algorithm is implemented with symbolic operations:

- **symbolicQLearning**: The main function implementing the ADD-Q algorithm
- **generateRandomState**: Generates random valid states for exploration
- **calculateAverageQValue**: Calculates average Q-values across states
- **calculateAverageDAGSize**: Measures ADD node counts
- **calculateBellmanError**: Computes Bellman error to evaluate learning

### Visualization and Testing

- **visualizeGrid**: Provides a text-based visualization of the grid and policy
- **printPolicy**: Displays the learned policy in a readable format
- **runSimulation**: Tests the learned policy through simulations

## Usage Examples

```bash
# Basic run with 8x8 grid, 10 obstacles, 5000 episodes
./ADD-Q_grid_world -gx 8 -gy 8 -obs 10 -e 5000 -v

# Run with metrics collection
./ADD-Q_grid_world -gx 8 -gy 8 -obs 10 -e 5000 -metrics -v

# Run with simulation to test the learned policy
./ADD-Q_grid_world -gx 8 -gy 8 -obs 10 -e 5000 -sim 100 -v

# Larger grid world
./ADD-Q_grid_world -gx 12 -gy 12 -obs 20 -e 10000 -metrics -sim 100 -v
```

## Implementation Details

The code efficiently handles grid world navigation:

- **Binary Encoding**: Positions are encoded using ⌈log₂(max(N,M))⌉ bits per coordinate
- **Epsilon-Greedy Exploration**: Balances exploration and exploitation
- **Automatic Reordering**: Uses CUDD's dynamic variable reordering for efficiency
- **Symbolic Updates**: Q-function updates are performed symbolically
- **Policy Extraction**: Extracts deterministic policy from the learned Q-functions

## Metrics and Evaluation

The implementation collects several metrics:

- **Average Q-Values**: Shows convergence of value function
- **Decision Diagram Sizes**: Measures memory usage efficiency
- **Bellman Error**: Indicates learning progress
- **Goal Visits per Episode**: Tracks how often the goal is reached during learning
- **Episode Times**: Measures computational efficiency
- **Path Lengths**: Records steps needed to reach the goal
- **Simulation Success Rate**: Percentage of simulations where the agent reaches the goal

## Visualization

The implementation includes a text-based visualization of the grid world:

```
  +---+---+---+---+---+
  |.^.|.^.|.^.|.^.|G*G|
  +---+---+---+---+---+
  |.<.|###|.^.|.>.|.^.|
  +---+---+---+---+---+
  |.<.|.<.|.<.|.>.|.^.|
  +---+---+---+---+---+
  |.<.|.<.|.<.|.>.|.^.|
  +---+---+---+---+---+
  |.<.|.<.|.<.|.>.|.^.|
  +---+---+---+---+---+
```

Legend:
- `.^.`, `.>.`, `.v.`, `.<.` - Empty cells with UP, RIGHT, DOWN, and LEFT policy
- `G*G` - Goal state
- `###` - Obstacle

## Conclusion

The Grid World implementation demonstrates the effectiveness of ADD-Q in deterministic environments with obstacles. It shows how symbolic techniques can efficiently represent and solve navigation problems, providing optimal policies for reaching the goal from any valid starting position.
