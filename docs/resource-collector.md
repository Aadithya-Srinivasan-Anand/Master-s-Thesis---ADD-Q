# Resource Collector ADD-Q Implementation

## Overview

The Resource Collector implementation demonstrates the ADD-Q algorithm in a grid world environment with an additional challenge: the agent must navigate to collect multiple resources scattered throughout the grid. This showcases how ADD-Q handles composite state spaces that combine positional information with resource collection status.

## Environment Description

1. The environment is an NxM grid with randomly placed obstacles.
2. Multiple resources are placed at random locations on the grid.
3. The agent can move in four directions: UP, RIGHT, DOWN, and LEFT.
4. If the agent moves onto a resource location, the resource is collected.
5. The agent receives a reward for each collected resource, a small negative cost for each move, and a larger negative reward for hitting obstacles.
6. The episode terminates when all resources have been collected.

## State Space

The state space combines the agent's position with the status of each resource:
- The agent's position: (x,y) coordinates on the grid
- For each resource: a binary status (0 = collected, 1 = still present)

For an NxM grid with R resources, the state space size is NxM×2^R. For example, with a 5×5 grid and 3 resources, there are 5×5×2^3 = 200 possible states.

## Task Explanation

Imagine you have a robot in a room with several valuable items scattered around. The robot's job is to collect all these items while navigating around furniture and obstacles. The challenge is finding the most efficient path to gather all items in the shortest time.

This problem is more complex than simple navigation because the robot needs to remember which items it has already collected. Its optimal strategy might change dramatically depending on which items remain - for example, taking a long detour to get one item might make sense early, but would be wasteful if it's the last item to collect.

The ADD-Q algorithm helps by learning the best action from any position with any combination of collected/uncollected items. It cleverly identifies patterns in this complex space. For instance, it might recognize that "when resources A and B are collected but C isn't, all positions in this corridor should move toward C" and store that efficiently.

As the robot explores the environment, it builds an optimal collection strategy that works from any starting position and any collection state, guiding it to complete its mission efficiently.

## Key Components

### Composite State Representation

The code uses a binary encoding that combines location and resource status:

- **getLocVarIndex**: Maps from coordinate type and bit to BDD variable index
- **getResourceVarIndex**: Maps from resource index to BDD variable index
- **getCoordValue/setCoordValue**: Handle location components of state
- **getResourceStatus/setResourceStatus**: Handle resource components of state
- **createStateVector**: Creates a complete state from position and resource status

### Environment Interaction

- **initializeEnvironment**: Randomly places obstacles and resources
- **isObstacle**: Checks if a location has an obstacle
- **applyAgentAction**: Applies agent actions, handles resource collection
- **isTerminalState**: Checks if all resources have been collected

### Symbolic Operations

- **createTerminalBDD**: Creates a BDD for terminal states (all resources collected)
- **createStateBDD**: Creates a BDD for a specific composite state
- **evaluateADD**: Evaluates an ADD for a specific state

### Learning Algorithm

- **symbolicQLearning**: Main function implementing the ADD-Q algorithm
- **calculateAverageQValue**: Computes average Q-values across states
- **calculateAverageDAGSize**: Measures decision diagram sizes
- **calculateBellmanError**: Computes Bellman error
- **countCollectedResources**: Counts resources collected in a state

### Visualization and Testing

- **visualizeGrid**: Displays grid with agent, obstacles, resources, and policy
- **printPolicy**: Shows the learned policy in a readable format
- **runSimulation**: Tests the policy through simulations
- **saveMetricsToCSV**: Exports metrics to CSV files

## Usage Examples

```bash
# Basic run with 5x5 grid, 3 resources, 10000 episodes
./add_q_resource_collector -gx 5 -gy 5 -res 3 -e 10000 -v

# With metrics collection
./add_q_resource_collector -gx 5 -gy 5 -res 3 -e 10000 -metrics -v

# Run with simulation to test policy
./add_q_resource_collector -gx 5 -gy 5 -res 3 -e 10000 -sim 100 -v

# Larger environment with more resources
./add_q_resource_collector -gx 8 -gy 8 -res 5 -e 20000 -metrics -sim 100 -v
```

## Implementation Details

The code efficiently handles the composite state space:

- **Binary Encoding**: Position uses ⌈log₂(max(N,M))⌉ bits per coordinate, each resource uses 1 bit
- **Terminal State Recognition**: Recognizes all-resources-collected quickly
- **Resource Collection**: Automatically updates resource status when agent reaches resource locations
- **Efficient State Generation**: Creates valid initial states without obstacles at resource locations

## Interesting Components

### Composite Terminal BDD

The terminal BDD represents all states where every resource bit is 0 (collected). This is created by AND-ing together the negation of each resource variable, regardless of agent position.

### Resource Collection Logic

When the agent moves to a resource location, the code:
1. Checks if the resource is still present (status bit is 1)
2. If present, awards the collection reward and updates the status bit to 0
3. Checks if this was the last resource (terminal state)

### Visualization with Resource Status

The visualization shows:
- Agent position (A)
- Obstacles (#)
- Uncollected resources ($)
- Collected resource locations (o)
- Policy directions (^, >, v, <) based on current resource status

## Metrics and Evaluation

The implementation collects several metrics:

- **Average Q-Values**: Shows convergence of the learned value function
- **Decision Diagram Sizes**: Measures memory efficiency
- **Bellman Error**: Indicates learning progress
- **Terminal State Visits**: Tracks episodes completed by collecting all resources
- **Resources Collected Per Episode**: Average number of resources collected
- **Episode Times**: Measures computational efficiency
- **Simulation Success Rate**: Percentage of simulations where all resources are collected

## Conclusion

The Resource Collector implementation demonstrates how ADD-Q can handle composite state spaces that combine positional information with collection status. It shows that:

1. ADD-Q efficiently represents value functions in product state spaces
2. The symbolic approach can handle dependencies between position and resource status
3. Optimal policies can be learned that account for which resources remain to be collected
4. The algorithm provides a complete solution to multi-objective collection problems

This implementation is particularly relevant for applications like robot navigation with multiple objectives, delivery route planning, and search-and-rescue scenarios where targets must be located and retrieved in an optimal sequence.
