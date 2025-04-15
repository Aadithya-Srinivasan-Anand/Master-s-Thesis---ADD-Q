# ADD-Q: Algebraic Decision Diagram for Symbolic Q-Learning in boolean domains

This repository contains implementations of ADD-Q, a symbolic approach to reinforcement learning that uses Algebraic Decision Diagrams for efficient representation of Q-functions in Boolean state spaces.

## What is ADD-Q?

ADD-Q is an uses combines the compact state space representation powers of  Algebraic Decision Diagrams (ADDs) in the implementation of symbolic Q-learning. Traditional tabular Q-learning generally becomes intractable for problems with large state spaces due to problems that arise due to excess dimensionality, ADD-Q overcomes this limitation by exploiting structure and redundancy in the value function.

### Key Advantages:

- **Efficient State Space Representation**: ADDs leverage shared structure in the value function to dramatically reduce memory usage.
- **Symbolic Updates**: Entire regions of the state space can be updated simultaneously.
- **Scalability**: Can handle problems with state spaces that would be impractical with tabular methods.
- **Preserves Optimality**: Maintains the theoretical guarantees of traditional Q-learning.

## How It Works

ADD-Q uses the CUDD (Colorado University Decision Diagram) library to represent and manipulate Binary Decision Diagrams (BDDs) and Algebraic Decision Diagrams (ADDs):

1. **State Encoding**: States are encoded as binary vectors, where each bit represents a feature of the state.
2. **Q-Function Representation**: Instead of a table, Q-functions are represented as ADDs that map state-action pairs to values.
3. **Symbolic Updates**: Q-value updates are performed symbolically using operations on the entire ADD structure.
4. **Compact Storage**: Structurally similar states share representation in memory.

## Implementations

This repository includes four different implementations of ADD-Q for various domains:

1. [**Dice Game**](docs/dice-game.md): A game where the goal is to make all dice show the same value.
2. [**Grid World Navigation**](docs/grid-world.md): A classic grid world problem with obstacles and a goal state.
3. [**Network Resource Balancing**](docs/network-resource.md): A problem of managing network sessions with bandwidth constraints.
4. [**Resource Collection**](docs/resource-collector.md): An agent must navigate a grid to collect resources at different locations.

Each implementation showcases different aspects of ADD-Q and includes visualization tools, metrics collection, and simulation capabilities.

## Getting Started

### Prerequisites

- C++ compiler with C++11 support
- CUDD library (version 3.0 or newer)
- CMake (version 3.10 or newer)


### Running Examples

Each implementation can be run with various parameters:

```bash
# Example: Dice Game
./add_q_dice_game -d 3 -f 6 -e 20000 -metrics 1000 -sim 100 -v

# Example: Grid World
./add_q_grid_world -gx 8 -gy 8 -obs 10 -e 5000 -metrics -sim 100

# Example: Network Resource Balancing
./add_q_network -n 8 -bw 15 -goal 2 -e 35000 -metrics 1000 -sim 100

# Example: Resource Collection
./add_q_resource_collector -gx 5 -gy 5 -res 3 -e 10000 -metrics -sim 100
```

Use the `-h` flag with any executable for a full list of parameters.

## Results and Metrics

Each implementation collects and can output various metrics to CSV files for analysis:

- Average Q-values over time
- Decision diagram sizes (node counts)
- Bellman error
- Success rates in simulations
- Memory usage estimates
- Execution times

These metrics help evaluate the efficiency and effectiveness of the ADD-Q approach compared to traditional methods.

## References

- Bryant, R. E. (1986). Graph-based algorithms for boolean function manipulation. IEEE Transactions on Computers, 100(8), 677-691.
- Bahar, R. I., Frohm, E. A., Gaona, C. M., Hachtel, G. D., Macii, E., Pardo, A., & Somenzi, F. (1997). Algebraic decision diagrams and their applications. Formal methods in system design, 10(2), 171-206.
- Hoey, J., St-Aubin, R., Hu, A., & Boutilier, C. (1999). SPUDD: Stochastic planning using decision diagrams. In Proceedings of the Fifteenth conference on Uncertainty in artificial intelligence (pp. 279-288).


## Acknowledgments

- The CUDD library developers (Prof. Fabio Somenzi)
