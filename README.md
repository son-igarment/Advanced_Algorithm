# Advanced Algorithm Project by Phạm Lê Ngọc Sơn

This repository contains the implementation of advanced algorithms for solving complex computational problems, developed by Phạm Lê Ngọc Sơn.

## Project Structure

The project consists of two main algorithm implementations:

1. `Unicode Algorithm.py` - A text processing and address standardization algorithm that uses various string matching techniques to standardize Vietnamese addresses.
2. `DDACS_Algorithm.py` - Dynamic Distributed Ant Colony System (DDACS) algorithm for solving resource-constrained project scheduling problems.

## Unicode Algorithm

The Unicode Algorithm is designed to standardize Vietnamese addresses by:

- Normalizing text through unicode normalization
- Correcting common Vietnamese character errors
- Using trie data structure for efficient text matching
- Implementing KMP (Knuth-Morris-Pratt) algorithm for pattern searching
- Using edit distance with Vietnamese character substitution matrix for fuzzy matching

### Features:
- Address parsing and normalization
- Correction of diacritical marks in Vietnamese text
- Matching addresses against standard databases
- Hierarchical structure recognition (province, district, ward)

### Usage:
```python
from Solution import Solution

# Initialize the solution
solution = Solution()
solution.init_process()

# Process an address
result = solution.process("tp. hồ chí minh, q.1, p. bến nghé")
print(result)
```

## DDACS Algorithm

The Dynamic Distributed Ant Colony System (DDACS) algorithm is implemented for solving resource-constrained project scheduling problems.

### Features:
- Ant Colony Optimization technique
- Dynamic scheduling with resource constraints
- Global and local pheromone update rules
- Heuristic-based decision making

### Usage:
```python
from DDACS_Algorithm import DDACS

# Define problem parameters (activities, resources, etc.)
N = 4  # 4 activities + 2 dummy
T = 20  # Maximum time
ant = 10  # Number of ants
alpha, beta = 1, 2
rho, delta = 0.1, 0.1
q0 = 0.9
q1 = 0.1
c = 2
c1 = 10
max_iter = 50
p = [0, 2, 3, 1, 2, 0]  # Processing time (0 and 5 are dummy)
R = [4]  # 1 resource type, limit 4 units
r = [[0], [2], [3], [1], [2], [0]]  # Resource requirements
predecessors = [[], [0], [0], [1], [2], [3, 4]]  # Dependency relationships
successors = [[1, 2], [3], [4], [5], [5], []]

# Initialize and run the algorithm
ddacs = DDACS(N, T, c, c1, ant, alpha, beta, rho, delta, q0, q1, max_iter, p, R, r, predecessors, successors)
best_solution, best_makespan = ddacs.run()
print(f"Best solution: {best_solution}")
print(f"Best makespan: {best_makespan}")
```

## Dataset Files

The Unicode Algorithm uses several data files:
- `list_province_standard.txt` - Standard provincial data
- `list_district_standard.txt` - Standard district data
- `list_ward_standard.txt` - Standard ward data
- `province_abbreviations.json` - Provincial abbreviations for matching

## Requirements

- Python 3.6+
- NumPy
- editdistance (for Unicode Algorithm)

## Author

**Phạm Lê Ngọc Sơn**

This project was developed by Phạm Lê Ngọc Sơn as part of research into advanced algorithmic approaches to solving complex computational problems in both natural language processing and optimization domains.