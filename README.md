<div align="center">

# Decision Tree Distance Calculator

[![Paper](https://img.shields.io/badge/arXiv-2211.11747-red)]([https://arxiv.org/abs/2311.13695](https://arxiv.org/abs/2305.17299))
[![PyPI package](https://badge.fury.io/py/backbone-learn.svg)]([https://pypi.org/project/backbone-learn/](https://pypi.org/project/dt-distance/))

</div>

### Overview
This Python package calculates the distance between two decision trees, facilitating model comparison and analysis. 
It implements the distance metric outlined in "Improving Stability in Decision Trees Models" by Bertsimas and Digalakis Jr ([arXiv:2305.17299](https://arxiv.org/abs/2305.17299)https://arxiv.org/abs/2305.17299).

## Getting Started

### Installation

Install dt-distance using pip:
```python
pip install dt-distance
```

### Usage
```
from distance_calculator import DistanceCalculator
# Initialize with two trained sklearn decision trees: tree1 and tree2
calculator = DistanceCalculator(tree1, tree2)
# Compute the distance
distance = calculator.compute_tree_distance()
print(f"Distance: {distance}")
```

### Main Parameters
- tree1, tree2: Two sklearn DecisionTreeClassifier objects.
- problem_params (optional): Specifies feature info. Defaults to inferred from trees if not provided.
- X, y (optional): Data matrices if problem_params is not used.
- max_depth (optional): For normalization, defaults to the deeper of the two trees.


## Citation
If using this tool in your research, please cite the associated paper:
```
@misc{bertsimas2023improving,
      title={Improving Stability in Decision Tree Models}, 
      author={Dimitris Bertsimas and Vassilis Digalakis Jr au2},
      year={2023},
      eprint={2305.17299},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```
