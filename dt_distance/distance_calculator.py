import numpy as np
from scipy.optimize import linprog
from .tree_parser import TreeConverter
from .data_processor import DataProcessor  # Used if X and y are provided
from .path_extractor import compute_path_weight, compute_paths_distance
from .problem_utils import ProblemParams  # Ensure you have this class defined appropriately

class DistanceCalculator:
    """
    Enhanced to automatically handle data, tree processing, and compute distances between
    decision trees with options for manual problem parameter specifications.
    """
    def __init__(self, tree1, tree2, problem_params=None, X=None, y=None, dataset_name=None,
                 max_depth=None, normalize_distance=True, outcome_weight_in_path=1,
                 print_solver_output=False):
        """
        Initializes with sklearn trees and either a ProblemParams object or data/dataset name,
        including options for normalization and outcome weighting.
        
        :param tree1: First sklearn decision tree model.
        :param tree2: Second sklearn decision tree model.
        :param problem_params: A ProblemParams object specifying feature information (optional).
        :param X: Feature data array (optional, used if problem_params is not provided).
        :param y: Target data array (optional, used if problem_params is not provided).
        :param dataset_name: Name of a predefined dataset to automatically load (optional).
        :param max_depth: Maximum depth of the trees for normalization.
        :param normalize_distance: Whether to normalize the distance calculation.
        :param outcome_weight_in_path: Weight of the outcome in the path distance calculation.
        :param print_solver_output: Flag to print the LP solver's output.
        """

        # Use provided problem_params or process data to obtain them
        if problem_params:
            self.problem_params = problem_params
        else:
            data_processor = DataProcessor(data=X, target=y, dataset_name=dataset_name)
            self.problem_params = data_processor.get_problem_params()

        # Automatically parse trees
        self.paths_tree_1 = TreeConverter(tree1, self.problem_params).get_paths()
        self.paths_tree_2 = TreeConverter(tree2, self.problem_params).get_paths()

        # Distance calculation parameters
        if max_depth == None:
            max_depth = max(tree1.get_depth(), tree2.get_depth())  # Determine max_depth based on the trees
        self.max_depth = max_depth
        self.normalize_distance = normalize_distance
        self.outcome_weight_in_path = outcome_weight_in_path
        self.print_solver_output = print_solver_output

        # Matching outcome parameters
        self.matching_done = False
        self.matching = None
        self.distance = np.nan

    def _compute_paths_weights(self):
        """
        Compute path weights for both trees based on feature bounds and categories.
        """
        self._w1 = {i: compute_path_weight(self.paths_tree_1[i], self.problem_params) for i in range(len(self.paths_tree_1))}
        self._w2 = {i: compute_path_weight(self.paths_tree_2[i], self.problem_params) for i in range(len(self.paths_tree_2))}

    def _compute_paths_distances(self):
        """
        Compute all pairwise path distances for input trees, including outcome weighting.
        """
        self._D = {}
        for i, path_i in enumerate(self.paths_tree_1):
            for j, path_j in enumerate(self.paths_tree_2):
                self._D[(i, j)] = compute_paths_distance(path_i, path_j, self.problem_params, self.outcome_weight_in_path)

    def _append_dummy_paths(self):
        """
        Append dummy paths to ensure both trees have the same number of paths for the optimization problem.
        The distance between a real path and a dummy path is set to the weight of the real path.
        """
        self.n_paths_1 = len(self.paths_tree_1)
        self.n_paths_2 = len(self.paths_tree_2)
        self.n_paths = max(self.n_paths_1, self.n_paths_2)
        # Update distances for dummy paths
        for i in range(self.n_paths):
            for j in range(self.n_paths):
                if i >= self.n_paths_1:  # Dummy paths in tree 1
                    self._D[(i, j)] = self._w2[j]  # Use the weight of the actual path from tree 2
                elif j >= self.n_paths_2:  # Dummy paths in tree 2
                    self._D[(i, j)] = self._w1[i]  # Use the weight of the actual path from tree 1

    def _generate_lp_constraints(self, num_paths_1, num_paths_2):
        """
        Generates linear programming constraints for path matching.
        """
        A_eq = []
        b_eq = [1] * (num_paths_1 + num_paths_2)
        bounds = [(0, 1) for _ in range(num_paths_1 * num_paths_2)]

        for i in range(num_paths_1):
            A_eq.append([1 if j // num_paths_2 == i else 0 for j in range(num_paths_1 * num_paths_2)])
        for j in range(num_paths_2):
            A_eq.append([1 if i % num_paths_2 == j else 0 for i in range(num_paths_1 * num_paths_2)])

        return A_eq, b_eq, bounds
    
    def _path_matching(self):
        # Create the cost vector for the LP problem
        c = [self._D.get((i, j), float('inf')) for i in range(self.n_paths) for j in range(self.n_paths)]
        
        # Constraints ensuring each path is matched exactly once
        A_eq = []
        b_eq = []
        bounds = []
        
        # Equality constraints for matching
        for i in range(self.n_paths):
            row = [1 if j // self.n_paths == i else 0 for j in range(self.n_paths**2)]
            A_eq.append(row)
            b_eq.append(1)
        for j in range(self.n_paths):
            col = [1 if i % self.n_paths == j else 0 for i in range(self.n_paths**2)]
            A_eq.append(col)
            b_eq.append(1)
        
        bounds = [(0, 1) for _ in range(self.n_paths**2)]
        
        # Solve the LP
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs', options={'disp': self.print_solver_output})
        
        if not result.success:
            raise ValueError("LP didn't solve successfully.")
        
        self._soln = result.x


    # def _path_matching(self):
    #     """
    #     Solves the path matching problem using linear programming with outcome weighting.
    #     """
    #     num_paths_1 = len(self.paths_tree_1)
    #     num_paths_2 = len(self.paths_tree_2)
    #     c = [self._D[(i, j)] for i in range(num_paths_1) for j in range(num_paths_2)]
    #     A_eq, b_eq, bounds = self._generate_lp_constraints(num_paths_1, num_paths_2)

    #     result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs', options={'disp': self.print_solver_output})
    #     if not result.success:
    #         raise ValueError("LP didn't solve successfully.")
    #     self._soln = result.x  # Store the solution vector for decoding

    def _decode_soln(self):
        """
        Decodes the solution from the linear programming output, identifying matched paths
        and calculating the total distance based on the matching.
        """
        self.matching = []
        self.distance = 0  # Reset distance to 0 before calculating
        num_paths_2 = len(self.paths_tree_2)
        for i, x in enumerate(self._soln):
            if x > 0.5:  # Assuming a threshold for binary decision
                path_1_index = i // num_paths_2
                path_2_index = i % num_paths_2
                matched_distance = self._D[(path_1_index, path_2_index)]
                self.matching.append((path_1_index, path_2_index))  # Store indices or path objects as needed
                self.distance += matched_distance

        if self.normalize_distance:
            self.distance /= np.power(2, self.max_depth) * (2 * self.max_depth + self.outcome_weight_in_path)

    def compute_tree_distance(self):
        """
        Computes the normalized distance between two decision trees.
        """
        self.matching_done = True
        # Compute path weights and distances between all pairs of paths
        self._compute_paths_weights()
        self._compute_paths_distances()
        # Append dummy paths for tree with fewer paths
        self._append_dummy_paths()
        # Run path matching
        self._path_matching()
        # Decode solution
        self._decode_soln()
        return self.distance
