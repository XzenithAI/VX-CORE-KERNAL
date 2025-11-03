"""
TRUE Causal Discovery Engine
Not correlation. Not association. ACTUAL causality through intervention and discovery.
Implements PC algorithm, conditional independence testing, and intervention-based causal inference.
"""

import numpy as np
from typing import Dict, Any, List, Set, Tuple, Optional, Callable
from itertools import combinations, permutations
from collections import defaultdict
import time

class CausalGraph:
    """Directed graph representing causal relationships"""

    def __init__(self):
        # Adjacency list: parent -> children
        self.edges: Dict[str, Set[str]] = defaultdict(set)
        # Edge strengths
        self.edge_weights: Dict[Tuple[str, str], float] = {}
        # Intervention history
        self.interventions: List[Dict[str, Any]] = []
        # Nodes
        self.nodes: Set[str] = set()

    def add_edge(self, cause: str, effect: str, weight: float = 1.0):
        """Add directed edge from cause to effect"""
        self.nodes.add(cause)
        self.nodes.add(effect)
        self.edges[cause].add(effect)
        self.edge_weights[(cause, effect)] = weight

    def remove_edge(self, cause: str, effect: str):
        """Remove edge"""
        if effect in self.edges[cause]:
            self.edges[cause].remove(effect)
        if (cause, effect) in self.edge_weights:
            del self.edge_weights[(cause, effect)]

    def get_parents(self, node: str) -> Set[str]:
        """Get direct causes of a node"""
        parents = set()
        for parent, children in self.edges.items():
            if node in children:
                parents.add(parent)
        return parents

    def get_children(self, node: str) -> Set[str]:
        """Get direct effects of a node"""
        return self.edges.get(node, set())

    def get_ancestors(self, node: str) -> Set[str]:
        """Get all ancestors (transitive causes)"""
        ancestors = set()
        to_visit = list(self.get_parents(node))

        while to_visit:
            current = to_visit.pop()
            if current not in ancestors:
                ancestors.add(current)
                to_visit.extend(self.get_parents(current))

        return ancestors

    def get_descendants(self, node: str) -> Set[str]:
        """Get all descendants (transitive effects)"""
        descendants = set()
        to_visit = list(self.get_children(node))

        while to_visit:
            current = to_visit.pop()
            if current not in descendants:
                descendants.add(current)
                to_visit.extend(self.get_children(current))

        return descendants

    def is_d_separated(self, x: str, y: str, z: Set[str]) -> bool:
        """Check if X and Y are d-separated given Z"""
        # Simplified d-separation check
        # In practice, this should implement full d-separation algorithm

        # If Z contains all paths from X to Y, they're d-separated
        paths = self._find_paths(x, y, z)
        return len(paths) == 0

    def _find_paths(self, start: str, end: str, blocked: Set[str], visited: Set[str] = None) -> List[List[str]]:
        """Find all unblocked paths from start to end"""
        if visited is None:
            visited = set()

        if start == end:
            return [[start]]

        if start in visited or start in blocked:
            return []

        visited.add(start)
        paths = []

        # Forward edges
        for child in self.get_children(start):
            for path in self._find_paths(child, end, blocked, visited.copy()):
                paths.append([start] + path)

        # Backward edges (for undirected parts)
        for parent in self.get_parents(start):
            for path in self._find_paths(parent, end, blocked, visited.copy()):
                paths.append([start] + path)

        return paths


class ConditionalIndependenceTest:
    """Test for conditional independence using various methods"""

    @staticmethod
    def pearson_correlation(x: np.ndarray, y: np.ndarray, z: Optional[np.ndarray] = None,
                           threshold: float = 0.1) -> bool:
        """Test independence using partial correlation"""

        if z is None:
            # Simple correlation
            corr = np.corrcoef(x, y)[0, 1]
            return abs(corr) < threshold

        # Partial correlation
        # Regress X on Z
        if len(z.shape) == 1:
            z = z.reshape(-1, 1)

        try:
            # X | Z
            beta_xz = np.linalg.lstsq(z, x, rcond=None)[0]
            residual_x = x - z @ beta_xz

            # Y | Z
            beta_yz = np.linalg.lstsq(z, y, rcond=None)[0]
            residual_y = y - z @ beta_yz

            # Correlation of residuals
            partial_corr = np.corrcoef(residual_x, residual_y)[0, 1]

            return abs(partial_corr) < threshold

        except:
            return False

    @staticmethod
    def mutual_information(x: np.ndarray, y: np.ndarray, z: Optional[np.ndarray] = None,
                          threshold: float = 0.1) -> bool:
        """Test independence using mutual information"""

        def entropy(data):
            _, counts = np.unique(data, return_counts=True)
            probs = counts / len(data)
            return -np.sum(probs * np.log2(probs + 1e-10))

        def mutual_info(x, y):
            return entropy(x) + entropy(y) - entropy(np.column_stack([x, y]))

        if z is None:
            mi = mutual_info(x, y)
            return mi < threshold

        # Conditional mutual information
        cmi = mutual_info(x, y) - mutual_info(x, z) - mutual_info(y, z) + entropy(z)
        return cmi < threshold


class PCAlgorithm:
    """Peter-Clark algorithm for causal structure learning"""

    def __init__(self, independence_test: str = 'pearson', significance: float = 0.05):
        self.independence_test = independence_test
        self.significance = significance
        self.graph = CausalGraph()

    def learn_structure(self, data: Dict[str, np.ndarray]) -> CausalGraph:
        """Learn causal structure from observational data"""

        variables = list(data.keys())
        n_vars = len(variables)

        # Phase 1: Start with complete undirected graph
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                self.graph.add_edge(variables[i], variables[j], 0.5)
                self.graph.add_edge(variables[j], variables[i], 0.5)

        # Phase 2: Remove edges based on conditional independence
        for order in range(n_vars - 1):
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    var_i = variables[i]
                    var_j = variables[j]

                    if var_j not in self.graph.get_children(var_i):
                        continue

                    # Get neighbors
                    neighbors = (self.graph.get_children(var_i) | self.graph.get_parents(var_i)) - {var_j}

                    if len(neighbors) < order:
                        continue

                    # Test all subsets of neighbors of size 'order'
                    for subset in combinations(neighbors, order):
                        # Test if X and Y are independent given subset
                        if self._test_independence(data[var_i], data[var_j],
                                                   [data[v] for v in subset]):
                            # Remove edge
                            self.graph.remove_edge(var_i, var_j)
                            self.graph.remove_edge(var_j, var_i)
                            break

        # Phase 3: Orient edges (simplified)
        self._orient_edges(variables)

        return self.graph

    def _test_independence(self, x: np.ndarray, y: np.ndarray, z_list: List[np.ndarray]) -> bool:
        """Test conditional independence"""

        z = np.column_stack(z_list) if z_list else None

        if self.independence_test == 'pearson':
            return ConditionalIndependenceTest.pearson_correlation(x, y, z, self.significance)
        elif self.independence_test == 'mi':
            return ConditionalIndependenceTest.mutual_information(x, y, z, self.significance)
        else:
            return False

    def _orient_edges(self, variables: List[str]):
        """Orient edges to create DAG (simplified)"""

        # Rule 1: Orient v-structures (X -> Z <- Y where X and Y are not adjacent)
        for z in variables:
            parents_z = self.graph.get_parents(z)
            if len(parents_z) >= 2:
                for x, y in combinations(parents_z, 2):
                    # If X and Y not adjacent, orient as v-structure
                    if y not in self.graph.get_children(x) and x not in self.graph.get_children(y):
                        # Keep X -> Z and Y -> Z, remove Z -> X and Z -> Y
                        self.graph.remove_edge(z, x)
                        self.graph.remove_edge(z, y)

        # Rule 2: Orient remaining edges to avoid cycles (simplified)
        for var in variables:
            children = list(self.graph.get_children(var))
            for child in children:
                if var in self.graph.get_children(child):
                    # Bidirectional edge - remove one direction
                    self.graph.remove_edge(child, var)


class InterventionEngine:
    """Perform interventions to discover causal relationships"""

    def __init__(self, system_executor: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """
        system_executor: function that takes interventions and returns observed outcomes
        """
        self.executor = system_executor
        self.intervention_history = []

    def do_intervention(self, variable: str, value: Any) -> Dict[str, Any]:
        """Perform do(X=x) intervention"""

        intervention = {
            'variable': variable,
            'value': value,
            'timestamp': time.time()
        }

        # Execute intervention
        outcomes = self.executor({variable: value})

        intervention['outcomes'] = outcomes
        self.intervention_history.append(intervention)

        return outcomes

    def estimate_causal_effect(self, cause: str, effect: str,
                               cause_values: List[Any], n_samples: int = 100) -> float:
        """Estimate causal effect of cause on effect using interventions"""

        effects = []

        for value in cause_values:
            # Perform intervention multiple times
            value_effects = []
            for _ in range(n_samples):
                outcome = self.do_intervention(cause, value)
                if effect in outcome:
                    value_effects.append(outcome[effect])

            if value_effects:
                effects.append(np.mean(value_effects))

        if len(effects) >= 2:
            # Estimate average causal effect
            return np.std(effects)  # Variance indicates causal strength
        else:
            return 0.0

    def discover_causal_direction(self, var_x: str, var_y: str,
                                  test_values: List[Any]) -> Optional[str]:
        """Determine causal direction between X and Y using interventions"""

        # Test X -> Y
        effects_x_on_y = []
        for value in test_values:
            outcome = self.do_intervention(var_x, value)
            if var_y in outcome:
                effects_x_on_y.append(outcome[var_y])

        # Test Y -> X
        effects_y_on_x = []
        for value in test_values:
            outcome = self.do_intervention(var_y, value)
            if var_x in outcome:
                effects_y_on_x.append(outcome[var_x])

        # Compare effect strengths
        strength_x_to_y = np.std(effects_x_on_y) if effects_x_on_y else 0
        strength_y_to_x = np.std(effects_y_on_x) if effects_y_on_x else 0

        if strength_x_to_y > strength_y_to_x * 1.5:
            return f"{var_x} -> {var_y}"
        elif strength_y_to_x > strength_x_to_y * 1.5:
            return f"{var_y} -> {var_x}"
        else:
            return None  # Bidirectional or no clear direction


class CausalDiscoveryEngine:
    """Complete causal discovery system combining observation and intervention"""

    def __init__(self):
        self.causal_graph = CausalGraph()
        self.pc_algorithm = PCAlgorithm()
        self.intervention_engine = None
        self.observational_data: Dict[str, List[Any]] = defaultdict(list)

    def observe(self, variable: str, value: Any):
        """Record observational data"""
        self.observational_data[variable].append(value)

    def learn_from_observations(self) -> CausalGraph:
        """Learn causal structure from accumulated observations"""

        # Convert to numpy arrays
        data = {var: np.array(values) for var, values in self.observational_data.items()}

        # Run PC algorithm
        self.causal_graph = self.pc_algorithm.learn_structure(data)

        return self.causal_graph

    def set_intervention_executor(self, executor: Callable):
        """Set the system executor for performing interventions"""
        self.intervention_engine = InterventionEngine(executor)

    def validate_causal_edge(self, cause: str, effect: str, test_values: List[Any]) -> bool:
        """Validate a causal edge using interventions"""

        if self.intervention_engine is None:
            return False

        causal_effect = self.intervention_engine.estimate_causal_effect(
            cause, effect, test_values, n_samples=50
        )

        # Significant causal effect indicates true causal relationship
        return causal_effect > 0.1

    def get_causal_explanation(self, effect: str) -> Dict[str, Any]:
        """Generate causal explanation for an effect"""

        direct_causes = self.causal_graph.get_parents(effect)
        all_causes = self.causal_graph.get_ancestors(effect)

        # Estimate strength of each cause
        cause_strengths = {}
        for cause in direct_causes:
            edge_weight = self.causal_graph.edge_weights.get((cause, effect), 0.5)
            cause_strengths[cause] = edge_weight

        return {
            'effect': effect,
            'direct_causes': list(direct_causes),
            'all_causes': list(all_causes),
            'cause_strengths': cause_strengths,
            'explanation': self._generate_explanation(effect, direct_causes, cause_strengths)
        }

    def _generate_explanation(self, effect: str, causes: Set[str],
                            strengths: Dict[str, float]) -> str:
        """Generate natural language causal explanation"""

        if not causes:
            return f"{effect} appears to have no identified causes in the current model."

        # Sort by strength
        sorted_causes = sorted(strengths.items(), key=lambda x: x[1], reverse=True)

        explanation = f"{effect} is caused by:\n"
        for cause, strength in sorted_causes:
            explanation += f"  - {cause} (causal strength: {strength:.2f})\n"

        return explanation

    def discover_mechanisms(self) -> List[Dict[str, Any]]:
        """Discover causal mechanisms (functional relationships)"""

        mechanisms = []

        for effect in self.causal_graph.nodes:
            causes = self.causal_graph.get_parents(effect)

            if causes and effect in self.observational_data:
                # Try to learn functional form
                effect_data = np.array(self.observational_data[effect])
                cause_data = np.column_stack([
                    self.observational_data[cause] for cause in causes
                ])

                # Fit linear model as first approximation
                try:
                    coeffs = np.linalg.lstsq(cause_data, effect_data, rcond=None)[0]

                    mechanisms.append({
                        'effect': effect,
                        'causes': list(causes),
                        'type': 'linear',
                        'coefficients': coeffs.tolist(),
                        'equation': self._format_equation(effect, list(causes), coeffs)
                    })
                except:
                    pass

        return mechanisms

    def _format_equation(self, effect: str, causes: List[str], coeffs: np.ndarray) -> str:
        """Format causal mechanism as equation"""
        terms = [f"{coeff:.2f}*{cause}" for cause, coeff in zip(causes, coeffs)]
        return f"{effect} = {' + '.join(terms)}"
