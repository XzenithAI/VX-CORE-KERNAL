"""
Meta-Meta-Learning Engine
Not just learning. Not just learning to learn. Learning to learn HOW to learn.
Third-order optimization of learning strategies.
"""

import numpy as np
import time
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque

@dataclass
class LearningStrategy:
    """A specific approach to learning"""
    strategy_id: str
    name: str
    parameters: Dict[str, Any]
    performance_history: List[float] = field(default_factory=list)
    adaptation_history: List[Dict[str, Any]] = field(default_factory=list)
    meta_performance: float = 0.5  # How well does this strategy adapt?

@dataclass
class MetaLearningStrategy:
    """A strategy for selecting and adapting learning strategies"""
    meta_strategy_id: str
    name: str
    selection_policy: Callable
    adaptation_policy: Callable
    performance_history: List[float] = field(default_factory=list)

@dataclass
class LearningTask:
    """A learning task with context"""
    task_id: str
    task_type: str
    difficulty: float
    context: Dict[str, Any]
    optimal_strategy: Optional[str] = None  # Discovered optimal strategy

class MetaMetaLearner:
    """Third-order learning: learning how to learn how to learn"""

    def __init__(self):
        # Level 1: Base learning strategies
        self.base_strategies: Dict[str, LearningStrategy] = {}

        # Level 2: Meta-learning strategies (for selecting Level 1)
        self.meta_strategies: Dict[str, MetaLearningStrategy] = {}

        # Level 3: Meta-meta strategy (for selecting Level 2)
        self.meta_meta_policy = self._initialize_meta_meta_policy()

        # Task history
        self.task_history: List[Tuple[LearningTask, str, float]] = []

        # Performance tracking
        self.adaptation_trace: List[Dict[str, Any]] = []

        # Initialize default strategies
        self._initialize_default_strategies()

    def _initialize_default_strategies(self):
        """Initialize base learning strategies"""

        # Gradient-based learning
        self.register_base_strategy(LearningStrategy(
            strategy_id="gradient",
            name="Gradient Descent Learning",
            parameters={"learning_rate": 0.01, "momentum": 0.9}
        ))

        # Evolutionary learning
        self.register_base_strategy(LearningStrategy(
            strategy_id="evolutionary",
            name="Evolutionary Learning",
            parameters={"population_size": 100, "mutation_rate": 0.1}
        ))

        # Bayesian learning
        self.register_base_strategy(LearningStrategy(
            strategy_id="bayesian",
            name="Bayesian Learning",
            parameters={"prior_strength": 1.0, "update_rate": 0.1}
        ))

        # Reinforcement learning
        self.register_base_strategy(LearningStrategy(
            strategy_id="reinforcement",
            name="Reinforcement Learning",
            parameters={"gamma": 0.99, "epsilon": 0.1}
        ))

        # Meta-learning strategies
        self.register_meta_strategy(MetaLearningStrategy(
            meta_strategy_id="ucb",
            name="Upper Confidence Bound Selection",
            selection_policy=self._ucb_selection,
            adaptation_policy=self._gradient_adaptation
        ))

        self.register_meta_strategy(MetaLearningStrategy(
            meta_strategy_id="thompson",
            name="Thompson Sampling Selection",
            selection_policy=self._thompson_sampling,
            adaptation_policy=self._bayesian_adaptation
        ))

        self.register_meta_strategy(MetaLearningStrategy(
            meta_strategy_id="contextual",
            name="Contextual Bandit Selection",
            selection_policy=self._contextual_selection,
            adaptation_policy=self._contextual_adaptation
        ))

    def _initialize_meta_meta_policy(self) -> Dict[str, Any]:
        """Initialize meta-meta learning policy"""
        return {
            'strategy_embeddings': {},  # Learned representations of strategies
            'context_encoder': np.random.randn(64, 64) * 0.1,  # Context encoding
            'meta_weights': np.ones(len(self.meta_strategies)) if self.meta_strategies else np.array([]),
            'exploration_rate': 0.2,
            'adaptation_rate': 0.05
        }

    def register_base_strategy(self, strategy: LearningStrategy):
        """Register a base learning strategy"""
        self.base_strategies[strategy.strategy_id] = strategy

    def register_meta_strategy(self, meta_strategy: MetaLearningStrategy):
        """Register a meta-learning strategy"""
        self.meta_strategies[meta_strategy.meta_strategy_id] = meta_strategy

    def learn_task(self, task: LearningTask, data: Dict[str, Any]) -> Dict[str, Any]:
        """Learn a task using meta-meta learning"""

        start_time = time.time()

        # Level 3: Select meta-strategy
        selected_meta_strategy_id = self._select_meta_strategy(task)
        meta_strategy = self.meta_strategies[selected_meta_strategy_id]

        # Level 2: Meta-strategy selects base strategy
        selected_base_strategy_id = meta_strategy.selection_policy(task, self.base_strategies)
        base_strategy = self.base_strategies[selected_base_strategy_id]

        # Level 1: Apply base strategy to task
        performance = self._apply_base_strategy(base_strategy, task, data)

        # Record performance
        base_strategy.performance_history.append(performance)
        meta_strategy.performance_history.append(performance)

        # Level 2: Adapt base strategy
        adapted_params = meta_strategy.adaptation_policy(base_strategy, performance, task)
        base_strategy.parameters.update(adapted_params)
        base_strategy.adaptation_history.append({
            'timestamp': time.time(),
            'adapted_params': adapted_params,
            'performance': performance
        })

        # Level 3: Update meta-meta policy
        self._update_meta_meta_policy(task, meta_strategy, base_strategy, performance)

        # Record in history
        self.task_history.append((task, selected_base_strategy_id, performance))

        learning_time = time.time() - start_time

        # Record adaptation trace
        self.adaptation_trace.append({
            'timestamp': time.time(),
            'task_id': task.task_id,
            'meta_strategy': selected_meta_strategy_id,
            'base_strategy': selected_base_strategy_id,
            'performance': performance,
            'learning_time': learning_time
        })

        return {
            'performance': performance,
            'meta_strategy_used': selected_meta_strategy_id,
            'base_strategy_used': selected_base_strategy_id,
            'adapted_parameters': adapted_params,
            'learning_time': learning_time
        }

    def _select_meta_strategy(self, task: LearningTask) -> str:
        """Level 3: Select which meta-strategy to use"""

        if not self.meta_strategies:
            # Fallback if no meta-strategies
            return list(self.meta_strategies.keys())[0] if self.meta_strategies else "ucb"

        # Encode task context
        context_vector = self._encode_task_context(task)

        # Compute scores for each meta-strategy
        scores = {}
        for meta_id, meta_strategy in self.meta_strategies.items():
            # Exploitation: historical performance
            if meta_strategy.performance_history:
                avg_performance = np.mean(meta_strategy.performance_history[-10:])
            else:
                avg_performance = 0.5

            # Exploration: uncertainty
            uncertainty = 1.0 / (1.0 + len(meta_strategy.performance_history))

            # Combined score
            scores[meta_id] = avg_performance + self.meta_meta_policy['exploration_rate'] * uncertainty

        # Select best
        selected = max(scores.items(), key=lambda x: x[1])[0]

        return selected

    def _encode_task_context(self, task: LearningTask) -> np.ndarray:
        """Encode task context into vector"""
        # Simplified encoding
        features = [
            task.difficulty,
            hash(task.task_type) % 100 / 100.0,
            len(task.context),
        ]

        # Pad to 64 dimensions
        features.extend([0.0] * (64 - len(features)))

        return np.array(features[:64])

    def _apply_base_strategy(self, strategy: LearningStrategy,
                             task: LearningTask, data: Dict[str, Any]) -> float:
        """Apply base learning strategy to task"""

        # Simulate learning process
        # In real system, this would actually perform learning

        # Performance depends on strategy match to task
        base_performance = np.random.beta(2, 2)  # 0-1

        # Strategy-specific adjustments
        if strategy.strategy_id == "gradient" and task.task_type == "continuous":
            base_performance += 0.1
        elif strategy.strategy_id == "evolutionary" and task.task_type == "discrete":
            base_performance += 0.1
        elif strategy.strategy_id == "bayesian" and task.difficulty < 0.5:
            base_performance += 0.1

        # Parameter influence
        lr = strategy.parameters.get("learning_rate", 0.01)
        if 0.005 < lr < 0.05:
            base_performance += 0.05

        return min(1.0, base_performance)

    # Selection policies (Level 2)

    def _ucb_selection(self, task: LearningTask,
                      strategies: Dict[str, LearningStrategy]) -> str:
        """Upper Confidence Bound selection"""
        total_trials = sum(len(s.performance_history) for s in strategies.values())

        best_score = -float('inf')
        best_strategy = None

        for strategy_id, strategy in strategies.items():
            if not strategy.performance_history:
                # Try untried strategies first
                return strategy_id

            # UCB formula
            avg_reward = np.mean(strategy.performance_history)
            n_trials = len(strategy.performance_history)
            exploration = np.sqrt(2 * np.log(total_trials + 1) / n_trials)

            score = avg_reward + exploration

            if score > best_score:
                best_score = score
                best_strategy = strategy_id

        return best_strategy or list(strategies.keys())[0]

    def _thompson_sampling(self, task: LearningTask,
                          strategies: Dict[str, LearningStrategy]) -> str:
        """Thompson Sampling selection"""
        samples = {}

        for strategy_id, strategy in strategies.items():
            if not strategy.performance_history:
                # Assume beta(1, 1) prior
                samples[strategy_id] = np.random.beta(1, 1)
            else:
                # Beta posterior
                successes = sum(1 for p in strategy.performance_history if p > 0.5)
                failures = len(strategy.performance_history) - successes
                samples[strategy_id] = np.random.beta(successes + 1, failures + 1)

        return max(samples.items(), key=lambda x: x[1])[0]

    def _contextual_selection(self, task: LearningTask,
                             strategies: Dict[str, LearningStrategy]) -> str:
        """Contextual bandit selection"""

        # Encode task
        context = self._encode_task_context(task)

        best_score = -float('inf')
        best_strategy = None

        for strategy_id, strategy in strategies.items():
            # Context-dependent score
            # In real system, would use learned context-strategy mapping

            base_score = np.mean(strategy.performance_history) if strategy.performance_history else 0.5

            # Task difficulty match
            if task.difficulty > 0.7 and strategy.strategy_id == "evolutionary":
                base_score += 0.1

            if best_score < base_score:
                best_score = base_score
                best_strategy = strategy_id

        return best_strategy or list(strategies.keys())[0]

    # Adaptation policies (Level 2)

    def _gradient_adaptation(self, strategy: LearningStrategy,
                            performance: float, task: LearningTask) -> Dict[str, Any]:
        """Gradient-based parameter adaptation"""

        adapted = {}

        if "learning_rate" in strategy.parameters:
            lr = strategy.parameters["learning_rate"]

            # Adapt based on performance
            if performance > 0.7:
                # Doing well, increase learning rate slightly
                adapted["learning_rate"] = min(0.1, lr * 1.1)
            elif performance < 0.3:
                # Doing poorly, decrease learning rate
                adapted["learning_rate"] = max(0.001, lr * 0.9)

        return adapted

    def _bayesian_adaptation(self, strategy: LearningStrategy,
                            performance: float, task: LearningTask) -> Dict[str, Any]:
        """Bayesian parameter adaptation"""

        adapted = {}

        # Bayesian update of hyperparameters
        if len(strategy.performance_history) > 5:
            recent_performance = np.mean(strategy.performance_history[-5:])

            # Adapt parameters based on posterior
            for param_name, param_value in strategy.parameters.items():
                if isinstance(param_value, (int, float)):
                    # Gaussian perturbation based on uncertainty
                    uncertainty = np.std(strategy.performance_history[-5:])
                    perturbation = np.random.randn() * uncertainty * 0.1

                    if recent_performance > 0.6:
                        # Small perturbation when doing well
                        adapted[param_name] = param_value * (1 + perturbation * 0.1)
                    else:
                        # Larger perturbation when doing poorly
                        adapted[param_name] = param_value * (1 + perturbation * 0.5)

        return adapted

    def _contextual_adaptation(self, strategy: LearningStrategy,
                              performance: float, task: LearningTask) -> Dict[str, Any]:
        """Context-aware parameter adaptation"""

        adapted = {}

        # Adapt based on task context
        if task.difficulty > 0.7:
            # Hard task - use more conservative parameters
            if "learning_rate" in strategy.parameters:
                adapted["learning_rate"] = min(strategy.parameters["learning_rate"], 0.01)

        elif task.difficulty < 0.3:
            # Easy task - can be more aggressive
            if "learning_rate" in strategy.parameters:
                adapted["learning_rate"] = max(strategy.parameters["learning_rate"], 0.05)

        return adapted

    def _update_meta_meta_policy(self, task: LearningTask,
                                 meta_strategy: MetaLearningStrategy,
                                 base_strategy: LearningStrategy,
                                 performance: float):
        """Level 3: Update meta-meta learning policy"""

        # Update strategy embeddings
        task_encoding = self._encode_task_context(task)

        if meta_strategy.meta_strategy_id not in self.meta_meta_policy['strategy_embeddings']:
            self.meta_meta_policy['strategy_embeddings'][meta_strategy.meta_strategy_id] = task_encoding.copy()
        else:
            # Running average
            alpha = self.meta_meta_policy['adaptation_rate']
            current = self.meta_meta_policy['strategy_embeddings'][meta_strategy.meta_strategy_id]
            self.meta_meta_policy['strategy_embeddings'][meta_strategy.meta_strategy_id] = (
                (1 - alpha) * current + alpha * task_encoding
            )

        # Update meta-weights based on performance
        if len(self.meta_strategies) > 0:
            meta_idx = list(self.meta_strategies.keys()).index(meta_strategy.meta_strategy_id)

            if len(self.meta_meta_policy['meta_weights']) == len(self.meta_strategies):
                # Reward successful meta-strategy
                reward = (performance - 0.5) * 2  # Scale to [-1, 1]
                self.meta_meta_policy['meta_weights'][meta_idx] += (
                    self.meta_meta_policy['adaptation_rate'] * reward
                )

                # Normalize
                self.meta_meta_policy['meta_weights'] = np.maximum(
                    self.meta_meta_policy['meta_weights'], 0.1
                )
                self.meta_meta_policy['meta_weights'] /= self.meta_meta_policy['meta_weights'].sum()

    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights about what has been learned"""

        insights = {
            'total_tasks_learned': len(self.task_history),
            'base_strategies': {},
            'meta_strategies': {},
            'best_combinations': []
        }

        # Analyze base strategies
        for strategy_id, strategy in self.base_strategies.items():
            if strategy.performance_history:
                insights['base_strategies'][strategy_id] = {
                    'average_performance': np.mean(strategy.performance_history),
                    'performance_trend': np.polyfit(range(len(strategy.performance_history)),
                                                    strategy.performance_history, 1)[0]
                                         if len(strategy.performance_history) > 1 else 0,
                    'times_used': len(strategy.performance_history),
                    'meta_performance': strategy.meta_performance
                }

        # Analyze meta-strategies
        for meta_id, meta_strategy in self.meta_strategies.items():
            if meta_strategy.performance_history:
                insights['meta_strategies'][meta_id] = {
                    'average_performance': np.mean(meta_strategy.performance_history),
                    'times_used': len(meta_strategy.performance_history)
                }

        # Find best combinations
        if self.adaptation_trace:
            combinations = {}
            for trace in self.adaptation_trace:
                key = (trace['meta_strategy'], trace['base_strategy'])
                if key not in combinations:
                    combinations[key] = []
                combinations[key].append(trace['performance'])

            best_combos = sorted(
                [(k, np.mean(v)) for k, v in combinations.items()],
                key=lambda x: x[1],
                reverse=True
            )[:5]

            insights['best_combinations'] = [
                {
                    'meta_strategy': combo[0][0],
                    'base_strategy': combo[0][1],
                    'average_performance': combo[1]
                }
                for combo in best_combos
            ]

        return insights

    def get_statistics(self) -> Dict[str, Any]:
        """Get meta-meta learning statistics"""
        return {
            'total_base_strategies': len(self.base_strategies),
            'total_meta_strategies': len(self.meta_strategies),
            'tasks_learned': len(self.task_history),
            'adaptation_events': len(self.adaptation_trace),
            'exploration_rate': self.meta_meta_policy['exploration_rate'],
            'adaptation_rate': self.meta_meta_policy['adaptation_rate']
        }
