"""
Meta-Learning and Adaptive Architecture
System learns how to learn better over time
"""
import numpy as np
import time
from typing import Dict, Any, List, Optional, Callable
from collections import deque

class LearningStrategy:
    """Represents a learning strategy that can be evaluated and optimized"""

    def __init__(self, name: str, parameters: Dict[str, Any]):
        self.name = name
        self.parameters = parameters
        self.performance_history = []
        self.success_count = 0
        self.failure_count = 0
        self.average_performance = 0.0

    def update_performance(self, performance: float):
        """Update performance metrics"""
        self.performance_history.append({
            'timestamp': time.time(),
            'performance': performance
        })

        if performance > 0.5:
            self.success_count += 1
        else:
            self.failure_count += 1

        # Calculate moving average
        recent = [p['performance'] for p in self.performance_history[-10:]]
        self.average_performance = np.mean(recent)

    def get_statistics(self) -> Dict[str, Any]:
        """Get strategy statistics"""
        total = self.success_count + self.failure_count
        return {
            'name': self.name,
            'parameters': self.parameters,
            'success_rate': self.success_count / max(total, 1),
            'average_performance': self.average_performance,
            'total_uses': len(self.performance_history)
        }


class MetaLearner:
    """Learns which learning strategies work best"""

    def __init__(self):
        self.strategies: Dict[str, LearningStrategy] = {}
        self.strategy_selection_history = []
        self.performance_predictor = self._initialize_predictor()
        self.adaptation_rate = 0.1

    def _initialize_predictor(self) -> Dict[str, Any]:
        """Initialize performance prediction model"""
        return {
            'weights': np.random.randn(10) * 0.1,
            'bias': 0.0,
            'learning_rate': 0.01
        }

    def register_strategy(self, name: str, parameters: Dict[str, Any]):
        """Register a new learning strategy"""
        self.strategies[name] = LearningStrategy(name, parameters)

    def select_strategy(self, context: Dict[str, Any]) -> str:
        """Select best strategy for current context using meta-learning"""
        if not self.strategies:
            return "default"

        # Predict performance for each strategy
        predictions = {}
        for name, strategy in self.strategies.items():
            context_features = self._extract_context_features(context, strategy)
            predicted_performance = self._predict_performance(context_features)
            predictions[name] = predicted_performance

        # Select strategy with highest predicted performance
        # with some exploration (epsilon-greedy)
        if np.random.random() < 0.1:  # 10% exploration
            selected = np.random.choice(list(self.strategies.keys()))
        else:
            selected = max(predictions.items(), key=lambda x: x[1])[0]

        self.strategy_selection_history.append({
            'timestamp': time.time(),
            'selected': selected,
            'context': context,
            'predictions': predictions
        })

        return selected

    def update_from_outcome(self, strategy_name: str, performance: float,
                           context: Dict[str, Any]):
        """Update meta-learner based on actual performance"""
        if strategy_name not in self.strategies:
            return

        strategy = self.strategies[strategy_name]
        strategy.update_performance(performance)

        # Update performance predictor
        context_features = self._extract_context_features(context, strategy)
        predicted = self._predict_performance(context_features)
        error = performance - predicted

        # Gradient descent update
        self.performance_predictor['weights'] -= (
            self.performance_predictor['learning_rate'] * error * context_features
        )
        self.performance_predictor['bias'] -= (
            self.performance_predictor['learning_rate'] * error
        )

    def _extract_context_features(self, context: Dict[str, Any],
                                  strategy: LearningStrategy) -> np.ndarray:
        """Extract features from context for prediction"""
        features = []

        # Context features
        features.append(context.get('complexity', 0.5))
        features.append(context.get('uncertainty', 0.5))
        features.append(context.get('novelty', 0.5))

        # Strategy features
        features.append(strategy.average_performance)
        features.append(strategy.success_count / max(len(strategy.performance_history), 1))

        # Interaction features
        features.append(context.get('complexity', 0.5) * strategy.average_performance)

        # Pad to 10 features
        while len(features) < 10:
            features.append(0.0)

        return np.array(features[:10])

    def _predict_performance(self, features: np.ndarray) -> float:
        """Predict performance using learned model"""
        prediction = (np.dot(self.performance_predictor['weights'], features) +
                     self.performance_predictor['bias'])
        # Sigmoid to bound between 0 and 1
        return float(1.0 / (1.0 + np.exp(-prediction)))

    def get_best_strategies(self, top_k: int = 3) -> List[Dict[str, Any]]:
        """Get top performing strategies"""
        strategy_stats = [s.get_statistics() for s in self.strategies.values()]
        strategy_stats.sort(key=lambda x: x['average_performance'], reverse=True)
        return strategy_stats[:top_k]


class AdaptiveArchitecture:
    """Self-adapting neural architecture"""

    def __init__(self, initial_config: Dict[str, Any]):
        self.config = initial_config
        self.architecture_history = [initial_config.copy()]
        self.performance_history = []
        self.adaptation_count = 0

        # Architecture search space
        self.search_space = {
            'layer_sizes': [32, 64, 128, 256],
            'activation_functions': ['relu', 'tanh', 'sigmoid', 'swish'],
            'learning_rates': [0.001, 0.01, 0.1],
            'dropout_rates': [0.0, 0.1, 0.2, 0.3]
        }

        # Performance tracking for architecture search
        self.architecture_performance = {}

    def propose_architecture_change(self, current_performance: float,
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """Propose architecture modification based on performance"""
        # Decide if we should adapt
        should_adapt = self._should_adapt(current_performance)

        if not should_adapt:
            return {'action': 'no_change', 'reason': 'performance_acceptable'}

        # Determine what to change
        change_type = self._select_change_type(context)

        # Generate new configuration
        new_config = self.config.copy()

        if change_type == 'add_layer':
            new_size = np.random.choice(self.search_space['layer_sizes'])
            new_config['layers'] = new_config.get('layers', [64, 32]) + [new_size]

        elif change_type == 'modify_layer':
            if 'layers' in new_config and new_config['layers']:
                layer_idx = np.random.randint(len(new_config['layers']))
                new_size = np.random.choice(self.search_space['layer_sizes'])
                new_config['layers'][layer_idx] = new_size

        elif change_type == 'change_learning_rate':
            new_config['learning_rate'] = np.random.choice(self.search_space['learning_rates'])

        elif change_type == 'adjust_dropout':
            new_config['dropout'] = np.random.choice(self.search_space['dropout_rates'])

        return {
            'action': 'propose_change',
            'change_type': change_type,
            'old_config': self.config,
            'new_config': new_config,
            'reason': f'performance {current_performance:.2f} triggered adaptation'
        }

    def apply_architecture_change(self, new_config: Dict[str, Any]):
        """Apply proposed architecture change"""
        self.config = new_config
        self.architecture_history.append(new_config.copy())
        self.adaptation_count += 1

    def _should_adapt(self, current_performance: float) -> bool:
        """Decide if architecture should be adapted"""
        if len(self.performance_history) < 5:
            return False

        # Check if performance is stagnating
        recent_performance = [p['performance'] for p in self.performance_history[-5:]]
        performance_variance = np.var(recent_performance)

        if performance_variance < 0.01:  # Stagnating
            return True

        # Check if performance is declining
        if len(recent_performance) >= 2:
            if recent_performance[-1] < recent_performance[-2] - 0.1:
                return True

        # Periodic adaptation every 50 evaluations
        if len(self.performance_history) % 50 == 0:
            return True

        return False

    def _select_change_type(self, context: Dict[str, Any]) -> str:
        """Select what type of architecture change to make"""
        complexity = context.get('complexity', 0.5)
        uncertainty = context.get('uncertainty', 0.5)

        # High complexity -> add capacity
        if complexity > 0.7:
            return 'add_layer'

        # High uncertainty -> adjust learning dynamics
        if uncertainty > 0.7:
            return 'change_learning_rate'

        # Otherwise, modify existing structure
        return np.random.choice(['modify_layer', 'adjust_dropout'])

    def record_performance(self, performance: float, config: Dict[str, Any]):
        """Record performance for current configuration"""
        self.performance_history.append({
            'timestamp': time.time(),
            'performance': performance,
            'config': config.copy()
        })

        # Track architecture-specific performance
        config_key = self._config_to_key(config)
        if config_key not in self.architecture_performance:
            self.architecture_performance[config_key] = []
        self.architecture_performance[config_key].append(performance)

    def _config_to_key(self, config: Dict[str, Any]) -> str:
        """Convert config to hashable key"""
        return str(sorted(config.items()))

    def get_statistics(self) -> Dict[str, Any]:
        """Get adaptive architecture statistics"""
        return {
            'current_config': self.config,
            'adaptation_count': self.adaptation_count,
            'architectures_explored': len(self.architecture_performance),
            'total_evaluations': len(self.performance_history),
            'current_performance': self.performance_history[-1]['performance']
                                  if self.performance_history else 0.0
        }


class ContinualLearningBuffer:
    """Buffer for continual learning that prevents catastrophic forgetting"""

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.importance_weights = {}
        self.access_counts = {}

    def add(self, experience: Dict[str, Any], importance: float = 1.0):
        """Add experience to buffer"""
        exp_id = f"exp_{time.time()}_{hash(str(experience))}"
        experience['id'] = exp_id
        experience['added_at'] = time.time()

        self.buffer.append(experience)
        self.importance_weights[exp_id] = importance
        self.access_counts[exp_id] = 0

    def sample(self, batch_size: int = 32) -> List[Dict[str, Any]]:
        """Sample experiences for replay"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)

        # Sample based on importance weights
        buffer_list = list(self.buffer)
        ids = [exp['id'] for exp in buffer_list]
        weights = np.array([self.importance_weights.get(exp_id, 1.0) for exp_id in ids])

        # Normalize weights
        weights = weights / weights.sum()

        # Sample
        indices = np.random.choice(len(buffer_list), size=batch_size,
                                  replace=False, p=weights)

        sampled = [buffer_list[i] for i in indices]

        # Update access counts
        for exp in sampled:
            self.access_counts[exp['id']] += 1

        return sampled

    def update_importance(self, exp_id: str, new_importance: float):
        """Update importance weight for an experience"""
        if exp_id in self.importance_weights:
            self.importance_weights[exp_id] = new_importance

    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        if not self.buffer:
            return {'size': 0, 'capacity': self.capacity}

        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'utilization': len(self.buffer) / self.capacity,
            'average_importance': np.mean(list(self.importance_weights.values())),
            'total_accesses': sum(self.access_counts.values())
        }
