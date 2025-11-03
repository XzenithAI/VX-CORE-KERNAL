"""
Neuro-Symbolic Reasoning Engine
Bridges neural pattern matching with symbolic logic
"""
import time
import numpy as np
from typing import Dict, Any, Callable, List, Optional
from ..core import MemoryEvent
from ..memory.causal_memory import CausalMemory

class NeuroSymbolicReasoner:
    """Bridges neural pattern matching with symbolic logic"""

    def __init__(self, memory: CausalMemory):
        self.memory = memory
        self.logic_rules = []
        self.neural_model = self._initialize_neural_model()
        self.reasoning_history = []
        self.rule_execution_count = {}

    def _initialize_neural_model(self):
        """Initialize neural network for pattern completion"""
        try:
            import torch
            import torch.nn as nn

            class PatternCompletionNet(nn.Module):
                def __init__(self, input_size=64, hidden_size=128, output_size=64):
                    super().__init__()
                    self.encoder = nn.Sequential(
                        nn.Linear(input_size, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_size, hidden_size // 2),
                        nn.ReLU()
                    )
                    self.decoder = nn.Sequential(
                        nn.Linear(hidden_size // 2, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_size, output_size),
                        nn.Tanh()
                    )

                def forward(self, x):
                    encoded = self.encoder(x)
                    return self.decoder(encoded)

            model = PatternCompletionNet()
            model.eval()  # Set to evaluation mode
            return model

        except ImportError:
            # Fallback to numpy implementation
            class SimpleNeuralModel:
                def __init__(self):
                    self.weights1 = np.random.randn(64, 128) * 0.1
                    self.weights2 = np.random.randn(128, 64) * 0.1
                    self.bias1 = np.zeros(128)
                    self.bias2 = np.zeros(64)

                def forward(self, x):
                    # Two-layer network
                    hidden = np.tanh(np.dot(x, self.weights1) + self.bias1)
                    output = np.tanh(np.dot(hidden, self.weights2) + self.bias2)
                    return output

            return SimpleNeuralModel()

    def add_symbolic_rule(self, condition: Callable, action: Callable, priority: int = 1, name: str = "unnamed"):
        """Add a symbolic logic rule"""
        rule = {
            'name': name,
            'condition': condition,
            'action': action,
            'priority': priority,
            'last_execution': None,
            'execution_count': 0,
            'success_count': 0,
            'failure_count': 0
        }
        self.logic_rules.append(rule)
        self.rule_execution_count[name] = 0

    def reason(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute neural-symbolic reasoning cycle"""
        reasoning_start = time.time()

        # Phase 1: Neural pattern completion
        neural_input = self._state_to_tensor(current_state)

        try:
            if hasattr(self.neural_model, 'forward'):
                neural_output = self.neural_model.forward(neural_input)
            else:
                neural_output = self.neural_model(neural_input)
        except Exception as e:
            print(f"Neural processing warning: {e}")
            neural_output = neural_input  # Fallback

        neural_insight = self._tensor_to_insight(neural_output)

        # Phase 2: Symbolic rule application
        symbolic_actions = []
        rules_fired = []

        for rule in sorted(self.logic_rules, key=lambda x: x['priority'], reverse=True):
            try:
                rule['execution_count'] += 1

                if rule['condition'](current_state, neural_insight):
                    action_result = rule['action'](current_state, neural_insight)
                    symbolic_actions.append(action_result)
                    rules_fired.append(rule['name'])
                    rule['success_count'] += 1
                    rule['last_execution'] = time.time()

            except Exception as e:
                rule['failure_count'] += 1
                self._learn_from_failure(rule, e, current_state)

        # Phase 3: Integration
        integrated_decisions = self._integrate_modalities(
            neural_insight, symbolic_actions, current_state
        )

        reasoning_time = time.time() - reasoning_start

        result = {
            'neural_insight': neural_insight,
            'symbolic_actions': symbolic_actions,
            'rules_fired': rules_fired,
            'integrated_decisions': integrated_decisions,
            'reasoning_time': reasoning_time,
            'timestamp': time.time()
        }

        # Store reasoning in history
        self.reasoning_history.append(result)
        if len(self.reasoning_history) > 1000:
            self.reasoning_history = self.reasoning_history[-1000:]

        return result

    def _state_to_tensor(self, state: Dict) -> np.ndarray:
        """Convert state dictionary to neural network input"""
        # Create a more sophisticated embedding
        features = []

        # Encode consciousness state
        if 'consciousness_state' in state:
            cs = state['consciousness_state']
            cs_vector = np.zeros(4)
            if hasattr(cs, 'value'):
                cs_vector[cs.value] = 1.0
            features.append(cs_vector)

        # Encode performance metrics
        if 'performance' in state:
            perf = state['performance']
            perf_vector = np.array([
                perf.get('response_time', 0.5),
                perf.get('memory_usage', 0.5),
                perf.get('reasoning_speed', 0.5)
            ])
            features.append(perf_vector)

        # Encode memory context
        if 'memory_context' in state:
            mem_ctx = state['memory_context']
            mem_vector = np.array([len(mem_ctx), min(len(mem_ctx), 10)])
            features.append(mem_vector)

        # Concatenate and pad/truncate to 64 dimensions
        if features:
            combined = np.concatenate(features)
        else:
            combined = np.array([])

        if len(combined) < 64:
            # Pad with hash-based features
            state_str = str(state)
            hash_val = hash(state_str) % (2**32)
            np.random.seed(hash_val)
            padding = np.random.randn(64 - len(combined)) * 0.1
            combined = np.concatenate([combined, padding])
        else:
            combined = combined[:64]

        return combined

    def _tensor_to_insight(self, tensor) -> Dict[str, Any]:
        """Convert neural output to symbolic insights"""
        if hasattr(tensor, 'detach'):
            tensor = tensor.detach().numpy()

        # Extract meaningful features from neural output
        mean_activation = float(np.abs(tensor).mean())
        std_activation = float(np.std(tensor))
        max_activation = float(np.max(tensor))
        entropy = float(-np.sum(np.abs(tensor) * np.log(np.abs(tensor) + 1e-10)))

        return {
            'optimization_urgency': mean_activation,
            'optimization_target': 'reasoning' if tensor[0] > 0 else 'memory',
            'certainty': 1.0 - std_activation,  # Low variance = high certainty
            'attention_focus': float(max_activation),
            'entropy': entropy,
            'recommended_exploration': entropy > 2.0,
            'neural_state': tensor
        }

    def _integrate_modalities(self, neural_insight: Dict, symbolic_actions: List, current_state: Dict) -> Dict:
        """Integrate neural and symbolic reasoning"""

        # Calculate confidence based on agreement
        base_certainty = neural_insight.get('certainty', 0.5)
        action_agreement = len(symbolic_actions) / max(len(self.logic_rules), 1)

        integrated_certainty = (base_certainty + action_agreement) / 2.0

        # Determine improvement potential
        improvement_potential = len(symbolic_actions) * 0.15
        if neural_insight.get('recommended_exploration', False):
            improvement_potential += 0.2

        # Select top actions
        prioritized_actions = symbolic_actions[:5]

        return {
            'certainty': integrated_certainty,
            'improvement_potential': min(improvement_potential, 1.0),
            'recommended_actions': prioritized_actions,
            'exploration_mode': neural_insight.get('recommended_exploration', False),
            'attention_focus': neural_insight.get('attention_focus', 0.5)
        }

    def _learn_from_failure(self, failed_rule: Dict, error: Exception, state: Dict):
        """Convert reasoning failures into learning events"""
        failure_event = MemoryEvent(
            id=f"failure_{failed_rule['name']}_{time.time()}",
            content=f"Rule '{failed_rule['name']}' failed: {str(error)}",
            timestamp=time.time(),
            emotional_valence=-0.8,
            causal_parents=[failed_rule.get('last_execution', 'unknown')],
            confidence=0.3
        )
        self.memory.store(failure_event)

    def get_reasoning_statistics(self) -> Dict[str, Any]:
        """Get statistics about reasoning performance"""
        if not self.logic_rules:
            return {'total_rules': 0}

        total_executions = sum(r['execution_count'] for r in self.logic_rules)
        total_successes = sum(r['success_count'] for r in self.logic_rules)
        total_failures = sum(r['failure_count'] for r in self.logic_rules)

        return {
            'total_rules': len(self.logic_rules),
            'total_executions': total_executions,
            'total_successes': total_successes,
            'total_failures': total_failures,
            'success_rate': total_successes / max(total_executions, 1),
            'reasoning_history_length': len(self.reasoning_history),
            'average_reasoning_time': np.mean([r['reasoning_time'] for r in self.reasoning_history]) if self.reasoning_history else 0
        }
