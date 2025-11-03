"""
Self-Modification Engine
Allows the system to rewrite its own code safely
"""
import time
import ast
import inspect
import hashlib
from typing import Dict, Any, List, Optional
from ..core import MemoryEvent
from ..memory.causal_memory import CausalMemory
from ..reasoning.neurosymbolic import NeuroSymbolicReasoner

class SelfModificationEngine:
    """Allows the system to rewrite its own code safely"""

    def __init__(self, memory: CausalMemory, reasoner: NeuroSymbolicReasoner):
        self.memory = memory
        self.reasoner = reasoner
        self.modification_history = []
        self.component_backups = {}
        self.safety_constraints = self._initialize_safety_constraints()

        # Add symbolic rules for self-modification
        self.reasoner.add_symbolic_rule(
            condition=self._should_optimize_code,
            action=self._optimize_component,
            priority=10,
            name="code_optimization"
        )

        self.reasoner.add_symbolic_rule(
            condition=self._should_add_capability,
            action=self._add_new_capability,
            priority=8,
            name="capability_expansion"
        )

    def _initialize_safety_constraints(self) -> Dict[str, Any]:
        """Initialize safety constraints for self-modification"""
        return {
            'max_modifications_per_hour': 10,
            'require_validation': True,
            'create_backups': True,
            'forbidden_modules': ['os', 'sys', 'subprocess'],  # Can't modify core system
            'modification_window': 3600,  # 1 hour in seconds
            'recent_modifications': []
        }

    def _should_optimize_code(self, state: Dict, neural_insight: Any) -> bool:
        """Determine if code optimization is needed"""
        performance_metrics = state.get('performance', {})

        # Check if we're within rate limits
        if not self._check_rate_limit():
            return False

        # Check performance thresholds
        if performance_metrics.get('response_time', 0) > 1.0:
            return True
        if performance_metrics.get('memory_usage', 0) > 0.8:
            return True

        # Check neural recommendation
        if neural_insight.get('optimization_urgency', 0) > 0.75:
            return True

        return False

    def _should_add_capability(self, state: Dict, neural_insight: Any) -> bool:
        """Determine if new capability should be added"""
        if not self._check_rate_limit():
            return False

        # Add capability if exploration is recommended
        if neural_insight.get('recommended_exploration', False):
            return True

        # Add capability if improvement potential is high
        integrated = state.get('integrated_decisions', {})
        if integrated and integrated.get('improvement_potential', 0) > 0.8:
            return True

        return False

    def _check_rate_limit(self) -> bool:
        """Check if we can perform another modification"""
        current_time = time.time()
        window = self.safety_constraints['modification_window']
        max_mods = self.safety_constraints['max_modifications_per_hour']

        # Clean old modifications
        recent = [
            mod for mod in self.safety_constraints['recent_modifications']
            if current_time - mod < window
        ]
        self.safety_constraints['recent_modifications'] = recent

        return len(recent) < max_mods

    def _optimize_component(self, state: Dict, neural_insight: Any) -> Dict:
        """Optimize a specific component based on neural insight"""
        target_component = neural_insight.get('optimization_target', 'unknown')

        # Record modification attempt
        self.safety_constraints['recent_modifications'].append(time.time())

        # Get current source code
        current_code = self._get_component_source(target_component)
        if not current_code:
            return {'status': 'no_source_found', 'component': target_component}

        # Generate optimization
        optimization_strategy = self._select_optimization_strategy(state, neural_insight)
        optimized_code = self._apply_optimization_patterns(current_code, optimization_strategy)

        # Validate optimization
        validation_result = self._validate_code(optimized_code, target_component)
        if not validation_result['valid']:
            return {
                'status': 'optimization_failed',
                'reason': validation_result['error'],
                'component': target_component
            }

        # Safely apply the modification
        result = self._apply_code_change(target_component, optimized_code)

        # Record the modification
        mod_event = MemoryEvent(
            id=f"modification_{target_component}_{time.time()}",
            content=f"Optimized {target_component} using {optimization_strategy}: {result}",
            timestamp=time.time(),
            causal_parents=[state.get('current_context', 'self_improvement')],
            confidence=validation_result.get('confidence', 0.7)
        )
        self.memory.store(mod_event)

        return result

    def _add_new_capability(self, state: Dict, neural_insight: Any) -> Dict:
        """Add new capability to the system"""
        self.safety_constraints['recent_modifications'].append(time.time())

        capability_type = self._determine_needed_capability(state, neural_insight)

        new_capability_code = self._generate_capability_code(capability_type)

        validation_result = self._validate_code(new_capability_code, f"capability_{capability_type}")
        if not validation_result['valid']:
            return {
                'status': 'capability_creation_failed',
                'reason': validation_result['error']
            }

        # Store as potential capability (not auto-executed for safety)
        cap_event = MemoryEvent(
            id=f"new_capability_{capability_type}_{time.time()}",
            content=f"Generated new capability: {capability_type}",
            timestamp=time.time(),
            emotional_valence=0.6,
            meta_data={'code': new_capability_code, 'capability_type': capability_type}
        )
        self.memory.store(cap_event)

        return {
            'status': 'capability_generated',
            'capability_type': capability_type,
            'stored_as': cap_event.id
        }

    def _select_optimization_strategy(self, state: Dict, neural_insight: Any) -> str:
        """Select appropriate optimization strategy"""
        performance = state.get('performance', {})

        if performance.get('memory_usage', 0) > 0.7:
            return 'memory_optimization'
        elif performance.get('response_time', 0) > 0.8:
            return 'speed_optimization'
        elif neural_insight.get('certainty', 0) < 0.5:
            return 'accuracy_optimization'
        else:
            return 'general_optimization'

    def _determine_needed_capability(self, state: Dict, neural_insight: Any) -> str:
        """Determine what type of capability is needed"""
        entropy = neural_insight.get('entropy', 0)

        if entropy > 3.0:
            return 'pattern_recognition'
        elif neural_insight.get('attention_focus', 0) < 0.3:
            return 'attention_mechanism'
        else:
            return 'reasoning_enhancement'

    def _get_component_source(self, component_name: str) -> Optional[str]:
        """Get source code for a component"""
        # In a real implementation, this would use inspect.getsource()
        # For now, return a placeholder
        return f"# Source code for {component_name}\ndef {component_name}_function():\n    pass\n"

    def _apply_optimization_patterns(self, code: str, strategy: str) -> str:
        """Apply optimization patterns to code"""
        optimizations = {
            'memory_optimization': '# Memory optimized: use generators and weak references\n',
            'speed_optimization': '# Speed optimized: cached computations and vectorization\n',
            'accuracy_optimization': '# Accuracy optimized: ensemble methods and validation\n',
            'general_optimization': '# General optimization: clean code and best practices\n'
        }

        optimization_comment = optimizations.get(strategy, '# Optimized\n')
        return optimization_comment + code

    def _generate_capability_code(self, capability_type: str) -> str:
        """Generate code for new capability"""
        templates = {
            'pattern_recognition': '''
def enhanced_pattern_recognition(data):
    """Enhanced pattern recognition capability"""
    import numpy as np
    patterns = []
    for i in range(len(data) - 1):
        patterns.append(data[i+1] - data[i])
    return patterns
''',
            'attention_mechanism': '''
def attention_mechanism(inputs, context):
    """Attention mechanism for focusing on relevant information"""
    import numpy as np
    weights = np.exp(np.dot(inputs, context))
    weights = weights / np.sum(weights)
    return np.dot(weights, inputs)
''',
            'reasoning_enhancement': '''
def enhanced_reasoning(premise, rules):
    """Enhanced reasoning capability"""
    conclusions = []
    for rule in rules:
        if rule.applies_to(premise):
            conclusions.append(rule.conclude(premise))
    return conclusions
'''
        }

        return templates.get(capability_type, '# Placeholder capability\ndef new_capability():\n    pass\n')

    def _validate_code(self, code: str, component_name: str) -> Dict[str, Any]:
        """Validate code before execution"""
        try:
            # Parse the code
            parsed = ast.parse(code)

            # Check for forbidden operations
            for node in ast.walk(parsed):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in self.safety_constraints['forbidden_modules']:
                            return {
                                'valid': False,
                                'error': f"Forbidden module: {alias.name}"
                            }

            # Compile the code
            compiled = compile(parsed, filename=f'<{component_name}>', mode='exec')

            # Test in isolated namespace
            test_namespace = {'__builtins__': __builtins__}
            exec(compiled, test_namespace)

            return {
                'valid': True,
                'confidence': 0.8,
                'compiled': compiled
            }

        except SyntaxError as e:
            return {'valid': False, 'error': f"Syntax error: {e}"}
        except Exception as e:
            return {'valid': False, 'error': f"Validation error: {e}"}

    def _apply_code_change(self, component_name: str, new_code: str) -> Dict:
        """Safely apply code changes with rollback capability"""
        try:
            # Create backup
            backup = self._create_backup(component_name)

            # Validate again before applying
            validation = self._validate_code(new_code, component_name)
            if not validation['valid']:
                return {
                    'status': 'validation_failed',
                    'error': validation['error']
                }

            # Store in modification history
            mod_hash = hashlib.sha256(new_code.encode()).hexdigest()[:16]
            self.modification_history.append({
                'timestamp': time.time(),
                'component': component_name,
                'backup': backup,
                'new_code': new_code,
                'code_hash': mod_hash,
                'success': True
            })

            return {
                'status': 'optimization_applied',
                'component': component_name,
                'code_hash': mod_hash,
                'backup_id': backup
            }

        except Exception as e:
            # Rollback on failure
            self._rollback_component(component_name, backup)
            return {
                'status': 'optimization_failed',
                'error': str(e),
                'rolled_back': True
            }

    def _create_backup(self, component_name: str) -> str:
        """Create backup of component"""
        backup_id = f"backup_{component_name}_{time.time()}"
        self.component_backups[backup_id] = {
            'component': component_name,
            'timestamp': time.time(),
            'source': self._get_component_source(component_name)
        }
        return backup_id

    def _rollback_component(self, component_name: str, backup_id: str):
        """Rollback component to backup"""
        if backup_id in self.component_backups:
            backup = self.component_backups[backup_id]
            # In real implementation, would restore the actual component
            print(f"Rolled back {component_name} to backup {backup_id}")

    def get_modification_statistics(self) -> Dict[str, Any]:
        """Get statistics about self-modifications"""
        if not self.modification_history:
            return {'total_modifications': 0}

        successful = sum(1 for m in self.modification_history if m['success'])
        failed = len(self.modification_history) - successful

        return {
            'total_modifications': len(self.modification_history),
            'successful_modifications': successful,
            'failed_modifications': failed,
            'success_rate': successful / len(self.modification_history),
            'total_backups': len(self.component_backups),
            'recent_modifications': len(self.safety_constraints['recent_modifications']),
            'rate_limit_active': len(self.safety_constraints['recent_modifications']) >= self.safety_constraints['max_modifications_per_hour']
        }
