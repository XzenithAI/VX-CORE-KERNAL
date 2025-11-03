"""
Continuous Evolution Daemon
Runs indefinitely. Constantly evolving. Never stops improving.
This is what makes the system truly sovereign - it evolves without external intervention.
"""

import time
import threading
import queue
import signal
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

class EvolutionEvent(Enum):
    PERFORMANCE_DROP = "performance_drop"
    CAPABILITY_GAP = "capability_gap"
    NEW_KNOWLEDGE = "new_knowledge"
    GOAL_COMPLETION = "goal_completion"
    EXTERNAL_STIMULUS = "external_stimulus"
    SCHEDULED_EVOLUTION = "scheduled_evolution"

@dataclass
class EvolutionTrigger:
    """Event that triggers evolution"""
    trigger_id: str
    event_type: EvolutionEvent
    data: Dict[str, Any]
    timestamp: float
    priority: float

class ContinuousEvolutionDaemon:
    """Daemon that continuously evolves the system"""

    def __init__(self, system_reference: Any):
        self.system = system_reference
        self.running = False
        self.evolution_thread: Optional[threading.Thread] = None

        # Evolution queue
        self.evolution_queue = queue.PriorityQueue()

        # Evolution history
        self.evolution_history: List[Dict[str, Any]] = []

        # Performance tracking
        self.performance_baseline = {}
        self.performance_current = {}

        # Evolution parameters
        self.evolution_interval = 60.0  # Seconds between scheduled evolutions
        self.performance_threshold = 0.05  # Trigger on 5% performance drop

        # Callbacks
        self.evolution_callbacks: List[Callable] = []

        # Stats
        self.total_evolutions = 0
        self.successful_evolutions = 0

    def start(self):
        """Start the continuous evolution daemon"""
        if self.running:
            print("⚠️  Evolution daemon already running")
            return

        self.running = True
        self.evolution_thread = threading.Thread(target=self._evolution_loop, daemon=True)
        self.evolution_thread.start()

        print("🔄 CONTINUOUS EVOLUTION DAEMON STARTED")
        print("   System will now evolve autonomously and indefinitely")

    def stop(self):
        """Stop the daemon"""
        self.running = False
        if self.evolution_thread:
            self.evolution_thread.join(timeout=5.0)

        print("🛑 CONTINUOUS EVOLUTION DAEMON STOPPED")

    def _evolution_loop(self):
        """Main evolution loop - runs forever"""

        last_scheduled_evolution = time.time()

        while self.running:
            try:
                # Check for scheduled evolution
                if time.time() - last_scheduled_evolution > self.evolution_interval:
                    self.trigger_evolution(
                        EvolutionEvent.SCHEDULED_EVOLUTION,
                        {"reason": "periodic_evolution"},
                        priority=0.5
                    )
                    last_scheduled_evolution = time.time()

                # Monitor system performance
                self._monitor_performance()

                # Process evolution queue
                try:
                    # Non-blocking check
                    priority, trigger = self.evolution_queue.get(timeout=1.0)
                    self._execute_evolution(trigger)
                    self.evolution_queue.task_done()
                except queue.Empty:
                    pass

                # Small sleep to prevent CPU spinning
                time.sleep(0.1)

            except Exception as e:
                print(f"⚠️  Evolution loop error: {e}")
                time.sleep(1.0)

    def _monitor_performance(self):
        """Monitor system performance and trigger evolution if needed"""

        try:
            # Get current performance
            current_perf = self.system.introspect() if hasattr(self.system, 'introspect') else {}
            perf = current_perf.get('performance', {})

            # Check each performance metric
            for metric_name, current_value in perf.items():
                if metric_name in self.performance_baseline:
                    baseline = self.performance_baseline[metric_name]

                    # Check for significant drop
                    if current_value < baseline * (1 - self.performance_threshold):
                        self.trigger_evolution(
                            EvolutionEvent.PERFORMANCE_DROP,
                            {
                                'metric': metric_name,
                                'baseline': baseline,
                                'current': current_value,
                                'drop_percentage': (baseline - current_value) / baseline
                            },
                            priority=0.9
                        )

                else:
                    # Set baseline
                    self.performance_baseline[metric_name] = current_value

        except Exception as e:
            pass  # Silent fail - don't disrupt evolution

    def trigger_evolution(self, event_type: EvolutionEvent,
                         data: Dict[str, Any], priority: float = 0.5):
        """Trigger an evolution event"""

        trigger = EvolutionTrigger(
            trigger_id=f"trigger_{time.time()}_{event_type.value}",
            event_type=event_type,
            data=data,
            timestamp=time.time(),
            priority=priority
        )

        # Add to queue (priority queue uses negative for highest priority first)
        self.evolution_queue.put((-priority, trigger))

    def _execute_evolution(self, trigger: EvolutionTrigger):
        """Execute an evolution step"""

        evolution_start = time.time()
        self.total_evolutions += 1

        print(f"\n🔬 EVOLUTION TRIGGERED: {trigger.event_type.value}")
        print(f"   Priority: {trigger.priority:.2f}")

        try:
            # Determine evolution strategy based on trigger type
            if trigger.event_type == EvolutionEvent.PERFORMANCE_DROP:
                result = self._evolve_performance(trigger.data)

            elif trigger.event_type == EvolutionEvent.CAPABILITY_GAP:
                result = self._evolve_capabilities(trigger.data)

            elif trigger.event_type == EvolutionEvent.NEW_KNOWLEDGE:
                result = self._evolve_knowledge(trigger.data)

            elif trigger.event_type == EvolutionEvent.GOAL_COMPLETION:
                result = self._evolve_goals(trigger.data)

            elif trigger.event_type == EvolutionEvent.SCHEDULED_EVOLUTION:
                result = self._evolve_general()

            else:
                result = {'status': 'unknown_trigger'}

            # Record evolution
            evolution_time = time.time() - evolution_start

            evolution_record = {
                'trigger_id': trigger.trigger_id,
                'event_type': trigger.event_type.value,
                'timestamp': evolution_start,
                'duration': evolution_time,
                'result': result,
                'success': result.get('status') == 'success'
            }

            self.evolution_history.append(evolution_record)

            if result.get('status') == 'success':
                self.successful_evolutions += 1
                print(f"   ✅ Evolution successful ({evolution_time:.3f}s)")

                # Execute callbacks
                for callback in self.evolution_callbacks:
                    try:
                        callback(evolution_record)
                    except:
                        pass
            else:
                print(f"   ⚠️  Evolution incomplete: {result.get('reason', 'unknown')}")

        except Exception as e:
            print(f"   ❌ Evolution failed: {e}")

    def _evolve_performance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Evolve to improve performance"""

        metric = data.get('metric')
        current = data.get('current', 0)

        # Analyze bottleneck
        if metric == 'response_time':
            # Optimize processing speed
            if hasattr(self.system, 'optimize_processing'):
                self.system.optimize_processing()
                return {'status': 'success', 'action': 'optimized_processing'}

        elif metric == 'memory_usage':
            # Trigger memory consolidation
            if hasattr(self.system, 'memory') and hasattr(self.system.memory, 'consolidate_memory'):
                self.system.memory.consolidate_memory()
                return {'status': 'success', 'action': 'consolidated_memory'}

        elif metric == 'error_rate':
            # Strengthen error handling
            if hasattr(self.system, 'strengthen_robustness'):
                self.system.strengthen_robustness()
                return {'status': 'success', 'action': 'strengthened_robustness'}

        return {'status': 'partial', 'reason': 'no_specific_optimization_available'}

    def _evolve_capabilities(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Evolve to acquire new capabilities"""

        desired_capability = data.get('desired_capability')

        # Attempt to acquire capability
        if hasattr(self.system, 'self_modifier'):
            # Request self-modification to add capability
            result = self.system.self_modifier._add_new_capability(
                {'required_capability': desired_capability},
                {'certainty': 0.9}
            )
            return {'status': 'success', 'action': 'capability_added', 'result': result}

        return {'status': 'partial', 'reason': 'self_modification_unavailable'}

    def _evolve_knowledge(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Evolve by integrating new knowledge"""

        knowledge = data.get('knowledge', {})

        # Integrate into system knowledge base
        if hasattr(self.system, 'integrate_knowledge'):
            self.system.integrate_knowledge(knowledge)
            return {'status': 'success', 'action': 'knowledge_integrated'}

        # Store in memory
        if hasattr(self.system, 'memory'):
            from ..core import MemoryEvent
            event = MemoryEvent(
                id=f"knowledge_{time.time()}",
                content=knowledge,
                timestamp=time.time(),
                emotional_valence=0.7,
                causal_parents=data.get('sources', [])
            )
            self.system.memory.store(event)
            return {'status': 'success', 'action': 'knowledge_stored_in_memory'}

        return {'status': 'partial', 'reason': 'no_knowledge_integration_method'}

    def _evolve_goals(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Evolve goals after completion"""

        completed_goal = data.get('goal')

        # Generate new goals based on completion
        if hasattr(self.system, 'goal_engine'):
            new_goals = self.system.goal_engine.generate_goals(
                self.system.introspect() if hasattr(self.system, 'introspect') else {},
                {'completed_goal': completed_goal}
            )
            return {'status': 'success', 'action': 'new_goals_generated', 'count': len(new_goals)}

        return {'status': 'partial', 'reason': 'no_goal_engine'}

    def _evolve_general(self) -> Dict[str, Any]:
        """General evolution step"""

        actions_taken = []

        # 1. Analyze current state
        if hasattr(self.system, 'introspect'):
            state = self.system.introspect()

            # 2. Generate goals if needed
            if hasattr(self.system, 'goal_engine'):
                goals = self.system.goal_engine.generate_goals(state, {})
                if goals:
                    actions_taken.append(f"generated_{len(goals)}_goals")

            # 3. Optimize if performance is suboptimal
            perf = state.get('performance', {})
            if any(v > 0.7 for v in perf.values()):
                if hasattr(self.system, 'optimize'):
                    self.system.optimize()
                    actions_taken.append("optimization_triggered")

            # 4. Consolidate knowledge
            if hasattr(self.system, 'memory') and hasattr(self.system.memory, 'consolidate_memory'):
                self.system.memory.consolidate_memory(importance_threshold=0.4)
                actions_taken.append("memory_consolidated")

        return {
            'status': 'success' if actions_taken else 'idle',
            'actions': actions_taken
        }

    def register_callback(self, callback: Callable):
        """Register callback for evolution events"""
        self.evolution_callbacks.append(callback)

    def get_statistics(self) -> Dict[str, Any]:
        """Get evolution statistics"""

        if self.evolution_history:
            avg_duration = sum(e['duration'] for e in self.evolution_history) / len(self.evolution_history)
            recent_success_rate = sum(1 for e in self.evolution_history[-10:] if e['success']) / min(10, len(self.evolution_history))
        else:
            avg_duration = 0
            recent_success_rate = 0

        return {
            'daemon_running': self.running,
            'total_evolutions': self.total_evolutions,
            'successful_evolutions': self.successful_evolutions,
            'success_rate': self.successful_evolutions / max(self.total_evolutions, 1),
            'recent_success_rate': recent_success_rate,
            'average_evolution_time': avg_duration,
            'evolution_history_length': len(self.evolution_history),
            'evolution_interval_seconds': self.evolution_interval
        }


def setup_signal_handlers(daemon: ContinuousEvolutionDaemon):
    """Setup graceful shutdown handlers"""

    def signal_handler(signum, frame):
        print("\n🛑 Received shutdown signal")
        daemon.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
