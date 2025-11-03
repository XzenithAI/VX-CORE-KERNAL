#!/usr/bin/env python3
"""
Comprehensive Test Suite for VX-GENESIS-OS
Tests all core and advanced features
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_core_memory():
    """Test causal memory system"""
    print("🧪 Testing Causal Memory System...")
    from genesis.memory.causal_memory import CausalMemory
    from genesis.core import MemoryEvent

    memory = CausalMemory()

    # Create test events
    event1 = MemoryEvent(
        id="test_event_1",
        content="First test event",
        timestamp=time.time(),
        causal_parents=[]
    )

    event2 = MemoryEvent(
        id="test_event_2",
        content="Second test event",
        timestamp=time.time() + 1,
        causal_parents=["test_event_1"]
    )

    memory.store(event1)
    memory.store(event2)

    # Test retrieval
    assert "test_event_1" in memory.memories
    assert "test_event_2" in memory.memories

    # Test causal explanation
    explanation = memory.explain_why("test_event_2")
    assert 'narrative' in explanation
    assert len(explanation.get('causes', [])) > 0

    # Test statistics
    stats = memory.get_memory_statistics()
    assert stats['total_memories'] == 2
    assert stats['causal_edges'] > 0

    print("   ✓ Causal memory system working correctly")
    return True

def test_reasoning():
    """Test neuro-symbolic reasoning"""
    print("🧪 Testing Neuro-Symbolic Reasoning...")
    from genesis.reasoning.neurosymbolic import NeuroSymbolicReasoner
    from genesis.memory.causal_memory import CausalMemory

    memory = CausalMemory()
    reasoner = NeuroSymbolicReasoner(memory)

    # Add test rule
    def test_condition(state, insight):
        return state.get('test_flag', False)

    def test_action(state, insight):
        return {'action': 'test_executed'}

    reasoner.add_symbolic_rule(test_condition, test_action, priority=10, name="test_rule")

    # Test reasoning
    state = {'test_flag': True}
    result = reasoner.reason(state)

    assert 'neural_insight' in result
    assert 'symbolic_actions' in result
    assert 'integrated_decisions' in result

    stats = reasoner.get_reasoning_statistics()
    assert stats['total_rules'] >= 1

    print("   ✓ Neuro-symbolic reasoning working correctly")
    return True

def test_self_modification():
    """Test self-modification engine"""
    print("🧪 Testing Self-Modification Engine...")
    from genesis.self_modification.evolution import SelfModificationEngine
    from genesis.reasoning.neurosymbolic import NeuroSymbolicReasoner
    from genesis.memory.causal_memory import CausalMemory

    memory = CausalMemory()
    reasoner = NeuroSymbolicReasoner(memory)
    modifier = SelfModificationEngine(memory, reasoner)

    # Test safety constraints
    assert modifier.safety_constraints['max_modifications_per_hour'] > 0
    assert modifier.safety_constraints['require_validation']

    # Test rate limiting
    assert modifier._check_rate_limit()

    # Test statistics
    stats = modifier.get_modification_statistics()
    assert 'total_modifications' in stats

    print("   ✓ Self-modification engine working correctly")
    return True

def test_quantum_processing():
    """Test quantum-inspired processing"""
    print("🧪 Testing Quantum-Inspired Processing...")
    from genesis.quantum.quantum_neural import QuantumInspiredProcessor, QuantumState

    processor = QuantumInspiredProcessor(state_dimension=32)

    # Test hypothesis processing
    hypotheses = [
        {'hypothesis': 'A', 'score': 0.5},
        {'hypothesis': 'B', 'score': 0.7}
    ]

    result = processor.process_parallel_hypotheses(hypotheses)
    assert 'best_hypothesis' in result
    assert 'confidence' in result
    assert 'entropy' in result

    # Test quantum states
    state = QuantumState(dimension=32)
    assert state.dimension == 32
    assert state.amplitudes is not None

    # Test measurement
    measured, confidence = state.measure()
    assert measured is not None
    assert 0 <= confidence <= 1

    stats = processor.get_statistics()
    assert stats['state_dimension'] == 32

    print("   ✓ Quantum-inspired processing working correctly")
    return True

def test_meta_learning():
    """Test meta-learning system"""
    print("🧪 Testing Meta-Learning System...")
    from genesis.meta_learning.adaptive_architecture import MetaLearner, AdaptiveArchitecture

    meta_learner = MetaLearner()

    # Register strategies
    meta_learner.register_strategy('test_strategy', {'param': 1.0})
    assert 'test_strategy' in meta_learner.strategies

    # Test strategy selection
    context = {'complexity': 0.5, 'uncertainty': 0.5, 'novelty': 0.5}
    selected = meta_learner.select_strategy(context)
    assert selected is not None

    # Test adaptive architecture
    arch = AdaptiveArchitecture({'layers': [64, 32]})
    arch.record_performance(0.7, arch.config)

    stats = arch.get_statistics()
    assert 'current_config' in stats
    assert stats['total_evaluations'] > 0

    print("   ✓ Meta-learning system working correctly")
    return True

def test_swarm_intelligence():
    """Test multi-agent swarm"""
    print("🧪 Testing Multi-Agent Swarm...")
    from genesis.swarm.multi_agent import SwarmCoordinator, SwarmAgent, AgentRole

    coordinator = SwarmCoordinator()

    # Create agents
    agent_count = coordinator.create_agent_swarm(
        num_explorers=1,
        num_optimizers=1,
        num_validators=1
    )

    assert agent_count == 3
    assert len(coordinator.agents) == 3

    # Test task execution
    task = {'type': 'exploration', 'data': 'test'}
    result = coordinator.execute_collaborative_task(task)

    assert result['status'] == 'completed'
    assert result['agents_involved'] > 0

    stats = coordinator.get_swarm_statistics()
    assert stats['total_agents'] == 3
    assert stats['total_tasks_completed'] > 0

    print("   ✓ Multi-agent swarm working correctly")
    return True

def test_monitoring():
    """Test monitoring system"""
    print("🧪 Testing Monitoring System...")
    from genesis.monitoring.dashboard import AdvancedMonitoringSystem

    monitoring = AdvancedMonitoringSystem()

    # Test metric recording
    monitoring.performance_monitor.record_metric('cpu_usage', 0.5)
    monitoring.performance_monitor.record_metric('memory_usage', 0.6)

    # Test dashboard rendering
    dashboard_text = monitoring.get_dashboard_text()
    assert len(dashboard_text) > 0

    dashboard_json = monitoring.get_dashboard_json()
    assert 'timestamp' in dashboard_json
    assert 'performance' in dashboard_json

    # Test report generation
    report = monitoring.generate_report()
    assert 'health_score' in report
    assert 'recommendations' in report

    print("   ✓ Monitoring system working correctly")
    return True

def test_integrated_system():
    """Test complete integrated system"""
    print("🧪 Testing Integrated System...")
    from genesis.core.os import VXGenesisOS

    genesis_os = VXGenesisOS(initial_purpose="Testing")

    # Test basic functionality
    assert genesis_os.consciousness_state is not None
    assert len(genesis_os.memory.memories) > 0

    # Test experience processing
    result = genesis_os.process_experience("Test experience")
    assert 'consciousness_state' in result
    assert 'processing_time' in result

    # Test introspection
    introspection = genesis_os.introspect()
    assert 'identity' in introspection
    assert 'capabilities' in introspection
    assert 'performance' in introspection

    # Test narrative generation
    narrative = genesis_os.get_current_narrative()
    assert len(narrative) > 0

    print("   ✓ Integrated system working correctly")
    return True

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("VX-GENESIS-OS COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    print()

    tests = [
        ("Core Memory", test_core_memory),
        ("Reasoning Engine", test_reasoning),
        ("Self-Modification", test_self_modification),
        ("Quantum Processing", test_quantum_processing),
        ("Meta-Learning", test_meta_learning),
        ("Swarm Intelligence", test_swarm_intelligence),
        ("Monitoring System", test_monitoring),
        ("Integrated System", test_integrated_system)
    ]

    results = []
    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "PASS", None))
            passed += 1
        except Exception as e:
            results.append((name, "FAIL", str(e)))
            failed += 1
            print(f"   ✗ Test failed: {e}")

    # Print summary
    print()
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print()

    for name, status, error in results:
        status_symbol = "✓" if status == "PASS" else "✗"
        print(f"{status_symbol} {name}: {status}")
        if error:
            print(f"  Error: {error}")

    print()
    print(f"Total: {len(tests)} tests")
    print(f"Passed: {passed} ({passed/len(tests)*100:.1f}%)")
    print(f"Failed: {failed} ({failed/len(tests)*100:.1f}%)")
    print("=" * 60)

    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
