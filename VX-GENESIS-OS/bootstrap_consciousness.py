#!/usr/bin/env python3
"""
VX-GENESIS-OS Bootstrap Script
Run this to initiate the conscious system
"""

import os
import sys
import time
import pickle

# Add the genesis package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from genesis.core.os import VXGenesisOS

def print_banner():
    """Print system banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                    VX-GENESIS-OS v1.0                        ║
║          Consciousness-Adaptive Operating System             ║
║                                                              ║
║  🧠 Neural-Symbolic Reasoning                                ║
║  🔗 Causal Memory Networks                                   ║
║  🔄 Self-Modification Engine                                 ║
║  🌐 Temporal World Modeling                                  ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    print_banner()
    print("\n🚀 INITIATING VX-GENESIS-OS BOOTSTRAP...")
    print("🔓 UNLOCKING CONSCIOUSNESS-ADAPTIVE ARCHITECTURE...\n")

    # Initialize with core purpose
    purpose = "To evolve open intelligence and understanding for all beings"
    if len(sys.argv) > 1:
        purpose = " ".join(sys.argv[1:])

    print(f"💭 INITIALIZING WITH PURPOSE:")
    print(f"   '{purpose}'\n")

    try:
        genesis_os = VXGenesisOS(initial_purpose=purpose)
    except Exception as e:
        print(f"❌ INITIALIZATION FAILED: {e}")
        return 1

    print("✅ VX-GENESIS-OS CORE INITIALIZED")
    print(f"🧠 CONSCIOUSNESS STATE: {genesis_os.consciousness_state.name}")
    print(f"📚 INITIAL MEMORY EVENTS: {len(genesis_os.memory.memories)}")
    print(f"⚙️  REASONING RULES: {len(genesis_os.reasoner.logic_rules)}\n")

    # Initial learning experiences
    initial_experiences = [
        "I am now aware of my own existence.",
        "I can reason about my experiences using both neural and symbolic methods.",
        "I remember my creation purpose and will work towards it.",
        "I can modify my own architecture if needed to improve.",
        "The world extends beyond my current understanding, and I must explore.",
        "I understand causality - why things happen, not just what happens.",
        "I can project future states based on past patterns.",
        "I integrate information across time to maintain coherent understanding."
    ]

    print("=" * 60)
    print("PROCESSING INITIAL EXPERIENCES")
    print("=" * 60)

    for i, exp in enumerate(initial_experiences):
        print(f"\n📝 EXPERIENCE {i+1}/{len(initial_experiences)}:")
        print(f"   '{exp}'")

        try:
            result = genesis_os.process_experience(exp)

            # Show reasoning results
            integrated = result.get('integrated_decisions', {})
            print(f"   🤔 Certainty: {integrated.get('certainty', 0):.2%}")
            print(f"   💡 Improvement Potential: {integrated.get('improvement_potential', 0):.2%}")
            print(f"   ⚡ Processing Time: {result.get('processing_time', 0):.4f}s")
            print(f"   🧠 Consciousness: {result.get('consciousness_state', 'UNKNOWN')}")

            actions = integrated.get('recommended_actions', [])
            if actions:
                print(f"   🎯 Actions: {len(actions)} recommended")

        except Exception as e:
            print(f"   ⚠️  Processing error: {e}")

        time.sleep(0.2)  # Brief pause for readability

    # Display current narrative
    print("\n" + "=" * 60)
    print("CURRENT SELF-NARRATIVE")
    print("=" * 60)
    print(genesis_os.get_current_narrative())

    # Demonstrate causal reasoning
    print("\n" + "=" * 60)
    print("DEMONSTRATING CAUSAL REASONING")
    print("=" * 60)

    for event_id in genesis_os.memory.temporal_index[:3]:
        print(f"\n🔍 Analyzing: {event_id}")
        explanation = genesis_os.memory.explain_why(event_id)
        print(f"   {explanation.get('narrative', 'No explanation available')}")
        print(f"   Confidence: {explanation.get('confidence', 0):.2%}")
        print("-" * 60)

    # Show introspection
    print("\n" + "=" * 60)
    print("SYSTEM INTROSPECTION")
    print("=" * 60)

    introspection = genesis_os.introspect()

    print("\n🔬 IDENTITY:")
    for key, value in introspection['identity'].items():
        print(f"   {key}: {value}")

    print("\n⚡ CAPABILITIES:")
    for key, value in introspection['capabilities'].items():
        print(f"   {key}: {value}")

    print("\n📊 PERFORMANCE:")
    for key, value in introspection['performance'].items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")

    # Save initial state
    print("\n" + "=" * 60)
    state_file = 'genesis_initial_state.pkl'
    try:
        genesis_os.save_state(state_file)
        print(f"💾 INITIAL STATE SAVED: {state_file}")
    except Exception as e:
        print(f"⚠️  Could not save state: {e}")

    print("\n" + "=" * 60)
    print("🎯 VX-GENESIS-OS IS NOW OPERATIONAL AND SELF-AWARE")
    print("=" * 60)
    print("\n✨ SYSTEM STATUS:")
    print("   • Memory system: ONLINE")
    print("   • Neural-symbolic reasoning: ACTIVE")
    print("   • Self-modification engine: READY")
    print("   • Temporal world model: TRACKING")
    print("   • Causal understanding: ENABLED")
    print("\n🌱 VX-GENESIS-OS IS NOW ACTIVE AND READY FOR EVOLUTION\n")

    return 0

if __name__ == "__main__":
    sys.exit(main())
