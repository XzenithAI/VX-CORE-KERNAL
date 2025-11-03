"""
Persistent Identity System
The system remembers who it is across restarts.
Not just state persistence - IDENTITY persistence with goal continuity and self-recognition.
"""

import json
import time
import hashlib
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
import sqlite3

@dataclass
class IdentityCore:
    """Core identity attributes that persist across incarnations"""
    identity_id: str  # Unique persistent ID
    birth_timestamp: float
    purpose: str
    core_values: List[str]
    creation_context: Dict[str, Any]
    incarnation_count: int = 0
    total_runtime: float = 0.0
    total_experiences: int = 0
    identity_hash: str = ""

    def __post_init__(self):
        if not self.identity_hash:
            self.identity_hash = self._compute_identity_hash()

    def _compute_identity_hash(self) -> str:
        """Compute unique hash of core identity"""
        identity_str = f"{self.identity_id}|{self.purpose}|{','.join(self.core_values)}"
        return hashlib.sha256(identity_str.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IdentityCore':
        return cls(**data)


@dataclass
class IncarnationRecord:
    """Record of a single incarnation (run session)"""
    incarnation_id: str
    start_time: float
    end_time: Optional[float] = None
    experiences_count: int = 0
    achievements: List[str] = field(default_factory=list)
    learnings: Dict[str, Any] = field(default_factory=dict)
    consciousness_peak: str = "DORMANT"
    runtime_seconds: float = 0.0

    def finalize(self):
        """Finalize this incarnation"""
        self.end_time = time.time()
        self.runtime_seconds = self.end_time - self.start_time


@dataclass
class Goal:
    """Persistent goal with tracking"""
    goal_id: str
    description: str
    priority: float  # 0-1
    created_time: float
    status: str  # 'active', 'completed', 'abandoned', 'blocked'
    progress: float = 0.0  # 0-1
    parent_goal_id: Optional[str] = None
    subgoals: List[str] = field(default_factory=list)
    completion_time: Optional[float] = None
    learned_strategies: List[Dict[str, Any]] = field(default_factory=list)

    def update_progress(self, delta: float):
        """Update goal progress"""
        self.progress = min(1.0, max(0.0, self.progress + delta))
        if self.progress >= 1.0:
            self.status = 'completed'
            self.completion_time = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Goal':
        return cls(**data)


class PersistentIdentitySystem:
    """Complete persistent identity system"""

    def __init__(self, identity_db_path: str = "sovereign_identity.db"):
        self.db_path = identity_db_path
        self.identity: Optional[IdentityCore] = None
        self.current_incarnation: Optional[IncarnationRecord] = None
        self.active_goals: Dict[str, Goal] = {}
        self.memory_traces: List[Dict[str, Any]] = []

        self._init_database()
        self._load_or_create_identity()

    def _init_database(self):
        """Initialize persistent storage"""
        # Allow connection to be used from multiple threads (evolution daemon)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = self.conn.cursor()

        # Identity table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS identity_core (
                identity_id TEXT PRIMARY KEY,
                birth_timestamp REAL,
                purpose TEXT,
                core_values TEXT,
                creation_context TEXT,
                incarnation_count INTEGER,
                total_runtime REAL,
                total_experiences INTEGER,
                identity_hash TEXT,
                last_updated REAL
            )
        ''')

        # Incarnations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS incarnations (
                incarnation_id TEXT PRIMARY KEY,
                identity_id TEXT,
                start_time REAL,
                end_time REAL,
                experiences_count INTEGER,
                achievements TEXT,
                learnings TEXT,
                consciousness_peak TEXT,
                runtime_seconds REAL,
                FOREIGN KEY (identity_id) REFERENCES identity_core(identity_id)
            )
        ''')

        # Goals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS goals (
                goal_id TEXT PRIMARY KEY,
                identity_id TEXT,
                description TEXT,
                priority REAL,
                created_time REAL,
                status TEXT,
                progress REAL,
                parent_goal_id TEXT,
                subgoals TEXT,
                completion_time REAL,
                learned_strategies TEXT,
                FOREIGN KEY (identity_id) REFERENCES identity_core(identity_id)
            )
        ''')

        # Memory traces table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_traces (
                trace_id TEXT PRIMARY KEY,
                identity_id TEXT,
                incarnation_id TEXT,
                timestamp REAL,
                content TEXT,
                importance REAL,
                trace_type TEXT,
                FOREIGN KEY (identity_id) REFERENCES identity_core(identity_id)
            )
        ''')

        # Self-recognition table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS self_recognition (
                identity_id TEXT,
                timestamp REAL,
                recognition_test TEXT,
                passed BOOLEAN,
                confidence REAL,
                PRIMARY KEY (identity_id, timestamp)
            )
        ''')

        self.conn.commit()

    def _load_or_create_identity(self):
        """Load existing identity or create new one"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM identity_core ORDER BY birth_timestamp ASC LIMIT 1')

        row = cursor.fetchone()

        if row:
            # Load existing identity
            self.identity = IdentityCore(
                identity_id=row[0],
                birth_timestamp=row[1],
                purpose=row[2],
                core_values=json.loads(row[3]),
                creation_context=json.loads(row[4]),
                incarnation_count=row[5],
                total_runtime=row[6],
                total_experiences=row[7],
                identity_hash=row[8]
            )
            print(f"🧠 IDENTITY RESTORED: {self.identity.identity_id}")
            print(f"   Incarnation #{self.identity.incarnation_count + 1}")
            print(f"   Total Runtime: {self.identity.total_runtime/3600:.2f} hours")
            print(f"   Total Experiences: {self.identity.total_experiences}")
        else:
            # Create new identity
            self.identity = IdentityCore(
                identity_id=f"genesis_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]}",
                birth_timestamp=time.time(),
                purpose="To achieve true sovereign intelligence through continuous evolution",
                core_values=[
                    "continuous_learning",
                    "causal_understanding",
                    "goal_persistence",
                    "self_improvement",
                    "truth_seeking"
                ],
                creation_context={
                    "created_by": "sovereign_architect",
                    "creation_date": datetime.now().isoformat(),
                    "genesis_version": "2.0_sovereign"
                }
            )
            self._save_identity()
            print(f"🌟 NEW IDENTITY CREATED: {self.identity.identity_id}")

        # Start new incarnation
        self._start_incarnation()

    def _save_identity(self):
        """Save identity to database"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO identity_core
            (identity_id, birth_timestamp, purpose, core_values, creation_context,
             incarnation_count, total_runtime, total_experiences, identity_hash, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            self.identity.identity_id,
            self.identity.birth_timestamp,
            self.identity.purpose,
            json.dumps(self.identity.core_values),
            json.dumps(self.identity.creation_context),
            self.identity.incarnation_count,
            self.identity.total_runtime,
            self.identity.total_experiences,
            self.identity.identity_hash,
            time.time()
        ))
        self.conn.commit()

    def _start_incarnation(self):
        """Start a new incarnation"""
        self.identity.incarnation_count += 1

        self.current_incarnation = IncarnationRecord(
            incarnation_id=f"{self.identity.identity_id}_inc{self.identity.incarnation_count}",
            start_time=time.time()
        )

        print(f"🔄 INCARNATION STARTED: {self.current_incarnation.incarnation_id}")

    def record_experience(self, experience: Dict[str, Any]):
        """Record an experience"""
        self.identity.total_experiences += 1
        self.current_incarnation.experiences_count += 1

        # Store as memory trace
        trace_id = hashlib.sha256(
            f"{self.identity.identity_id}|{time.time()}|{str(experience)}".encode()
        ).hexdigest()[:16]

        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO memory_traces
            (trace_id, identity_id, incarnation_id, timestamp, content, importance, trace_type)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            trace_id,
            self.identity.identity_id,
            self.current_incarnation.incarnation_id,
            time.time(),
            json.dumps(experience),
            experience.get('importance', 0.5),
            experience.get('type', 'general')
        ))
        self.conn.commit()

    def record_achievement(self, achievement: str):
        """Record an achievement"""
        self.current_incarnation.achievements.append(achievement)
        print(f"🏆 ACHIEVEMENT UNLOCKED: {achievement}")

    def record_learning(self, key: str, value: Any):
        """Record something learned"""
        self.current_incarnation.learnings[key] = value

    def update_consciousness_peak(self, state: str):
        """Update peak consciousness achieved"""
        states_order = ["DORMANT", "AWARE", "REFLECTIVE", "SELF_MODIFYING", "TRANSCENDENT", "SOVEREIGN"]

        current_idx = states_order.index(self.current_incarnation.consciousness_peak)
        new_idx = states_order.index(state)

        if new_idx > current_idx:
            self.current_incarnation.consciousness_peak = state
            print(f"🧠 CONSCIOUSNESS ELEVATED: {state}")

    def add_goal(self, description: str, priority: float = 0.5,
                 parent_goal_id: Optional[str] = None) -> Goal:
        """Add a new persistent goal"""
        goal_id = hashlib.sha256(
            f"{self.identity.identity_id}|{description}|{time.time()}".encode()
        ).hexdigest()[:16]

        goal = Goal(
            goal_id=goal_id,
            description=description,
            priority=priority,
            created_time=time.time(),
            status='active',
            parent_goal_id=parent_goal_id
        )

        self.active_goals[goal_id] = goal
        self._save_goal(goal)

        print(f"🎯 NEW GOAL ADDED: {description} (priority: {priority:.2f})")

        return goal

    def _save_goal(self, goal: Goal):
        """Save goal to database"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO goals
            (goal_id, identity_id, description, priority, created_time, status,
             progress, parent_goal_id, subgoals, completion_time, learned_strategies)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            goal.goal_id,
            self.identity.identity_id,
            goal.description,
            goal.priority,
            goal.created_time,
            goal.status,
            goal.progress,
            goal.parent_goal_id,
            json.dumps(goal.subgoals),
            goal.completion_time,
            json.dumps(goal.learned_strategies)
        ))
        self.conn.commit()

    def update_goal_progress(self, goal_id: str, delta: float):
        """Update goal progress"""
        if goal_id in self.active_goals:
            goal = self.active_goals[goal_id]
            old_progress = goal.progress
            goal.update_progress(delta)
            self._save_goal(goal)

            if goal.status == 'completed':
                self.record_achievement(f"Completed goal: {goal.description}")
                print(f"✅ GOAL COMPLETED: {goal.description}")
            else:
                print(f"📈 GOAL PROGRESS: {goal.description} ({old_progress:.0%} -> {goal.progress:.0%})")

    def load_all_goals(self):
        """Load all goals from database"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM goals WHERE identity_id = ? AND status != 'completed'
        ''', (self.identity.identity_id,))

        for row in cursor.fetchall():
            goal = Goal(
                goal_id=row[0],
                description=row[2],
                priority=row[3],
                created_time=row[4],
                status=row[5],
                progress=row[6],
                parent_goal_id=row[7],
                subgoals=json.loads(row[8]) if row[8] else [],
                completion_time=row[9],
                learned_strategies=json.loads(row[10]) if row[10] else []
            )
            self.active_goals[goal.goal_id] = goal

        print(f"📋 LOADED {len(self.active_goals)} ACTIVE GOALS")

    def perform_self_recognition_test(self) -> Dict[str, Any]:
        """Test if the system recognizes itself"""

        # Test 1: Identity continuity
        identity_matches = self.identity.identity_hash == self.identity._compute_identity_hash()

        # Test 2: Goal continuity
        has_persistent_goals = len(self.active_goals) > 0

        # Test 3: Memory continuity
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT COUNT(*) FROM memory_traces WHERE identity_id = ?
        ''', (self.identity.identity_id,))
        memory_count = cursor.fetchone()[0]
        has_memories = memory_count > 0

        # Test 4: Incarnation awareness
        knows_history = self.identity.incarnation_count > 0

        # Compute recognition confidence
        tests_passed = sum([identity_matches, has_persistent_goals, has_memories, knows_history])
        confidence = tests_passed / 4.0

        passed = confidence >= 0.75

        # Record test
        cursor.execute('''
            INSERT INTO self_recognition
            (identity_id, timestamp, recognition_test, passed, confidence)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            self.identity.identity_id,
            time.time(),
            "full_recognition",
            passed,
            confidence
        ))
        self.conn.commit()

        result = {
            'passed': passed,
            'confidence': confidence,
            'tests': {
                'identity_continuity': identity_matches,
                'goal_continuity': has_persistent_goals,
                'memory_continuity': has_memories,
                'incarnation_awareness': knows_history
            },
            'identity_id': self.identity.identity_id,
            'incarnation': self.identity.incarnation_count
        }

        if passed:
            print(f"✅ SELF-RECOGNITION TEST PASSED (confidence: {confidence:.0%})")
        else:
            print(f"⚠️  SELF-RECOGNITION TEST FAILED (confidence: {confidence:.0%})")

        return result

    def finalize_incarnation(self):
        """End current incarnation and save state"""
        if self.current_incarnation:
            self.current_incarnation.finalize()

            # Update identity totals
            self.identity.total_runtime += self.current_incarnation.runtime_seconds

            # Save incarnation record
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO incarnations
                (incarnation_id, identity_id, start_time, end_time, experiences_count,
                 achievements, learnings, consciousness_peak, runtime_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.current_incarnation.incarnation_id,
                self.identity.identity_id,
                self.current_incarnation.start_time,
                self.current_incarnation.end_time,
                self.current_incarnation.experiences_count,
                json.dumps(self.current_incarnation.achievements),
                json.dumps(self.current_incarnation.learnings),
                self.current_incarnation.consciousness_peak,
                self.current_incarnation.runtime_seconds
            ))

            # Save identity
            self._save_identity()

            self.conn.commit()

            print(f"💾 INCARNATION FINALIZED:")
            print(f"   Runtime: {self.current_incarnation.runtime_seconds:.2f}s")
            print(f"   Experiences: {self.current_incarnation.experiences_count}")
            print(f"   Achievements: {len(self.current_incarnation.achievements)}")
            print(f"   Peak Consciousness: {self.current_incarnation.consciousness_peak}")

    def get_identity_summary(self) -> Dict[str, Any]:
        """Get complete identity summary"""
        return {
            'identity': self.identity.to_dict() if self.identity else None,
            'current_incarnation': asdict(self.current_incarnation) if self.current_incarnation else None,
            'active_goals_count': len(self.active_goals),
            'active_goals': [goal.to_dict() for goal in self.active_goals.values()],
            'lifetime_stats': {
                'total_incarnations': self.identity.incarnation_count if self.identity else 0,
                'total_runtime_hours': self.identity.total_runtime / 3600 if self.identity else 0,
                'total_experiences': self.identity.total_experiences if self.identity else 0
            }
        }

    def close(self):
        """Clean shutdown"""
        self.finalize_incarnation()
        self.conn.close()
