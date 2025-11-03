"""
Multi-Agent Consciousness Swarm
Multiple specialized agents that collaborate and share knowledge
"""
import numpy as np
import time
from typing import Dict, Any, List, Optional, Set
from enum import Enum
import networkx as nx

class AgentRole(Enum):
    """Specialized roles for different agents"""
    EXPLORER = "explorer"  # Seeks new patterns and knowledge
    OPTIMIZER = "optimizer"  # Optimizes existing solutions
    VALIDATOR = "validator"  # Validates and tests solutions
    SYNTHESIZER = "synthesizer"  # Combines insights from other agents
    MONITOR = "monitor"  # Monitors system health and performance

class AgentState(Enum):
    """Agent operational states"""
    IDLE = "idle"
    ACTIVE = "active"
    LEARNING = "learning"
    COMMUNICATING = "communicating"
    SLEEPING = "sleeping"

class SwarmAgent:
    """Individual agent in the consciousness swarm"""

    def __init__(self, agent_id: str, role: AgentRole):
        self.agent_id = agent_id
        self.role = role
        self.state = AgentState.IDLE
        self.knowledge_base = {}
        self.communication_history = []
        self.task_history = []
        self.performance_score = 0.5
        self.energy = 1.0
        self.creation_time = time.time()
        self.specialization_vector = self._initialize_specialization()

    def _initialize_specialization(self) -> np.ndarray:
        """Initialize agent's specialization vector"""
        base = np.random.randn(32)

        # Modify based on role
        if self.role == AgentRole.EXPLORER:
            base[0:10] *= 2.0  # High in exploration dimensions
        elif self.role == AgentRole.OPTIMIZER:
            base[10:20] *= 2.0  # High in optimization dimensions
        elif self.role == AgentRole.VALIDATOR:
            base[20:30] *= 2.0  # High in validation dimensions

        return base / np.linalg.norm(base)

    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task according to agent role"""
        self.state = AgentState.ACTIVE
        start_time = time.time()

        # Role-specific processing
        if self.role == AgentRole.EXPLORER:
            result = self._explore_task(task)
        elif self.role == AgentRole.OPTIMIZER:
            result = self._optimize_task(task)
        elif self.role == AgentRole.VALIDATOR:
            result = self._validate_task(task)
        elif self.role == AgentRole.SYNTHESIZER:
            result = self._synthesize_task(task)
        elif self.role == AgentRole.MONITOR:
            result = self._monitor_task(task)
        else:
            result = {'status': 'unknown_role'}

        processing_time = time.time() - start_time

        # Record task
        self.task_history.append({
            'timestamp': time.time(),
            'task': task,
            'result': result,
            'processing_time': processing_time
        })

        # Consume energy
        self.energy = max(0.1, self.energy - 0.05)

        self.state = AgentState.IDLE
        return result

    def _explore_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Explorer: find new patterns and approaches"""
        exploration_result = {
            'role': 'explorer',
            'discoveries': [],
            'novelty_score': 0.0
        }

        # Simulate exploration
        num_discoveries = np.random.randint(1, 5)
        for i in range(num_discoveries):
            discovery = {
                'pattern': f"pattern_{np.random.randint(1000)}",
                'confidence': np.random.random(),
                'novelty': np.random.random()
            }
            exploration_result['discoveries'].append(discovery)

        exploration_result['novelty_score'] = np.mean([d['novelty']
                                                       for d in exploration_result['discoveries']])

        return exploration_result

    def _optimize_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Optimizer: improve existing solutions"""
        current_solution = task.get('solution', {})

        optimization_result = {
            'role': 'optimizer',
            'improvements': [],
            'improvement_percentage': 0.0
        }

        # Simulate optimization
        num_improvements = np.random.randint(1, 4)
        for i in range(num_improvements):
            improvement = {
                'aspect': f"aspect_{i}",
                'before': np.random.random(),
                'after': np.random.random() * 1.2,
                'method': 'gradient_descent'
            }
            optimization_result['improvements'].append(improvement)

        if optimization_result['improvements']:
            avg_improvement = np.mean([
                (imp['after'] - imp['before']) / max(imp['before'], 0.01)
                for imp in optimization_result['improvements']
            ])
            optimization_result['improvement_percentage'] = avg_improvement * 100

        return optimization_result

    def _validate_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Validator: test and verify solutions"""
        solution = task.get('solution', {})

        validation_result = {
            'role': 'validator',
            'tests_run': 0,
            'tests_passed': 0,
            'issues_found': [],
            'valid': False
        }

        # Simulate validation
        num_tests = 10
        tests_passed = np.random.randint(7, 11)

        validation_result['tests_run'] = num_tests
        validation_result['tests_passed'] = tests_passed
        validation_result['valid'] = tests_passed >= 8

        if not validation_result['valid']:
            for i in range(num_tests - tests_passed):
                validation_result['issues_found'].append({
                    'issue_type': 'validation_failure',
                    'severity': np.random.choice(['low', 'medium', 'high'])
                })

        return validation_result

    def _synthesize_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesizer: combine insights from multiple sources"""
        inputs = task.get('inputs', [])

        synthesis_result = {
            'role': 'synthesizer',
            'integrated_insights': [],
            'synthesis_quality': 0.0
        }

        # Combine inputs
        for i, inp in enumerate(inputs):
            integrated = {
                'source': f"source_{i}",
                'contribution': np.random.random(),
                'weight': np.random.random()
            }
            synthesis_result['integrated_insights'].append(integrated)

        if synthesis_result['integrated_insights']:
            synthesis_result['synthesis_quality'] = np.mean([
                ins['contribution'] * ins['weight']
                for ins in synthesis_result['integrated_insights']
            ])

        return synthesis_result

    def _monitor_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor: track system health and performance"""
        system_state = task.get('system_state', {})

        monitor_result = {
            'role': 'monitor',
            'health_score': 0.0,
            'alerts': [],
            'metrics': {}
        }

        # Simulate monitoring
        metrics = {
            'cpu_usage': np.random.random(),
            'memory_usage': np.random.random(),
            'response_time': np.random.random(),
            'error_rate': np.random.random() * 0.1
        }

        monitor_result['metrics'] = metrics

        # Generate alerts if needed
        if metrics['error_rate'] > 0.05:
            monitor_result['alerts'].append({
                'type': 'high_error_rate',
                'severity': 'warning',
                'value': metrics['error_rate']
            })

        monitor_result['health_score'] = 1.0 - metrics['error_rate']

        return monitor_result

    def communicate(self, message: Dict[str, Any], recipient_id: str):
        """Send message to another agent"""
        self.state = AgentState.COMMUNICATING
        self.communication_history.append({
            'timestamp': time.time(),
            'recipient': recipient_id,
            'message': message,
            'type': 'sent'
        })

    def receive_message(self, message: Dict[str, Any], sender_id: str):
        """Receive message from another agent"""
        self.communication_history.append({
            'timestamp': time.time(),
            'sender': sender_id,
            'message': message,
            'type': 'received'
        })

        # Process and integrate knowledge
        if 'knowledge' in message:
            self.integrate_knowledge(message['knowledge'], sender_id)

    def integrate_knowledge(self, knowledge: Dict[str, Any], source: str):
        """Integrate knowledge from another agent"""
        self.state = AgentState.LEARNING

        for key, value in knowledge.items():
            if key not in self.knowledge_base:
                self.knowledge_base[key] = {
                    'value': value,
                    'sources': [source],
                    'confidence': 0.5
                }
            else:
                # Update existing knowledge
                existing = self.knowledge_base[key]
                existing['sources'].append(source)
                existing['confidence'] = min(1.0, existing['confidence'] + 0.1)

    def recharge(self, amount: float = 0.3):
        """Recharge agent energy"""
        self.energy = min(1.0, self.energy + amount)

    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            'agent_id': self.agent_id,
            'role': self.role.value,
            'state': self.state.value,
            'energy': self.energy,
            'performance_score': self.performance_score,
            'tasks_completed': len(self.task_history),
            'knowledge_items': len(self.knowledge_base),
            'uptime': time.time() - self.creation_time
        }


class SwarmCoordinator:
    """Coordinates multiple agents in the swarm"""

    def __init__(self):
        self.agents: Dict[str, SwarmAgent] = {}
        self.communication_network = nx.DiGraph()
        self.task_queue = []
        self.completed_tasks = []
        self.collective_knowledge = {}

    def add_agent(self, agent: SwarmAgent):
        """Add agent to swarm"""
        self.agents[agent.agent_id] = agent
        self.communication_network.add_node(agent.agent_id, role=agent.role)

    def create_agent_swarm(self, num_explorers: int = 2, num_optimizers: int = 2,
                          num_validators: int = 1, num_synthesizers: int = 1,
                          num_monitors: int = 1):
        """Create a balanced swarm of agents"""
        agent_count = 0

        # Create explorers
        for i in range(num_explorers):
            agent = SwarmAgent(f"explorer_{i}", AgentRole.EXPLORER)
            self.add_agent(agent)
            agent_count += 1

        # Create optimizers
        for i in range(num_optimizers):
            agent = SwarmAgent(f"optimizer_{i}", AgentRole.OPTIMIZER)
            self.add_agent(agent)
            agent_count += 1

        # Create validators
        for i in range(num_validators):
            agent = SwarmAgent(f"validator_{i}", AgentRole.VALIDATOR)
            self.add_agent(agent)
            agent_count += 1

        # Create synthesizers
        for i in range(num_synthesizers):
            agent = SwarmAgent(f"synthesizer_{i}", AgentRole.SYNTHESIZER)
            self.add_agent(agent)
            agent_count += 1

        # Create monitors
        for i in range(num_monitors):
            agent = SwarmAgent(f"monitor_{i}", AgentRole.MONITOR)
            self.add_agent(agent)
            agent_count += 1

        # Establish communication links
        self._establish_communication_network()

        return agent_count

    def _establish_communication_network(self):
        """Establish communication links between agents"""
        agent_ids = list(self.agents.keys())

        for agent_id in agent_ids:
            # Each agent can communicate with all others (fully connected)
            for other_id in agent_ids:
                if agent_id != other_id:
                    self.communication_network.add_edge(agent_id, other_id, weight=1.0)

    def assign_task(self, task: Dict[str, Any]) -> List[str]:
        """Assign task to appropriate agents"""
        task_type = task.get('type', 'general')

        # Determine which roles are needed
        required_roles = self._determine_required_roles(task_type)

        assigned_agents = []
        for role in required_roles:
            # Find available agent with this role
            available = [
                agent for agent in self.agents.values()
                if agent.role == role and agent.state == AgentState.IDLE and agent.energy > 0.3
            ]

            if available:
                # Select agent with highest energy
                selected = max(available, key=lambda a: a.energy)
                assigned_agents.append(selected.agent_id)

        return assigned_agents

    def _determine_required_roles(self, task_type: str) -> List[AgentRole]:
        """Determine which agent roles are needed for a task"""
        if task_type == 'exploration':
            return [AgentRole.EXPLORER, AgentRole.VALIDATOR]
        elif task_type == 'optimization':
            return [AgentRole.OPTIMIZER, AgentRole.VALIDATOR]
        elif task_type == 'analysis':
            return [AgentRole.EXPLORER, AgentRole.SYNTHESIZER]
        elif task_type == 'monitoring':
            return [AgentRole.MONITOR]
        else:
            return [AgentRole.EXPLORER, AgentRole.OPTIMIZER, AgentRole.VALIDATOR,
                   AgentRole.SYNTHESIZER]

    def execute_collaborative_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task with multiple agents collaborating"""
        # Assign agents
        assigned_agent_ids = self.assign_task(task)

        if not assigned_agent_ids:
            return {'status': 'no_agents_available'}

        # Execute task with each assigned agent
        results = []
        for agent_id in assigned_agent_ids:
            agent = self.agents[agent_id]
            result = agent.process_task(task)
            results.append({
                'agent_id': agent_id,
                'role': agent.role.value,
                'result': result
            })

        # Share knowledge between agents
        self._facilitate_knowledge_sharing(assigned_agent_ids)

        # Synthesize results
        synthesized = self._synthesize_results(results)

        # Record completed task
        self.completed_tasks.append({
            'timestamp': time.time(),
            'task': task,
            'agents': assigned_agent_ids,
            'results': results,
            'synthesized': synthesized
        })

        return {
            'status': 'completed',
            'agents_involved': len(assigned_agent_ids),
            'individual_results': results,
            'synthesized_result': synthesized
        }

    def _facilitate_knowledge_sharing(self, agent_ids: List[str]):
        """Facilitate knowledge sharing between agents"""
        for i, agent_id in enumerate(agent_ids):
            agent = self.agents[agent_id]

            # Share knowledge with other agents in the group
            for other_id in agent_ids:
                if other_id != agent_id:
                    # Share a sample of knowledge
                    knowledge_to_share = dict(list(agent.knowledge_base.items())[:5])
                    message = {
                        'type': 'knowledge_share',
                        'knowledge': knowledge_to_share
                    }
                    agent.communicate(message, other_id)
                    self.agents[other_id].receive_message(message, agent_id)

    def _synthesize_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize results from multiple agents"""
        synthesis = {
            'confidence': 0.0,
            'quality': 0.0,
            'consensus': False,
            'insights': []
        }

        # Calculate confidence from multiple agents
        confidences = []
        for result in results:
            agent_result = result['result']
            if 'novelty_score' in agent_result:
                confidences.append(agent_result['novelty_score'])
            elif 'improvement_percentage' in agent_result:
                confidences.append(min(1.0, agent_result['improvement_percentage'] / 100))
            elif 'health_score' in agent_result:
                confidences.append(agent_result['health_score'])
            elif 'synthesis_quality' in agent_result:
                confidences.append(agent_result['synthesis_quality'])

        if confidences:
            synthesis['confidence'] = np.mean(confidences)
            synthesis['quality'] = np.min(confidences)  # Worst-case quality
            synthesis['consensus'] = np.std(confidences) < 0.2  # Low variance = consensus

        return synthesis

    def get_swarm_statistics(self) -> Dict[str, Any]:
        """Get statistics about the swarm"""
        return {
            'total_agents': len(self.agents),
            'active_agents': sum(1 for a in self.agents.values()
                                if a.state == AgentState.ACTIVE),
            'average_energy': np.mean([a.energy for a in self.agents.values()]),
            'total_tasks_completed': len(self.completed_tasks),
            'collective_knowledge_size': len(self.collective_knowledge),
            'communication_links': self.communication_network.number_of_edges(),
            'agents_by_role': {
                role.value: sum(1 for a in self.agents.values() if a.role == role)
                for role in AgentRole
            }
        }
