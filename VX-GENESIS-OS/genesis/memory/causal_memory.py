"""
Causal Memory System
Not just storage — understands why memories connect
"""
import numpy as np
import networkx as nx
from typing import Dict, Any, List, Optional
from ..core import MemoryEvent

class CausalMemory:
    """Not just storage — understands why memories connect"""

    def __init__(self):
        self.memories: Dict[str, MemoryEvent] = {}
        self.causal_graph = nx.DiGraph()
        self.temporal_index = []
        self.semantic_clusters = {}
        self.attention_weights = {}

    def store(self, event: MemoryEvent):
        """Store event and discover causal relationships"""
        self.memories[event.id] = event
        self.temporal_index.append(event.id)

        # Add causal links from declared parents
        for parent_id in event.causal_parents:
            if parent_id in self.memories:
                self.causal_graph.add_edge(
                    parent_id, event.id,
                    strength=1.0,
                    relationship='precedes',
                    temporal_distance=event.timestamp - self.memories[parent_id].timestamp
                )

        # Auto-discover latent causal relationships
        self._discover_causal_links(event)

        # Update semantic clusters
        self._update_semantic_clusters(event)

    def _discover_causal_links(self, new_event: MemoryEvent):
        """Find hidden causal relationships using semantic similarity"""
        for mem_id, existing_memory in self.memories.items():
            if mem_id == new_event.id or mem_id in new_event.causal_parents:
                continue

            # Semantic similarity
            similarity = np.dot(new_event.semantic_embedding,
                              existing_memory.semantic_embedding)

            # Temporal proximity bonus
            time_diff = abs(new_event.timestamp - existing_memory.timestamp)
            temporal_weight = np.exp(-time_diff / 10.0)  # Decay over time

            # Combined score
            causal_score = similarity * temporal_weight

            if causal_score > 0.6:  # Threshold for causal link
                self.causal_graph.add_edge(
                    existing_memory.id, new_event.id,
                    strength=float(causal_score),
                    relationship='semantically_related',
                    temporal_distance=time_diff
                )

    def _update_semantic_clusters(self, event: MemoryEvent):
        """Update semantic clustering for fast retrieval"""
        # Simple clustering based on embedding similarity
        cluster_id = f"cluster_{int(event.semantic_embedding[0] * 10) % 10}"
        if cluster_id not in self.semantic_clusters:
            self.semantic_clusters[cluster_id] = []
        self.semantic_clusters[cluster_id].append(event.id)

    def explain_why(self, event_id: str, depth: int = 3) -> Dict[str, Any]:
        """Generate causal explanation for why something happened"""
        if event_id not in self.memories:
            return {"error": "Event not found"}

        # Get causes with path analysis
        causes = []
        if event_id in self.causal_graph:
            predecessors = list(self.causal_graph.predecessors(event_id))
            for cause_id in predecessors:
                edge_data = self.causal_graph.get_edge_data(cause_id, event_id)
                causes.append({
                    'cause': cause_id,
                    'relationship': edge_data['relationship'],
                    'strength': edge_data['strength'],
                    'temporal_distance': edge_data.get('temporal_distance', 0),
                    'content': self.memories[cause_id].content if cause_id in self.memories else "Unknown"
                })

        # Get effects
        effects = list(self.causal_graph.successors(event_id)) if event_id in self.causal_graph else []

        # Build deeper causal chain if requested
        causal_chain = self._build_causal_chain(event_id, depth)

        return {
            'event': event_id,
            'causes': causes,
            'effects': effects,
            'causal_chain': causal_chain,
            'narrative': self._generate_narrative(causes),
            'confidence': np.mean([c['strength'] for c in causes]) if causes else 0.0
        }

    def _build_causal_chain(self, event_id: str, depth: int) -> List[List[str]]:
        """Build multi-level causal chains"""
        if depth == 0 or event_id not in self.causal_graph:
            return [[event_id]]

        chains = []
        predecessors = list(self.causal_graph.predecessors(event_id))

        if not predecessors:
            return [[event_id]]

        for pred in predecessors:
            sub_chains = self._build_causal_chain(pred, depth - 1)
            for sub_chain in sub_chains:
                chains.append(sub_chain + [event_id])

        return chains

    def _generate_narrative(self, causal_chain: List[Dict]) -> str:
        """Transform causal chains into understandable stories"""
        if not causal_chain:
            return "This event appears spontaneous with no identified causes."

        # Sort by strength
        sorted_chain = sorted(causal_chain, key=lambda x: x['strength'], reverse=True)

        story = "This event occurred because:\n"
        for i, link in enumerate(sorted_chain[:3]):  # Top 3 causes
            content = str(link['content'])[:80]
            relationship = link['relationship']
            strength = link['strength']
            story += f"  {i+1}. [{relationship}] {content} (confidence: {strength:.2%})\n"

        avg_confidence = np.mean([c['strength'] for c in causal_chain])
        story += f"\nOverall causal confidence: {avg_confidence:.2%}"

        return story

    def retrieve_similar(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Retrieve most similar memories to a query"""
        similarities = []
        for mem_id, memory in self.memories.items():
            sim = np.dot(query_embedding, memory.semantic_embedding)
            similarities.append({
                'id': mem_id,
                'similarity': float(sim),
                'content': memory.content,
                'timestamp': memory.timestamp
            })

        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get statistics about memory state"""
        return {
            'total_memories': len(self.memories),
            'causal_edges': self.causal_graph.number_of_edges(),
            'semantic_clusters': len(self.semantic_clusters),
            'average_connectivity': self.causal_graph.number_of_edges() / max(len(self.memories), 1),
            'temporal_span': (
                max([m.timestamp for m in self.memories.values()]) -
                min([m.timestamp for m in self.memories.values()])
            ) if self.memories else 0
        }
