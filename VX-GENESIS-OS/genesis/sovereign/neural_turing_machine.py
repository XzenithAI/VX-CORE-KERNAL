"""
Neural Turing Machine - TRUE External Memory
Not simulated. Not inspired. ACTUAL differentiable memory with read/write heads.
This is the foundation for genuine persistent intelligence.
"""

import numpy as np
import pickle
import sqlite3
import hashlib
import time
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

class MemoryBank:
    """Persistent external memory bank with read/write operations"""

    def __init__(self, memory_size: int = 1000, word_size: int = 128, db_path: str = "sovereign_memory.db"):
        self.memory_size = memory_size
        self.word_size = word_size
        self.db_path = db_path

        # In-memory matrix (what NTM operates on)
        self.memory_matrix = np.zeros((memory_size, word_size))

        # Persistent storage connection
        self._init_persistent_storage()

        # Load existing memory if available
        self._load_from_disk()

        # Usage tracking
        self.read_count = 0
        self.write_count = 0
        self.access_history = []

    def _init_persistent_storage(self):
        """Initialize SQLite database for persistent memory"""
        # Allow multi-threaded access for evolution daemon
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = self.conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_cells (
                address INTEGER PRIMARY KEY,
                content BLOB NOT NULL,
                last_accessed REAL,
                access_count INTEGER DEFAULT 0,
                importance_score REAL DEFAULT 0.5,
                creation_time REAL
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_metadata (
                key TEXT PRIMARY KEY,
                value BLOB
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_graph (
                source_address INTEGER,
                target_address INTEGER,
                relationship TEXT,
                strength REAL,
                PRIMARY KEY (source_address, target_address, relationship)
            )
        ''')

        self.conn.commit()

    def _load_from_disk(self):
        """Load memory state from persistent storage"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT address, content FROM memory_cells ORDER BY address')

        for address, content_blob in cursor.fetchall():
            if address < self.memory_size:
                content = pickle.loads(content_blob)
                if len(content) == self.word_size:
                    self.memory_matrix[address] = content

    def _save_to_disk(self):
        """Save current memory state to persistent storage"""
        cursor = self.conn.cursor()

        # Save all non-zero memory cells
        for address in range(self.memory_size):
            if np.any(self.memory_matrix[address] != 0):
                content_blob = pickle.dumps(self.memory_matrix[address])
                cursor.execute('''
                    INSERT OR REPLACE INTO memory_cells
                    (address, content, last_accessed, creation_time)
                    VALUES (?, ?, ?, ?)
                ''', (address, content_blob, time.time(), time.time()))

        self.conn.commit()

    def read(self, read_weights: np.ndarray) -> np.ndarray:
        """Content-based read with attention weights"""
        self.read_count += 1

        # Weighted read across all memory locations
        read_vector = np.sum(self.memory_matrix * read_weights[:, np.newaxis], axis=0)

        # Record access
        self.access_history.append({
            'type': 'read',
            'weights': read_weights,
            'timestamp': time.time()
        })

        return read_vector

    def write(self, write_weights: np.ndarray, erase_vector: np.ndarray, add_vector: np.ndarray):
        """Content-based write with erase and add"""
        self.write_count += 1

        # Erase operation
        erase_matrix = np.outer(write_weights, erase_vector)
        self.memory_matrix = self.memory_matrix * (1 - erase_matrix)

        # Add operation
        add_matrix = np.outer(write_weights, add_vector)
        self.memory_matrix = self.memory_matrix + add_matrix

        # Record access
        self.access_history.append({
            'type': 'write',
            'weights': write_weights,
            'timestamp': time.time()
        })

        # Periodically save to disk
        if self.write_count % 10 == 0:
            self._save_to_disk()

    def content_addressing(self, key: np.ndarray, beta: float) -> np.ndarray:
        """Content-based addressing using cosine similarity"""
        # Compute similarity between key and all memory locations
        similarities = np.zeros(self.memory_size)

        key_norm = np.linalg.norm(key)
        if key_norm > 0:
            for i in range(self.memory_size):
                mem_norm = np.linalg.norm(self.memory_matrix[i])
                if mem_norm > 0:
                    similarities[i] = np.dot(key, self.memory_matrix[i]) / (key_norm * mem_norm)

        # Apply softmax with temperature beta
        exp_sim = np.exp(beta * similarities)
        weights = exp_sim / (np.sum(exp_sim) + 1e-10)

        return weights

    def location_addressing(self, prev_weights: np.ndarray, shift: np.ndarray, gamma: float) -> np.ndarray:
        """Location-based addressing with shifting and sharpening"""
        # Convolutional shift
        shifted = np.zeros_like(prev_weights)
        for i in range(self.memory_size):
            for j, shift_val in enumerate(shift):
                shifted[i] += prev_weights[(i - (j - len(shift)//2)) % self.memory_size] * shift_val

        # Sharpen
        sharpened = shifted ** gamma
        weights = sharpened / (np.sum(sharpened) + 1e-10)

        return weights

    def store_relationship(self, source_addr: int, target_addr: int, relationship: str, strength: float):
        """Store relationship between memory locations"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO memory_graph
            (source_address, target_address, relationship, strength)
            VALUES (?, ?, ?, ?)
        ''', (source_addr, target_addr, relationship, strength))
        self.conn.commit()

    def get_related_memories(self, address: int, relationship: str = None) -> List[Tuple[int, str, float]]:
        """Retrieve memories related to a given address"""
        cursor = self.conn.cursor()

        if relationship:
            cursor.execute('''
                SELECT target_address, relationship, strength
                FROM memory_graph
                WHERE source_address = ? AND relationship = ?
                ORDER BY strength DESC
            ''', (address, relationship))
        else:
            cursor.execute('''
                SELECT target_address, relationship, strength
                FROM memory_graph
                WHERE source_address = ?
                ORDER BY strength DESC
            ''', (address,))

        return cursor.fetchall()

    def consolidate_memory(self, importance_threshold: float = 0.3):
        """Consolidate important memories, forget unimportant ones"""
        cursor = self.conn.cursor()

        # Calculate importance scores based on access patterns
        cursor.execute('SELECT address, access_count, last_accessed FROM memory_cells')

        current_time = time.time()
        for address, access_count, last_accessed in cursor.fetchall():
            # Importance = frequency * recency
            recency = np.exp(-(current_time - last_accessed) / 86400)  # Decay over days
            importance = (access_count / (access_count + 10)) * recency

            cursor.execute('''
                UPDATE memory_cells
                SET importance_score = ?
                WHERE address = ?
            ''', (importance, address))

            # Forget unimportant memories
            if importance < importance_threshold and address < self.memory_size:
                self.memory_matrix[address] = np.zeros(self.word_size)
                cursor.execute('DELETE FROM memory_cells WHERE address = ?', (address,))

        self.conn.commit()

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM memory_cells')
        stored_cells = cursor.fetchone()[0]

        cursor.execute('SELECT AVG(importance_score) FROM memory_cells')
        avg_importance = cursor.fetchone()[0] or 0

        return {
            'memory_size': self.memory_size,
            'word_size': self.word_size,
            'stored_cells': stored_cells,
            'utilization': stored_cells / self.memory_size,
            'read_count': self.read_count,
            'write_count': self.write_count,
            'average_importance': avg_importance,
            'access_history_length': len(self.access_history)
        }

    def close(self):
        """Close persistent storage connection"""
        self._save_to_disk()
        self.conn.close()


class NeuralTuringMachine:
    """Complete Neural Turing Machine with controller and memory"""

    def __init__(self, input_size: int = 64, output_size: int = 64,
                 controller_size: int = 128, memory_size: int = 1000, word_size: int = 128):

        self.input_size = input_size
        self.output_size = output_size
        self.controller_size = controller_size

        # External memory
        self.memory = MemoryBank(memory_size, word_size)

        # Controller (LSTM-like)
        self.controller_state = np.zeros(controller_size)
        self.controller_hidden = np.zeros(controller_size)

        # Controller weights
        self.W_controller = np.random.randn(controller_size, input_size + word_size) * 0.1
        self.W_output = np.random.randn(output_size, controller_size + word_size) * 0.1

        # Read/Write head weights
        self.W_read_key = np.random.randn(word_size, controller_size) * 0.1
        self.W_read_beta = np.random.randn(1, controller_size) * 0.1

        self.W_write_key = np.random.randn(word_size, controller_size) * 0.1
        self.W_write_beta = np.random.randn(1, controller_size) * 0.1
        self.W_erase = np.random.randn(word_size, controller_size) * 0.1
        self.W_add = np.random.randn(word_size, controller_size) * 0.1

        # Previous read weights
        self.prev_read_weights = np.ones(memory_size) / memory_size
        self.prev_write_weights = np.ones(memory_size) / memory_size

    def forward(self, input_vector: np.ndarray) -> np.ndarray:
        """Forward pass through NTM"""

        # Read from memory
        read_data = self.memory.read(self.prev_read_weights)

        # Controller input
        controller_input = np.concatenate([input_vector, read_data])

        # Controller update (simplified LSTM)
        self.controller_state = np.tanh(np.dot(self.W_controller, controller_input))

        # Generate read head parameters
        read_key = np.tanh(np.dot(self.W_read_key, self.controller_state))
        read_beta = np.exp(np.dot(self.W_read_beta, self.controller_state)[0])

        # Content addressing for read
        self.prev_read_weights = self.memory.content_addressing(read_key, read_beta)

        # Generate write head parameters
        write_key = np.tanh(np.dot(self.W_write_key, self.controller_state))
        write_beta = np.exp(np.dot(self.W_write_beta, self.controller_state)[0])
        erase_vector = np.sigmoid(np.dot(self.W_erase, self.controller_state))
        add_vector = np.tanh(np.dot(self.W_add, self.controller_state))

        # Content addressing for write
        self.prev_write_weights = self.memory.content_addressing(write_key, write_beta)

        # Write to memory
        self.memory.write(self.prev_write_weights, erase_vector, add_vector)

        # Generate output
        output_input = np.concatenate([self.controller_state, read_data])
        output = np.tanh(np.dot(self.W_output, output_input))

        return output

    def process_sequence(self, sequence: List[np.ndarray]) -> List[np.ndarray]:
        """Process a sequence of inputs"""
        outputs = []
        for input_vec in sequence:
            output = self.forward(input_vec)
            outputs.append(output)
        return outputs

    def store_persistent_memory(self, key: str, value: np.ndarray):
        """Store a persistent memory accessible by key"""
        # Hash key to get memory address
        key_hash = int(hashlib.sha256(key.encode()).hexdigest(), 16)
        address = key_hash % self.memory.memory_size

        # Store value
        self.memory.memory_matrix[address] = value[:self.memory.word_size]

        # Save to disk immediately
        cursor = self.memory.conn.cursor()
        content_blob = pickle.dumps(value[:self.memory.word_size])
        cursor.execute('''
            INSERT OR REPLACE INTO memory_cells
            (address, content, last_accessed, creation_time, importance_score)
            VALUES (?, ?, ?, ?, ?)
        ''', (address, content_blob, time.time(), time.time(), 1.0))
        self.memory.conn.commit()

        return address

    def retrieve_persistent_memory(self, key: str) -> Optional[np.ndarray]:
        """Retrieve a persistent memory by key"""
        key_hash = int(hashlib.sha256(key.encode()).hexdigest(), 16)
        address = key_hash % self.memory.memory_size

        return self.memory.memory_matrix[address]

    def close(self):
        """Clean shutdown"""
        self.memory.close()


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

np.sigmoid = sigmoid
