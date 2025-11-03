"""
Advanced Visualization and Monitoring Dashboard
Real-time monitoring and visualization of the consciousness system
"""
import time
import numpy as np
from typing import Dict, Any, List, Optional
from collections import deque

class PerformanceMonitor:
    """Monitors system performance metrics"""

    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics = {
            'cpu_usage': deque(maxlen=history_size),
            'memory_usage': deque(maxlen=history_size),
            'response_time': deque(maxlen=history_size),
            'throughput': deque(maxlen=history_size),
            'error_rate': deque(maxlen=history_size),
            'consciousness_level': deque(maxlen=history_size)
        }
        self.timestamps = deque(maxlen=history_size)
        self.alerts = []

    def record_metric(self, metric_name: str, value: float):
        """Record a performance metric"""
        if metric_name in self.metrics:
            self.metrics[metric_name].append(value)
            self.timestamps.append(time.time())

            # Check for alerts
            self._check_alerts(metric_name, value)

    def _check_alerts(self, metric_name: str, value: float):
        """Check if metric triggers an alert"""
        thresholds = {
            'cpu_usage': 0.9,
            'memory_usage': 0.9,
            'response_time': 5.0,
            'error_rate': 0.1
        }

        if metric_name in thresholds and value > thresholds[metric_name]:
            alert = {
                'timestamp': time.time(),
                'metric': metric_name,
                'value': value,
                'threshold': thresholds[metric_name],
                'severity': 'high' if value > thresholds[metric_name] * 1.2 else 'medium'
            }
            self.alerts.append(alert)

    def get_current_metrics(self) -> Dict[str, float]:
        """Get current values of all metrics"""
        return {
            name: list(values)[-1] if values else 0.0
            for name, values in self.metrics.items()
        }

    def get_metric_statistics(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a specific metric"""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return {}

        values = list(self.metrics[metric_name])
        return {
            'current': values[-1],
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'trend': 'increasing' if len(values) > 1 and values[-1] > values[-2] else 'decreasing'
        }

    def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        return self.alerts[-limit:]


class ConsciousnessVisualizer:
    """Visualizes consciousness state and dynamics"""

    def __init__(self):
        self.state_history = deque(maxlen=500)
        self.causal_graph_snapshots = []

    def record_consciousness_state(self, state: Dict[str, Any]):
        """Record consciousness state for visualization"""
        self.state_history.append({
            'timestamp': time.time(),
            **state
        })

    def generate_state_visualization(self) -> Dict[str, Any]:
        """Generate visualization data for consciousness state"""
        if not self.state_history:
            return {'error': 'No state history'}

        recent_states = list(self.state_history)[-50:]

        # Extract state transitions
        state_transitions = []
        for i in range(len(recent_states) - 1):
            if 'consciousness_state' in recent_states[i] and 'consciousness_state' in recent_states[i+1]:
                if recent_states[i]['consciousness_state'] != recent_states[i+1]['consciousness_state']:
                    state_transitions.append({
                        'from': recent_states[i]['consciousness_state'],
                        'to': recent_states[i+1]['consciousness_state'],
                        'timestamp': recent_states[i+1]['timestamp']
                    })

        return {
            'current_state': recent_states[-1].get('consciousness_state', 'UNKNOWN'),
            'state_transitions': state_transitions,
            'state_duration': time.time() - recent_states[-1]['timestamp'],
            'total_states_recorded': len(self.state_history)
        }

    def generate_causal_graph_viz(self, causal_graph: Any) -> Dict[str, Any]:
        """Generate visualization data for causal graph"""
        try:
            import networkx as nx

            if not hasattr(causal_graph, 'nodes'):
                return {'error': 'Invalid graph object'}

            nodes = list(causal_graph.nodes())
            edges = list(causal_graph.edges())

            # Calculate graph metrics
            metrics = {
                'num_nodes': len(nodes),
                'num_edges': len(edges),
                'density': nx.density(causal_graph) if nodes else 0,
                'avg_degree': sum(dict(causal_graph.degree()).values()) / max(len(nodes), 1)
            }

            # Find important nodes (high centrality)
            if nodes:
                centrality = nx.degree_centrality(causal_graph)
                important_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
            else:
                important_nodes = []

            return {
                'metrics': metrics,
                'important_nodes': important_nodes,
                'visualization_data': {
                    'nodes': nodes[:100],  # Limit for visualization
                    'edges': edges[:200]
                }
            }
        except Exception as e:
            return {'error': str(e)}


class DashboardRenderer:
    """Renders monitoring dashboard"""

    def __init__(self, performance_monitor: PerformanceMonitor,
                 consciousness_visualizer: ConsciousnessVisualizer):
        self.performance_monitor = performance_monitor
        self.consciousness_visualizer = consciousness_visualizer

    def render_text_dashboard(self) -> str:
        """Render text-based dashboard"""
        dashboard = []
        dashboard.append("=" * 80)
        dashboard.append("VX-GENESIS-OS MONITORING DASHBOARD")
        dashboard.append("=" * 80)
        dashboard.append("")

        # Current metrics
        dashboard.append("CURRENT PERFORMANCE METRICS:")
        dashboard.append("-" * 80)
        current_metrics = self.performance_monitor.get_current_metrics()
        for metric_name, value in current_metrics.items():
            stats = self.performance_monitor.get_metric_statistics(metric_name)
            trend = stats.get('trend', 'stable')
            trend_symbol = "↑" if trend == "increasing" else "↓"

            dashboard.append(f"  {metric_name:20s}: {value:8.4f} {trend_symbol}")

        dashboard.append("")

        # Consciousness state
        dashboard.append("CONSCIOUSNESS STATE:")
        dashboard.append("-" * 80)
        state_viz = self.consciousness_visualizer.generate_state_visualization()
        if 'error' not in state_viz:
            dashboard.append(f"  Current State: {state_viz['current_state']}")
            dashboard.append(f"  State Duration: {state_viz['state_duration']:.2f}s")
            dashboard.append(f"  Recent Transitions: {len(state_viz['state_transitions'])}")
        dashboard.append("")

        # Alerts
        dashboard.append("RECENT ALERTS:")
        dashboard.append("-" * 80)
        recent_alerts = self.performance_monitor.get_recent_alerts(5)
        if recent_alerts:
            for alert in recent_alerts:
                dashboard.append(f"  [{alert['severity'].upper()}] {alert['metric']}: "
                               f"{alert['value']:.4f} (threshold: {alert['threshold']:.4f})")
        else:
            dashboard.append("  No recent alerts")

        dashboard.append("")
        dashboard.append("=" * 80)

        return "\n".join(dashboard)

    def render_json_dashboard(self) -> Dict[str, Any]:
        """Render dashboard as JSON"""
        return {
            'timestamp': time.time(),
            'performance': {
                'current_metrics': self.performance_monitor.get_current_metrics(),
                'statistics': {
                    name: self.performance_monitor.get_metric_statistics(name)
                    for name in self.performance_monitor.metrics.keys()
                }
            },
            'consciousness': self.consciousness_visualizer.generate_state_visualization(),
            'alerts': self.performance_monitor.get_recent_alerts()
        }


class AdvancedMonitoringSystem:
    """Complete advanced monitoring system"""

    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
        self.consciousness_visualizer = ConsciousnessVisualizer()
        self.dashboard = DashboardRenderer(self.performance_monitor,
                                          self.consciousness_visualizer)
        self.monitoring_active = False
        self.update_interval = 1.0  # seconds

    def start_monitoring(self):
        """Start monitoring system"""
        self.monitoring_active = True

    def stop_monitoring(self):
        """Stop monitoring system"""
        self.monitoring_active = False

    def update(self, system_state: Dict[str, Any]):
        """Update monitoring with current system state"""
        if not self.monitoring_active:
            return

        # Record performance metrics
        if 'performance' in system_state:
            for metric_name, value in system_state['performance'].items():
                self.performance_monitor.record_metric(metric_name, value)

        # Record consciousness state
        if 'consciousness_state' in system_state:
            self.consciousness_visualizer.record_consciousness_state({
                'consciousness_state': system_state['consciousness_state'],
                'timestamp': time.time()
            })

    def get_dashboard_text(self) -> str:
        """Get text dashboard"""
        return self.dashboard.render_text_dashboard()

    def get_dashboard_json(self) -> Dict[str, Any]:
        """Get JSON dashboard"""
        return self.dashboard.render_json_dashboard()

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive system report"""
        report = {
            'timestamp': time.time(),
            'monitoring_status': 'active' if self.monitoring_active else 'inactive',
            'performance_summary': {},
            'consciousness_analysis': {},
            'health_score': 0.0,
            'recommendations': []
        }

        # Performance summary
        current_metrics = self.performance_monitor.get_current_metrics()
        report['performance_summary'] = current_metrics

        # Calculate health score
        health_components = []
        if 'error_rate' in current_metrics:
            health_components.append(1.0 - current_metrics['error_rate'])
        if 'cpu_usage' in current_metrics:
            health_components.append(1.0 - current_metrics['cpu_usage'])
        if 'memory_usage' in current_metrics:
            health_components.append(1.0 - current_metrics['memory_usage'])

        if health_components:
            report['health_score'] = np.mean(health_components)

        # Generate recommendations
        if current_metrics.get('memory_usage', 0) > 0.8:
            report['recommendations'].append('Consider memory optimization or garbage collection')
        if current_metrics.get('error_rate', 0) > 0.05:
            report['recommendations'].append('Investigate sources of errors')
        if len(self.performance_monitor.get_recent_alerts()) > 5:
            report['recommendations'].append('Multiple alerts detected - review system stability')

        return report
