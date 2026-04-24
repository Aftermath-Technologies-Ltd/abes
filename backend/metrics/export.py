# Author: Bradley R. Kinnard
"""
Metrics export utilities.
Export metrics to JSON, CSV, or Prometheus format.
"""

import csv
import io
import json
from dataclasses import asdict
from datetime import datetime
from typing import Optional

from . import (
    EcologyMetrics,
    AgentMetrics,
    IterationMetrics,
    MetricsCollector,
    get_metrics_collector,
)


class MetricsExporter:
    """Export metrics to various formats."""

    def __init__(self, collector: Optional[MetricsCollector] = None):
        self._collector = collector or get_metrics_collector()

    def _serialize_datetime(self, obj):
        """JSON serializer for datetime objects."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def export_ecology_json(self, limit: Optional[int] = None) -> str:
        """Export ecology metrics to JSON."""
        history = self._collector.get_ecology_history(limit)
        data = [asdict(m) for m in history]
        return json.dumps(data, default=self._serialize_datetime, indent=2)

    def export_iteration_json(self, limit: Optional[int] = None) -> str:
        """Export iteration metrics to JSON."""
        history = self._collector.get_iteration_history(limit)
        data = [asdict(m) for m in history]
        return json.dumps(data, default=self._serialize_datetime, indent=2)

    def export_agents_json(self) -> str:
        """Export agent metrics to JSON."""
        agents = self._collector.get_agent_metrics()
        data = {name: asdict(m) for name, m in agents.items()}
        return json.dumps(data, default=self._serialize_datetime, indent=2)

    def export_ecology_csv(self, limit: Optional[int] = None) -> str:
        """Export ecology metrics to CSV."""
        history = self._collector.get_ecology_history(limit)
        if not history:
            return ""

        output = io.StringIO()
        fields = list(asdict(history[0]).keys())
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()

        for m in history:
            row = asdict(m)
            row['computed_at'] = row['computed_at'].isoformat()
            writer.writerow(row)

        return output.getvalue()

    def export_iteration_csv(self, limit: Optional[int] = None) -> str:
        """Export iteration metrics to CSV."""
        history = self._collector.get_iteration_history(limit)
        if not history:
            return ""

        output = io.StringIO()
        fields = list(asdict(history[0]).keys())
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()

        for m in history:
            row = asdict(m)
            row['timestamp'] = row['timestamp'].isoformat()
            writer.writerow(row)

        return output.getvalue()

    def export_prometheus(self) -> str:
        """Export current metrics in Prometheus format."""
        latest = self._collector.get_latest_ecology()
        agents = self._collector.get_agent_metrics()

        lines = [
            "# HELP abes_beliefs_total Total number of beliefs",
            "# TYPE abes_beliefs_total gauge",
        ]

        if latest:
            lines.append(f"abes_beliefs_total {latest.total_beliefs}")
            lines.append(f"abes_beliefs_active {latest.active_beliefs}")
            lines.append(f"abes_beliefs_deprecated {latest.deprecated_beliefs}")
            lines.append(f"abes_clusters_total {latest.cluster_count}")
            lines.append(f"abes_confidence_avg {latest.avg_confidence:.4f}")
            lines.append(f"abes_tension_avg {latest.avg_tension:.4f}")

        lines.append("")
        lines.append("# HELP abes_agent_runs_total Agent run counts")
        lines.append("# TYPE abes_agent_runs_total counter")

        for name, am in agents.items():
            label = name.replace("-", "_")
            lines.append(f'abes_agent_runs_total{{agent="{label}"}} {am.run_count}')
            lines.append(f'abes_agent_duration_ms_avg{{agent="{label}"}} {am.avg_duration_ms:.2f}')
            lines.append(f'abes_agent_errors_total{{agent="{label}"}} {am.errors}')

        return "\n".join(lines)

    def export_summary(self) -> dict:
        """Export a summary of current metrics."""
        latest = self._collector.get_latest_ecology()
        agents = self._collector.get_agent_metrics()
        iterations = self._collector.get_iteration_history(10)

        summary = {
            "ecology": asdict(latest) if latest else None,
            "agents": {name: asdict(m) for name, m in agents.items()},
            "recent_iterations": len(iterations),
            "exported_at": datetime.now().isoformat(),
        }

        # convert datetimes
        if summary["ecology"] and summary["ecology"].get("computed_at"):
            summary["ecology"]["computed_at"] = summary["ecology"]["computed_at"].isoformat()

        for am in summary["agents"].values():
            if am.get("last_run"):
                am["last_run"] = am["last_run"].isoformat()

        return summary


__all__ = ["MetricsExporter"]
