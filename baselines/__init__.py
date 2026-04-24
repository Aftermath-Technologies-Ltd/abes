# Author: Bradley R. Kinnard
"""
Baseline systems for comparative measurement.
These implement minimal functionality for benchmarking, not feature parity.
"""

from .plain_llm_runner import PlainLLMRunner
from .append_only_memory import AppendOnlyMemory

__all__ = ["PlainLLMRunner", "AppendOnlyMemory"]
