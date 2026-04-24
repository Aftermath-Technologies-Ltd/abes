"""
Configuration for ABES using Pydantic settings.
Reads from environment variables or .env file.
"""

from enum import Enum
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DecayProfile(str, Enum):
    """Preset decay configurations for different use cases."""

    aggressive = "aggressive"  # 0.99 - forgets quickly, good for transient info
    moderate = "moderate"  # 0.995 - balanced (default)
    conservative = "conservative"  # 0.999 - retains beliefs longer
    persistent = "persistent"  # 0.9999 - almost no decay


# Decay rate mapping for presets
DECAY_PROFILES = {
    DecayProfile.aggressive: 0.99,
    DecayProfile.moderate: 0.995,
    DecayProfile.conservative: 0.999,
    DecayProfile.persistent: 0.9999,
}


class ABESSettings(BaseSettings):
    """Global config for the belief ecology system."""

    # environment
    environment: str = "development"

    # storage backend: "memory" or "sqlite"
    storage_backend: Literal["memory", "sqlite"] = "memory"
    database_url: str = "sqlite+aiosqlite:///./data/abes.db"

    # --- Embedding Configuration (spec 3.5) ---
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    embedding_batch_size: int = 64

    # --- Decay Parameters (spec 3.4.1) ---
    decay_profile: DecayProfile = DecayProfile.moderate
    decay_rate: float = 0.995  # per-hour multiplier, overridden by profile if set
    # half-life ~138 hours at 0.995

    # --- LLM Configuration ---
    # hybrid = local Ollama for beliefs, OpenAI for live/real-time queries
    llm_provider: Literal["ollama", "openai", "anthropic", "hybrid", "none"] = "ollama"
    llm_fallback_enabled: bool = True  # if LLM fails, return beliefs without response

    # --- Status Thresholds (spec 3.4.2) ---
    confidence_threshold_decaying: float = 0.3
    confidence_threshold_deprecated: float = 0.1
    stale_days_deprecated: int = 30  # unused beliefs older than this get deprecated

    # --- Contradiction & Tension (spec 3.4.3, 3.4.4) ---
    # low gate (0.25) lets NLI catch implicit contradictions like vegetarian/steak
    similarity_threshold_contradiction: float = 0.25
    tension_threshold_high: float = 0.65  # flags for resolution (lowered to catch NLI-detected pairs)
    tension_cap: float = 10.0

    # --- Relevance (spec 3.4.5) ---
    relevance_threshold_min: float = 0.1  # lowered for chat - be more inclusive

    # --- Ranking Weights (spec 3.4.6) ---
    ranking_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "relevance": 0.4,
            "confidence": 0.3,
            "recency": 0.2,
            "tension": 0.1,
        }
    )
    ranking_weight_bounds: tuple[float, float] = (0.1, 0.6)  # RL tuning limits
    recency_window_hours: int = 168  # 1 week for recency normalization

    # --- Mutation Triggers (spec 3.4.7) ---
    tension_threshold_mutation: float = 0.5  # lowered for chat responsiveness
    confidence_threshold_mutation: float = 0.9  # allow mutation of higher-confidence beliefs

    mutation_strategy: str = "rule"  # "rule" or "llm"

    # --- Resolution Triggers (spec 3.4.8) ---
    tension_threshold_resolution: float = 0.6  # lowered for chat
    confidence_threshold_resolution: float = 0.7

    # --- Clustering (spec 3.6) ---
    cluster_similarity_threshold: float = 0.7  # min similarity to join cluster
    cluster_merge_threshold: float = 0.85
    cluster_maintenance_interval: int = 100  # iterations between maintenance

    # --- Safety Limits (spec 3.8) ---
    max_belief_content_length: int = 2000
    max_active_beliefs: int = 10000
    max_beliefs_per_cluster: int = 500
    max_mutation_depth: int = 5

    # --- Belief Stack (active reasoning set) ---
    belief_stack_size: int = 50  # max beliefs in active reasoning stack
    dormancy_salience_threshold: float = 0.05  # below this -> dormant

    # --- Salience ---
    default_half_life_days: float = 7.0  # salience half-life default
    salience_boost_on_reinforce: float = 0.15  # bump when reinforced
    max_contradiction_pairs_per_iteration: int = 50000
    max_snapshot_size_mb: int = 50

    # --- Reinforcement (agent-level) ---
    reinforcement_similarity_threshold: float = 0.55
    reinforcement_confidence_boost: float = 0.1
    reinforcement_cooldown_seconds: int = 60
    max_reinforced_confidence: float = 0.95

    # --- Deduplication ---
    dedupe_similarity_threshold: float = 0.95

    # --- LLM / Ollama Configuration ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b-instruct-q4_0"
    ollama_timeout: float = 120.0
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1024
    llm_context_beliefs: int = 15  # max beliefs to include in LLM context

    # --- OpenAI Configuration ---
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"

    # --- Anthropic Configuration ---
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-haiku-20240307"

    # --- Authentication ---
    jwt_secret: str = ""  # auto-generated if empty
    jwt_expiration_hours: int = 168  # 1 week

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @model_validator(mode="after")
    def apply_decay_profile(self) -> "ABESSettings":
        """Apply decay profile to decay_rate if using a preset."""
        if self.decay_profile in DECAY_PROFILES:
            # Only override if decay_rate is still the default
            if self.decay_rate == 0.995:
                object.__setattr__(self, "decay_rate", DECAY_PROFILES[self.decay_profile])
        return self


# singleton instance
settings = ABESSettings()


__all__ = ["ABESSettings", "settings", "DecayProfile", "DECAY_PROFILES"]
