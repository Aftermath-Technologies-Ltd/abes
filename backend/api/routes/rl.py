# Author: Bradley R. Kinnard
"""RL status API routes."""

from fastapi import APIRouter

from ...core.deps import get_rl_state

router = APIRouter(prefix="/rl", tags=["rl"])


@router.get("/status")
async def get_rl_status():
    """Return current RL policy training state.

    Exposes the mutable :class:`~backend.core.deps.RLState` singleton so
    callers can observe whether the policy has been trained, the total number
    of trajectories collected, the most recent epoch loss, the current 7D
    action output means, and how many scheduler iterations remain before the
    next training run.

    Returns:
        JSON matching the RL status schema.
    """
    state = get_rl_state()
    return {
        "policy_trained": state.policy_trained,
        "total_trajectories": state.total_trajectories,
        "last_training_epoch_loss": state.last_training_epoch_loss,
        "current_action_means": state.current_action_means,
        "training_interval": state.training_interval,
        "iterations_until_next_training": state.iterations_until_next_training,
    }
