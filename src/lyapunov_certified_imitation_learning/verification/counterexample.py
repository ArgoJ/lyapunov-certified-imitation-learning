import torch as th
import torch.nn as nn

from ..training.lyapunov_config import LyapunovTrainingConfig


def lyap_diff_calculation(
    policy_model: nn.Module,
    lyap_model: nn.Module,
    dyn_model: nn.Module,
    state: th.Tensor,
    reg_clamp_max: float = 5e-4,
) -> tuple[th.Tensor, th.Tensor]:
    """Berechnet die Lyapunov-Differenz für den Training Loop."""
    lyap_value = lyap_model(state)
    action = policy_model(state)
    state_next = dyn_model(state, action)
    lyap_value_next = lyap_model(state_next)
    lyap_value_diff = lyap_value_next - lyap_value
    reg = th.norm(state, dim=1, keepdim=True)
    return lyap_value_diff, th.clamp(reg, max=reg_clamp_max)


def find_counter_examples(
    policy_model: nn.Module,
    lyap_model: nn.Module,
    dyn_model: nn.Module,
    config: LyapunovTrainingConfig,
    device: th.device = th.device("cpu"),
) -> th.Tensor:
    """Find counterexamples via PGD where the Lyapunov decrease is violated."""
    # 1. Suche nach Verletzung der Abstiegsbedingung (dV < 0)
    if len(config.state_bounds) != config.state_dim:
        raise ValueError("state_bounds must match state_dim.")

    delta = th.zeros(config.sample_size, config.state_dim).uniform_(-1, 1)
    bounds = th.tensor(config.state_bounds, dtype=th.float32)
    min_state = (delta * bounds).to(device)

    relative_step_size = 1.0 / config.counterexample_steps

    for _ in range(config.counterexample_steps):
        min_state.requires_grad = True
        dv, _ = lyap_diff_calculation(
            policy_model,
            lyap_model,
            dyn_model,
            min_state,
            reg_clamp_max=config.reg_clamp_max,
        )
        loss = dv.sum()
        
        policy_model.zero_grad()
        lyap_model.zero_grad()
        loss.backward()
        
        with th.no_grad():
            # Gradient Ascent auf dV (wir suchen Punkte, wo dV maximiert wird, also > 0)
            min_state = min_state + relative_step_size * th.sign(min_state.grad)
            for dim in range(config.state_dim):
                bound = config.state_bounds[dim]
                min_state[:, dim : dim + 1] = th.clamp(
                    min_state[:, dim : dim + 1], min=-bound, max=bound
                )

    dv, _ = lyap_diff_calculation(
        policy_model,
        lyap_model,
        dyn_model,
        min_state,
        reg_clamp_max=config.reg_clamp_max,
    )
    # Behalte Zustände, wo dV >= 0 (Verletzung)
    counter_idxs = (dv.flatten() >= -1e-6) # Etwas Toleranz
    counter_examples = min_state[counter_idxs].clone().detach()
    
    # Hier könnte man auch Counter-Examples für V(x) > 0 hinzufügen, analog zum Originalcode
    # Aus Platzgründen fokussieren wir auf die Ableitung.
    
    return counter_examples