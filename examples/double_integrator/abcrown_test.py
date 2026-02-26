import os
import torch as th

from torch import nn
from abcrown import (
    ABCrownSolver, 
    VerificationSpec, 
    ConfigBuilder, 
    input_vars, 
    output_vars
)
from double_integrator_dyn import DoubleIntegratorDynamics
from lcil.imitation_learning_mlp import MLPPolicy as Controller



# Define a NN Lyapunov function
class Lyapunov(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        layers = []
        for i in range(len(dims)-2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(dims[-2], dims[-1]))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)


# Construct the computation graph for one step closed-loop dynamics
class ClosedLoopComputationGraph(nn.Module):
    def __init__(self, dynamics: DoubleIntegratorDynamics, controller: Controller):
        super().__init__()
        self.dynamics = dynamics
        self.controller = controller

    def forward(self, x):
        u = self.controller(x)
        x_dot = self.dynamics.f_torch(x, u)
        return x_dot


# Construct the computation graph for Lyapunov analysis
# import JacobianOP from auto_LiRPA
from auto_LiRPA.jacobian import JacobianOP
class LyapunovComputationGraph(nn.Module):
    def __init__(self, dynamics: DoubleIntegratorDynamics, controller: Controller, lyapunov: Lyapunov):
        super().__init__()
        self.dynamics = dynamics
        self.controller = controller
        self.lyapunov = lyapunov

    def forward(self, x):
        x = x.clone().requires_grad_(True)
        V_x = self.lyapunov(x)
        u = self.controller(x)
        x_dot = self.dynamics(x, u)
        dVdx = JacobianOP.apply(self.lyapunov(x), x).squeeze(1)
        V_dot = th.sum(dVdx * x_dot, dim=1, keepdim=True)
        return th.cat((V_x, V_dot), dim=1)
    

def main() -> None:
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    dynamics = DoubleIntegratorDynamics(dt=0.1)
    policy_path = os.path.join(args.policy_path)
    controller = Controller.load(
        path=policy_path,
        map_location=device,
    ).to(device)
    lyapunov = Lyapunov(dims=[2, 40, 40, 1])
    model = LyapunovComputationGraph(dynamics, controller, lyapunov)
    figure_dir = os.path.join(os.path.dirname(__file__), "neural_lyapunov_dependency")
    ckpt_path = os.path.join(figure_dir, "seed_0.pth")
    v_min = 0.0106
    v_max = 0.989
    v_dot_min = 0.0
    state = th.load(ckpt_path, map_location=device)
    state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    x = input_vars(2)
    y = output_vars(2)  # y[0] = V(x), y[1] = V_dot
    input_constraint = (x >= [-4.8, -10.8]) & (x <= [4.8, 10.8])
    output_constraint = (y[0] < v_min) | (y[0] > v_max) | (y[1] < v_dot_min)
    spec = VerificationSpec.build_spec(
        input_vars=x,
        output_vars=y,
        input_constraint=input_constraint,
        output_constraint=output_constraint,
    )

    cfg = ConfigBuilder.from_defaults()
    cfg = cfg.set(model__with_jacobian=True)
    solver = ABCrownSolver(spec, model, config=cfg)
    result = solver.solve()

    print("[info] verifying Lyapunov tutorial graph with ABCrown API")
    print(f"status={result.status}, success={result.success}")


if __name__ == "__main__":
    main()