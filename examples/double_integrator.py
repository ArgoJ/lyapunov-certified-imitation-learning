import torch as th
import torch.nn as nn

from lyapunov_certified_imitation_learning.models import ICNN, MLP, NeuralLyapunovCandidate
from lyapunov_certified_imitation_learning.training import train_lyapunov, LyapunovTrainingConfig
import lyapunov_certified_imitation_learning.utils.plot as lcil_plt

# from lyapunov_certified_imitation_learning.utils.package_logger import PackageLogger
# import logging
# PackageLogger.setup(level=logging.DEBUG)


class DoubleIntegratorDynamics(nn.Module):
	"""Discrete-time double integrator dynamics."""

	def __init__(self, dt: float = 0.1):
		super().__init__()
		self.dt = dt

	def forward(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
		if u.ndim == 1:
			u = u.unsqueeze(1)
		x_pos = x[:, 0:1]
		x_vel = x[:, 1:2]
		x_next_pos = x_pos + self.dt * x_vel
		x_next_vel = x_vel + self.dt * u
		return th.cat([x_next_pos, x_next_vel], dim=1)


def main() -> None:
	device = th.device("cpu")

	policy_model = MLP([2, 16, 16, 1], ["tanh", "tanh", "identity"]).to(device)
	lyap_feature = ICNN([2, 32, 1], ["relu", "identity"]).to(device)
	lyap_model = NeuralLyapunovCandidate(
		feature_net=lyap_feature,
		state_dim=2,
		epsilon=1e-3,
	).to(device)
	dyn_model = DoubleIntegratorDynamics(dt=0.1).to(device)

	config = LyapunovTrainingConfig(
		state_dim=2,
		state_bounds=(2.0, 2.0),
		sample_size=1000,
		batch_size=512,
		outer_epochs=500,
		steps_per_epoch=10,
		counterexample_every=10,
		learning_rate=1e-2,
		seed=5912354,
		kappa=0.05,
		invariance_weight=1.0,
		rho_growth_gamma=1.1,
		roa_weight=0.1,
		l1_weight=1e-6,
		run_certification=True,
	)

	results = train_lyapunov(
		policy_model,
		lyap_model,
		dyn_model,
		config,
		device=device,
		models_prefix="results/models/double_integrator_lyap",
		results_path="results/double_integrator_crown_result.txt",
	)

	lcil_plt.certified_regions_2d(
		results["certified_regions"],
		results["failed_regions"],
		state_labels=["x", "v"],
		html_path="plots/certified_regions.html",
	)


if __name__ == "__main__":
	main()

