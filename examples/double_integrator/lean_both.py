import torch as th
import torch.nn as nn

from lcil.lyapunov_learning import LyapunovTrainingConfig, train_lyapunov, NeuralLyapunovCandidate
from lcil.certification import LyapunovCertificationConfig, certify_lyapunov
from lcil.utils import lcil_plt, ICNN, MLP

# from lyapunov_certified_imitation_learning.utils.package_logger import PackageLogger
# import logging
# PackageLogger.setup(level=logging.DEBUG)


from double_integrator_dyn import DoubleIntegratorDynamics


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

	training_config = LyapunovTrainingConfig(
		state_dim=2,
		state_bounds=(2.0, 2.0),
		sample_size=1000,
		batch_size=512,
		outer_epochs=10,
		steps_per_epoch=5,
		counterexample_every=10,
		learning_rate=1e-2,
		seed=5912354,
		kappa=0.05,
		invariance_weight=1.0,
		rho_growth_gamma=1.1,
		roa_weight=0.1,
		l1_weight=1e-6,
	)

	certification_config = LyapunovCertificationConfig.from_training_config(
		training_config,
		cert_step=1.0,
		cert_origin_exclusion=None,
		cert_rho_scaling=1.2,
		cert_bisection_tol=1e-3,
		cert_max_scale_steps=20,
		cert_max_bisection_steps=40,
		cert_method="alpha-crown",
	)

	train_results = train_lyapunov(
		policy_model,
		lyap_model,
		dyn_model,
		training_config,
		device=device,
		models_folder="results/models/double_integrator",
	)

	_, cert_results = certify_lyapunov(
		policy_model,
		lyap_model,
		dyn_model,
		certification_config,
		rho_estimate=train_results.rho_estimate,
		device=device,
	)

	lcil_plt.certified_regions_2d(
		cert_results.certified_regions,
		cert_results.failed_regions,
		state_labels=["x", "v"],
		html_path="results/plots/certified_regions.html",
	)


if __name__ == "__main__":
	main()