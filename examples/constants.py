from pathlib import Path

POLICY_MODEL_FILENAME = "policy_model.pt"

LYAPUNOV_MODEL_FILENAME = "lyapunov_model.pt"
LYAPUNOV_DIRNAME = "lyapunov"

CERTIFICATION_DIRNAME = "certification"
CERTIFICATION_CONFIG_FILENAME = "certification_config.json"
CERTIFICATION_TESTER_RESULTS_FILENAME = "certification_tester_results.json"
CERTIFICATION_DETAILS_FILENAME = "certification_details.npz"
LEVEL_SET_FILENAME = "level_set_estimate.json"

RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"