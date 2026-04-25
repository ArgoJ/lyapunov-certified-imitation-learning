import importlib
import unittest

import numpy as np
import torch as th
import torch.nn as nn


class _ZeroPolicy(nn.Module):
    def __init__(self, nu: int = 1):
        super().__init__()
        self.nu = nu

    def forward(self, x: th.Tensor) -> th.Tensor:
        return th.zeros((x.shape[0], self.nu), dtype=x.dtype, device=x.device)


class _IdentityDynamics(nn.Module):
    def forward(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
        del u
        return x


class _QuadraticLyapunov(nn.Module):
    def forward(self, x: th.Tensor) -> th.Tensor:
        return (x * x).sum(dim=1, keepdim=True)


class TestLiRPAWrapperGlobalBounds(unittest.TestCase):
    def test_setup_verifier_uses_global_certification_bounds(self) -> None:
        try:
            lirpa_wrapper = importlib.import_module("lcil.certification.lirpa_wrapper")
            config_module = importlib.import_module("lcil.certification.config")
        except Exception as exc:  # pragma: no cover - depends on local environment
            raise unittest.SkipTest(f"Could not import LiRPA certifier modules: {exc}") from exc

        LiRPACertifier = lirpa_wrapper.LiRPACertifier
        LyapunovCertificationConfig = config_module.LyapunovCertificationConfig

        config = LyapunovCertificationConfig(
            state_dim=3,
            cert_bounds=np.array([[-2.0, -3.0, -4.0], [2.0, 3.0, 4.0]], dtype=np.float32),
            kappa=0.1,
            bins_per_dim=2,
            origin_exclusion=0.0,
            cert_method="crown",
            batch_size=4,
            use_ibp_filter=False,
        )

        certifier = LiRPACertifier(
            policy_model=_ZeroPolicy(),
            lyap_model=_QuadraticLyapunov(),
            dyn_model=_IdentityDynamics(),
            config=config,
            device=th.device("cpu"),
        )

        verifier = certifier._setup_verifier()

        self.assertTrue(th.equal(verifier.lbx, certifier.bounds[0]))
        self.assertTrue(th.equal(verifier.ubx, certifier.bounds[1]))


if __name__ == "__main__":
    unittest.main(verbosity=2)