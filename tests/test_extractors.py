import sys
import unittest
from pathlib import Path

import numpy as np


# Ensure we can import from ./src without installing the package.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from lyapunov_certified_imitation_learning.data_generation.extractor import (
    LinearSystemExtractor, 
    MPCConfigExtractor
)

from acados_solver_example import get_basic_double_integrator_ocp_solver


class TestExtractors(unittest.TestCase):
    def __init__(self, methodName = "runTest"):
        super().__init__(methodName)
        self.solver, self.system_info = get_basic_double_integrator_ocp_solver()
        
    def test_passes_linear_system_extraction(self) -> None:
        extracted = LinearSystemExtractor.get_system(self.solver)

        np.testing.assert_allclose(extracted.A, self.system_info["A_d"])
        np.testing.assert_allclose(extracted.B, self.system_info["B_d"])
        
    def test_passes_cfg_extraction(self) -> None:
        extracted = MPCConfigExtractor.get_cfg(self.solver)
        
        N = self.solver.acados_ocp.solver_options.N_horizon
        dt = float(self.solver.acados_ocp.solver_options.tf) / int(N)

        self.assertEqual(extracted.N, N)
        self.assertEqual(extracted.dt, dt)
        self.assertEqual(extracted.T_sim, 0)
        self.assertEqual(extracted.nx, self.solver.acados_ocp.dims.nx)
        self.assertEqual(extracted.nu, self.solver.acados_ocp.dims.nu)
        
        # Check x0 only if it's not None
        if self.solver.acados_ocp.constraints.x0 is not None:
            np.testing.assert_allclose(extracted.x0, self.solver.acados_ocp.constraints.x0)
        else:
            self.assertIsNone(extracted.x0)


if __name__ == "__main__":
    unittest.main()
