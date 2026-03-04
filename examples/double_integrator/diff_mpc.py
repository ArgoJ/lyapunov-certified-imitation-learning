import torch
import torch.nn as nn
import inspect

from acados_template import AcadosOcpBatchSolver, AcadosOcpSolver
from leap_c.ocp.acados.torch import AcadosDiffMpcTorch
from leap_c.ocp.acados.utils import create_solver as leapc_create_solver


class _CodeGenOptsCompat:
    def __init__(self, ocp):
        self._ocp = ocp

    @property
    def code_export_directory(self):
        return getattr(self._ocp, "code_export_directory", None)

    @code_export_directory.setter
    def code_export_directory(self, value):
        self._ocp.code_export_directory = value


def _patch_batch_solver_api_compat() -> None:
    if getattr(leapc_create_solver, "_lcil_batch_solver_patched", False):
        return

    batch_solver_init_sig = inspect.signature(AcadosOcpBatchSolver.__init__)
    if "N_batch_init" in batch_solver_init_sig.parameters:
        leapc_create_solver._lcil_batch_solver_patched = True
        return

    class _CompatAcadosOcpBatchSolver(AcadosOcpBatchSolver):
        def __init__(self, *args, **kwargs):
            if "N_batch_init" in kwargs and "N_batch_max" not in kwargs:
                kwargs["N_batch_max"] = kwargs.pop("N_batch_init")
            kwargs.pop("check_code_reuse_possible", None)
            super().__init__(*args, **kwargs)

    leapc_create_solver.AcadosOcpBatchSolver = _CompatAcadosOcpBatchSolver
    leapc_create_solver._lcil_batch_solver_patched = True

class DiffMPCPolicy(nn.Module):
    def __init__(
        self,
        ocp_solver: AcadosOcpSolver,
        batch_size: int,
        num_threads_batch_solver: int = 1,
    ):
        super().__init__()
        ocp = ocp_solver.acados_ocp
        self.nx = ocp.dims.nx
        self.nu = ocp.dims.nu

        horizon = ocp.solver_options.N_horizon
        if horizon is None:
            horizon = ocp.dims.N
        if horizon is None or horizon <= 0:
            raise ValueError("Unable to determine MPC horizon N from ocp.solver_options.N_horizon or ocp.dims.N.")

        self.dt = ocp.solver_options.tf / horizon

        if not hasattr(ocp, "code_gen_opts"):
            ocp.code_gen_opts = _CodeGenOptsCompat(ocp)

        _patch_batch_solver_api_compat()
        
        self.diff_mpc = AcadosDiffMpcTorch(
            ocp,
            n_batch_init=batch_size,
            num_threads_batch_solver=num_threads_batch_solver,
            dtype=torch.float32
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        x : torch.Tensor
            State tensor of shape (Batch, nx)

        Returns
        -------
        torch.Tensor
            Control input tensor of shape (Batch, nu) corresponding 
            to the first control action u_0.
        """
        result = self.diff_mpc(x0=x)

        _, u_0, _, _, _ = result
        return u_0