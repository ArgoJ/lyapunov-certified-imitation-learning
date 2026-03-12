from __future__ import annotations

import torch as th
import torch.nn as nn
import numpy as np

from contextlib import nullcontext
from typing import Sequence
from dataclasses import dataclass, replace
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from pkg_logger import get_package_logger, PackageLogger

from .config import LyapunovCertificationConfig
from .models import ClosedLoopLyapunovConditionVerifier

__logger__ = get_package_logger(__name__)


@dataclass(frozen=True)
class RegionCertificationResult:
    """Result container for a full-region certification pass."""
    success: bool
    counter_examples: NDArray
    failed_regions: NDArray
    certified_regions: NDArray


class BaseCertifier(ABC):
    """
    Abstract base class for Lyapunov certifiers. 
    Definie the core logic for the Certification, while delegating backend-specific details to subclasses via abstract methods.
    """

    def __init__(
            self, 
            policy_model: nn.Module,
            lyap_model: nn.Module,
            dyn_model: nn.Module,
            config: LyapunovCertificationConfig,
            device: th.device = th.device("cpu"),
    ):
        """Initialize a backend-agnostic Lyapunov certifier.

        Parameters
        ----------
        policy_model : nn.Module
            Control policy network ``u = pi(x)`` used in the closed-loop verifier.
        lyap_model : nn.Module
            Candidate Lyapunov network ``V(x)``.
        dyn_model : nn.Module
            Dynamics model for one-step state propagation in closed loop.
        config : LyapunovCertificationConfig
            Certification configuration (bounds, grid step, rho search settings,
            backend options).
        device : th.device, optional
            Device used for all model parameters and certification tensors,
            by default ``th.device("cpu")``.
        """
        self.config = self._resolve_config(config)
        self.device = device

        self.policy_model = policy_model.to(self.device).eval()
        self.lyap_model = lyap_model.to(self.device).eval()
        self.dyn_model = dyn_model.to(self.device).eval()

        self.bounds = self._resolve_bounds(config.state_bounds, device)
        
        self.regions = None
        self.verifier = None

    # ==========================================
    # ABSTRACT METHODS
    # ==========================================

    @abstractmethod
    def setup_backend(self, *args, **kwargs) -> None:
        """Initialize backend-specific verifier state.

        Parameters
        ----------
        *args
            Backend-specific positional arguments.
        **kwargs
            Backend-specific keyword arguments.
        """
        pass

    @abstractmethod
    def _certify_batched_regions(
            self,
            lbs: th.Tensor,
            ubs: th.Tensor,
            rho: float,
            early_exit: bool = True,
            *args, **kwargs
    ) -> tuple[th.Tensor, th.Tensor]:
        """Certify a batch of axis-aligned regions.

        Parameters
        ----------
        lbs : th.Tensor
            Region lower bounds with shape ``(n, state_dim)``.
        ubs : th.Tensor
            Region upper bounds with shape ``(n, state_dim)``.
        rho : float
            Lyapunov level-set value to certify.
        early_exit : bool, optional
            If ``True``, backend may stop once a violation is found.
        *args
            Backend-specific positional arguments.
        **kwargs
            Backend-specific keyword arguments.

        Returns
        -------
        tuple[th.Tensor, th.Tensor]
            ``(is_safe, centers)`` where ``is_safe`` is boolean per region,
            ``centers`` contains representative points.
        """
        pass

    # ==========================================
    # CORE LOGIC
    # ==========================================

    @staticmethod
    def _resolve_config(config: LyapunovCertificationConfig) -> LyapunovCertificationConfig:
        """Normalize and validate certification configuration.

        Parameters
        ----------
        config : LyapunovCertificationConfig
            Raw certification configuration.

        Returns
        -------
        LyapunovCertificationConfig
            Copy of ``config`` with normalized fields.

        Raises
        ------
        ValueError
            If ``cert_bins_per_dim`` is invalid.
        """
        raw_bins = config.cert_bins_per_dim
        if isinstance(raw_bins, (int, np.integer)):
            bins_per_dim = (int(raw_bins),) * config.state_dim
        else:
            bins_per_dim = tuple(int(bins) for bins in raw_bins)
            if len(bins_per_dim) != config.state_dim:
                raise ValueError("cert_bins_per_dim must be scalar or match state_dim.")
        if any(bins <= 0 for bins in bins_per_dim):
            raise ValueError("cert_bins_per_dim must contain only positive integers.")

        resolved_config = replace(
            config,
            cert_method=config.cert_method.strip().lower(),
            rho_scaling=max(config.rho_scaling, 1.01),
            cert_bins_per_dim=bins_per_dim,
        )
        return resolved_config

    @staticmethod
    def _resolve_bounds(state_bounds: Sequence[float], device: th.device) -> th.Tensor:
        """Convert state bounds to a tensor on the target device.

        Parameters
        ----------
        state_bounds : Sequence[float]
            Lower and upper bounds as ``(2, state_dim)``.
        device : th.device
            Device where the tensor should be allocated.

        Returns
        -------
        th.Tensor
            Bounds tensor with shape ``(2, state_dim)`` and dtype ``float32``.

        Raises
        ------
        ValueError
            If the bounds do not have shape ``(2, state_dim)``.
        """
        bounds = th.as_tensor(state_bounds, dtype=th.float32, device=device)
        if bounds.ndim != 2 or bounds.shape[0] != 2:
            raise ValueError("state_bounds must be a sequence of shape (2, nx) [lb, ub].")
        return bounds

    @staticmethod
    def _iterative_rho_search(total:int, desc: str, initial_values: tuple[float, float], step_fn: callable) -> tuple[float | None, float]:
        """Run iterative rho updates until stopping criterion is met.

        Parameters
        ----------
        total : int
            Maximum number of iterations.
        desc : str
            Progress-bar description.
        initial_values : tuple[float, float]
            Initial ``(rho_lo, rho_up)`` values.
        step_fn : callable
            Function mapping ``(rho_lo, rho_up)`` to ``(stop, rho_lo, rho_up)``.

        Returns
        -------
        tuple[float | None, float]
            Final ``(rho_lo, rho_up)`` pair.
        """
        rho_lo, rho_up = initial_values
        with __logger__.tqdm(range(total), desc=desc) as pbar:
            for _ in pbar:
                try:
                    stop, rho_lo, rho_up = step_fn(rho_lo, rho_up)
                    if stop: 
                        break
                finally:
                    lo_display = f"{rho_lo:.4f}" if rho_lo is not None else "None"
                    pbar.set_postfix({"rho_lo": lo_display, "rho_up": f"{rho_up:.4f}"})
        return rho_lo, rho_up

    def _get_suppress_ctx(self):
        """Return context manager controlling native backend output.

        Returns
        -------
        contextlib.AbstractContextManager
            Suppression context if enabled, otherwise ``nullcontext``.
        """
        if self.config.suppress_native_output:
            lirpa_ctx = PackageLogger.suppress_native_output(suppress_stderr=True)
        else:
            lirpa_ctx = nullcontext()
        return lirpa_ctx

    def _filter_sublevel_regions(
        self,
        lbs: th.Tensor,
        ubs: th.Tensor,
        rho: float,
    ) -> tuple[th.Tensor, th.Tensor]:
        """Optionally prune boxes proven to lie completely outside ``V(x) <= rho``.

        The default implementation performs no pruning. Backends may override
        this with a sound lower-bound check for the Lyapunov model.
        """
        return lbs, ubs

    def _setup_verifier(self) -> ClosedLoopLyapunovConditionVerifier:
        """Construct and initialize the closed-loop Lyapunov verifier.

        Returns
        -------
        ClosedLoopLyapunovConditionVerifier
            Verifier module moved to ``self.device`` and set to eval mode.
        """
        verifier = ClosedLoopLyapunovConditionVerifier(
            policy_model=self.policy_model,
            lyap_model=self.lyap_model,
            dyn_model=self.dyn_model,
            lbx=self.bounds[0],
            ubx=self.bounds[1],
            invariance_weight=self.config.invariance_weight,
        ).to(self.device)
        verifier.eval()
        return verifier

    def _resolve_origin_exclusion(self) -> th.Tensor:
        """Resolve origin exclusion widths per state dimension.

        Returns
        -------
        th.Tensor
            Non-negative exclusion widths with shape ``(state_dim,)`` clipped to
            remain within available bounds around the origin.
        """
        # Default: 1% of per-dimension bound radius, capped at 0.1.
        default_exclusion = th.minimum(
            self.bounds.abs().max(dim=0).values * 0.01,
            th.full((self.config.state_dim,), 0.1, dtype=th.float32, device=self.device),
        )

        raw_exclusion = self.config.origin_exclusion
        if raw_exclusion is None:
            exclusion = default_exclusion
        elif isinstance(raw_exclusion, (int, float)):
            scalar = float(raw_exclusion)
            if scalar < 0:
                raise ValueError("cert_origin_exclusion must be non-negative.")
            exclusion = th.full(
                (self.config.state_dim,),
                scalar,
                dtype=th.float32,
                device=self.device,
            )
        else:
            exclusion = th.as_tensor(raw_exclusion, dtype=th.float32, device=self.device).reshape(-1)
            if exclusion.numel() != self.config.state_dim:
                raise ValueError(
                    "cert_origin_exclusion must be scalar or match state_dim."
                )
            if (exclusion < 0).any():
                raise ValueError("cert_origin_exclusion must be non-negative.")

        # Ensure the exclusion does not extend outside available bounds around zero.
        max_centered = th.clamp(th.minimum(-self.bounds[0], self.bounds[1]), min=0.0)
        return th.minimum(exclusion, max_centered)

    def _certify_regions_recursive(
            self, 
            lbs: th.Tensor, 
            ubs: th.Tensor,
            rho: float,
            collect_details: bool, 
            depth: int, 
            max_depth: int,
            filter: bool = False,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """Recursively certify axis-aligned regions for a fixed ``rho``.

        Parameters
        ----------
        lbs : th.Tensor
            Lower bounds of regions with shape ``(n, state_dim)``.
        ubs : th.Tensor
            Upper bounds of regions with shape ``(n, state_dim)``.
        rho : float
            Lyapunov level-set value to certify.
        collect_details : bool
            If ``False``, returns immediately once any failing region is found.
            If ``True``, recursively subdivides failed regions to collect detailed
            failed boxes and counterexamples.
        depth : int
            Current recursion depth.
        max_depth : int
            Maximum subdivision depth for failed regions.
        filter : bool, optional
            Filter the regions beforehand.

        Returns
        -------
        tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]
            ``(certified_lbs, certified_ubs, failed_lbs, failed_ubs, counter_examples)``.
            Each bound tensor has shape ``(k, state_dim)`` and counterexamples has
            shape ``(m, state_dim)``.
        """
        if len(lbs) == 0:
            empty = th.empty((0, self.config.state_dim), device=self.device)
            return empty, empty, empty, empty, empty

        if filter:
            lbs, ubs = self._filter_sublevel_regions(lbs, ubs, rho)
        if len(lbs) == 0:
            empty = th.empty((0, self.config.state_dim), device=self.device)
            return empty, empty, empty, empty, empty

        is_safe, centers = self._certify_batched_regions(
            lbs, ubs, rho,
            early_exit=not collect_details
        )

        certified_lbs = lbs[is_safe]
        certified_ubs = ubs[is_safe]

        failed_mask = ~is_safe
        failed_lbs = lbs[failed_mask]
        failed_ubs = ubs[failed_mask]
        counter_examples = centers[failed_mask]

        final_failed_lbs = th.empty((0, self.config.state_dim), device=self.device)
        final_failed_ubs = th.empty((0, self.config.state_dim), device=self.device)

        if len(failed_lbs) > 0:
            if not collect_details:
                # Early Exit
                return certified_lbs, certified_ubs, failed_lbs, failed_ubs, counter_examples

            if depth >= max_depth:
                final_failed_lbs = failed_lbs
                final_failed_ubs = failed_ubs
            else:
                # Split each failed box into 2 sub-boxes along its widest dimension.
                # This avoids the 2^state_dim combinatorial explosion in higher dimensions.
                mids = (failed_lbs + failed_ubs) / 2.0
                widths = failed_ubs - failed_lbs
                split_dims = th.argmax(widths, dim=1)
                row_idx = th.arange(failed_lbs.shape[0], device=self.device)

                low_lbs = failed_lbs.clone()
                low_ubs = failed_ubs.clone()
                low_ubs[row_idx, split_dims] = mids[row_idx, split_dims]

                high_lbs = failed_lbs.clone()
                high_ubs = failed_ubs.clone()
                high_lbs[row_idx, split_dims] = mids[row_idx, split_dims]

                sub_lbs = th.cat([low_lbs, high_lbs], dim=0)
                sub_ubs = th.cat([low_ubs, high_ubs], dim=0)

                # Rekursiv certification of all sub-boxes
                sub_c_lbs, sub_c_ubs, sub_f_lbs, sub_f_ubs, sub_cex = self._certify_regions_recursive(
                    sub_lbs, sub_ubs, rho, collect_details, depth + 1, max_depth
                )

                certified_lbs = th.cat([certified_lbs, sub_c_lbs], dim=0)
                certified_ubs = th.cat([certified_ubs, sub_c_ubs], dim=0)
                final_failed_lbs = sub_f_lbs
                final_failed_ubs = sub_f_ubs
                counter_examples = th.cat([counter_examples, sub_cex], dim=0)

        return certified_lbs, certified_ubs, final_failed_lbs, final_failed_ubs, counter_examples

    def _certify_regions(
            self, 
            rho: float,
            collect_details: bool = True
    ) -> RegionCertificationResult:
        """Certify all pre-built regions for a fixed ``rho``.

        Parameters
        ----------
        rho : float
            Lyapunov level-set value to test.
        collect_details : bool, optional
            Whether to recursively split failed regions to gather detailed failure
            information, by default ``True``.

        Returns
        -------
        RegionCertificationResult
            Aggregated success flag plus certified regions, failed regions, and
            counterexamples as NumPy arrays.
        """
        lbs, ubs = self.regions
        
        c_lbs, c_ubs, f_lbs, f_ubs, cex = self._certify_regions_recursive(
            lbs, ubs, rho, collect_details, depth=0, max_depth=3, filter=not collect_details
        )

        # convert for output -> NumPy arrays for easier downstream use
        c_lbs_np = c_lbs.cpu().numpy() if c_lbs.numel() > 0 else np.empty((0, self.config.state_dim), dtype=np.float32)
        c_ubs_np = c_ubs.cpu().numpy() if c_ubs.numel() > 0 else np.empty((0, self.config.state_dim), dtype=np.float32)
        f_lbs_np = f_lbs.cpu().numpy() if f_lbs.numel() > 0 else np.empty((0, self.config.state_dim), dtype=np.float32)
        f_ubs_np = f_ubs.cpu().numpy() if f_ubs.numel() > 0 else np.empty((0, self.config.state_dim), dtype=np.float32)

        # Stack into shape (N, 2, state_dim): [:,0,:] = lb, [:,1,:] = ub
        certified_regions_np = np.stack([c_lbs_np, c_ubs_np], axis=1) if c_lbs_np.shape[0] > 0 else np.empty((0, 2, self.config.state_dim), dtype=np.float32)
        failed_regions_np = np.stack([f_lbs_np, f_ubs_np], axis=1) if f_lbs_np.shape[0] > 0 else np.empty((0, 2, self.config.state_dim), dtype=np.float32)
        counter_examples_np = cex.cpu().numpy() if cex.numel() > 0 else np.empty((0, self.config.state_dim), dtype=np.float32)

        return RegionCertificationResult(
            success=failed_regions_np.shape[0] == 0,
            counter_examples=counter_examples_np,
            failed_regions=failed_regions_np,
            certified_regions=certified_regions_np,
        )

    def _scale_rho_up(self, rho_lo: float, rho_up: float) -> tuple[bool, float, float]:
        """Scales up ``rho_lo`` until the new trial is not certified.

        Parameters
        ----------
        rho_lo : float
            Lower bound of rho, updated if trial is certified.
        rho_up : float
            Upper bound of rho, updated if trial is not certified.

        Returns
        -------
        tuple[bool, float, float]
            A tuple of (stop, rho_lo, rho_up) where stop is a boolean indicating if the search should stop,
            and rho_lo and rho_up are the updated bounds.
        """
        trial = rho_up * self.config.rho_scaling
        if self.is_rho_certified(rho=trial):
            return False, trial, trial
        else:
            return True, rho_lo, trial

    def _scale_rho_down(self, rho_lo: float, rho_up: float) -> tuple[bool, float, float]:
        """Scales down ``rho_up`` until the new trial is either certified 
        or it is below rho_min, in which case we stop and return the last certified rho as lower bound.

        Parameters
        ----------
        rho_lo : float
            Lower bound of rho, updated if trial is certified or the trial is lower than ``rho_lo``.
        rho_up : float
            Upper bound of rho, updated if trial is not certified.

        Returns
        -------
        tuple[bool, float, float]
            A tuple of (stop, rho_lo, rho_up) where stop is a boolean indicating if the search should stop,
            and rho_lo and rho_up are the updated bounds.
        """
        trial = max(self.config.rho_min, rho_up / self.config.rho_scaling)
        if self.is_rho_certified(rho=trial):
            return True, trial, rho_up
         
        if trial <= self.config.rho_min:
            return True, None, trial
        return False, rho_lo, trial

    def _bisect_rho(self, rho_lo: float, rho_up: float) -> tuple[bool, float, float]:
        """Uses the midpoint of rho upper and lower for bisection.

        Parameters
        ----------
        rho_lo : float
            Lower bound of rho (known to be certified).
        rho_up : float
            Upper bound of rho (known to be **not** certified).

        Returns
        -------
        tuple[bool, float, float]
            A tuple of (stop, rho_lo, rho_up) where stop is a boolean indicating if the search should stop,
            and rho_lo and rho_up are the updated bounds.
        """
        rho_mid = 0.5 * (rho_lo + rho_up)

        if self.is_rho_certified(rho=rho_mid):
            rho_lo = rho_mid
        else:
            rho_up = rho_mid

        if rho_up - rho_lo <= self.config.bisection_tol:
            return True, rho_lo, rho_up
        else:
            return False, rho_lo, rho_up
    
    def build_regions(self) -> tuple[th.Tensor, th.Tensor]:
        """Build axis-aligned grid cells and remove cells intersecting the origin window.

        Returns
        -------
        tuple[th.Tensor, th.Tensor]
            ``(lbs, ubs)`` where each tensor has shape ``(n, state_dim)``.
        """
        origin_exclusion = self._resolve_origin_exclusion()
        lbx, ubx = self.bounds[0], self.bounds[1]
        bin_counts = self.config.cert_bins_per_dim

        lb_axes = []
        ub_axes = []
        for idx in range(self.config.state_dim):
            edges = th.linspace(
                lbx[idx],
                ubx[idx],
                steps=int(bin_counts[idx]) + 1,
                device=self.device,
            )
            lb_axes.append(edges[:-1])
            ub_axes.append(edges[1:])

        lb_mesh = th.meshgrid(*lb_axes, indexing="ij")
        ub_mesh = th.meshgrid(*ub_axes, indexing="ij")
        lbs = th.stack([axis.flatten() for axis in lb_mesh], dim=1)
        ubs = th.stack([axis.flatten() for axis in ub_mesh], dim=1)

        overlaps_origin_per_dim = (lbs < origin_exclusion.unsqueeze(0)) & (ubs > -origin_exclusion.unsqueeze(0))
        overlaps_origin = overlaps_origin_per_dim.all(dim=1)
        valid_mask = ~overlaps_origin

        lbs = lbs[valid_mask]
        ubs = ubs[valid_mask]
        return lbs, ubs

    def is_rho_certified(self, rho: float) -> bool:
        """Check whether all regions satisfy Lyapunov conditions at ``rho``.

        Parameters
        ----------
        rho : float
            Lyapunov level-set value to test.

        Returns
        -------
        bool
            ``True`` if certification succeeds for all regions.
        """
        result = self._certify_regions(rho=rho, collect_details=False) 
        return result.success

    def certify(self, rho_estimate: float) -> tuple[float, RegionCertificationResult]:
        """Search for the largest certifiable rho and return details.

        Parameters
        ----------
        rho_estimate : float
            Positive initial guess for rho search.

        Returns
        -------
        tuple[float, RegionCertificationResult]
            Best certified ``rho`` and detailed region-level certification result.

        Raises
        ------
        ValueError
            If ``rho_estimate`` is not positive.
        """
        if rho_estimate <= 0:
            raise ValueError("rho_estimate must be positive.")

        __logger__.info("Starting Lyapunov certification with %s method.", self.config.cert_method.upper())

        self.verifier = self._setup_verifier()
        self.regions = self.build_regions()
        __logger__.info("Built %d certification regions (state_dim=%d).", len(self.regions[0]), self.config.state_dim)
        self.setup_backend()

        if self.config.rho_min > rho_estimate:
            __logger__.warning(
                "Provided rho_estimate (%.4f) is below rho_min (%.4f). Starting search from rho_min.", 
                rho_estimate, self.config.rho_min
            )
            initial_rho = self.config.rho_min
        else:
            initial_rho = float(rho_estimate)

        initial_ok = self.is_rho_certified(rho=initial_rho)

        # Scale up
        if initial_ok:
            rho_lo, rho_up = self._iterative_rho_search(
                total=self.config.max_scale_steps,
                desc="Scale up",
                initial_values=(initial_rho, initial_rho),
                step_fn=self._scale_rho_up
            )
            
            if rho_lo == rho_up:
                __logger__.warning("Maximum scaling steps reached without finding an upper bound.")
                return rho_lo, self._certify_regions(rho=rho_lo, collect_details=True)

        # Scale down
        else:
            rho_lo, rho_up = self._iterative_rho_search(
                total=self.config.max_scale_steps,
                desc="Scale down",
                initial_values=(None, initial_rho),
                step_fn=self._scale_rho_down
            )

            # Fallback
            if rho_lo is None:
                if not self.is_rho_certified(rho=self.config.rho_min):
                    __logger__.error("Could not even certify rho_min (%.4f).", self.config.rho_min)
                    return self.config.rho_min, self._certify_regions(rho=self.config.rho_min, collect_details=True)
                
                rho_lo = self.config.rho_min
                if rho_up <= rho_lo:
                    rho_up = rho_lo * self.config.rho_scaling

        # Bisect
        rho_lo, rho_up = self._iterative_rho_search(
            total=self.config.max_bisection_steps,
            desc="Bisect rho",
            initial_values=(rho_lo, rho_up),
            step_fn=self._bisect_rho
        )

        details = self._certify_regions(rho=rho_lo, collect_details=True)
        return rho_lo, details
