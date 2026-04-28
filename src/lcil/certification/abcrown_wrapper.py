from __future__ import annotations
import logging
from time import perf_counter
from typing import Any

import torch as th
import torch.nn as nn

from abcrown import (
    ABCrownSolver, 
    VerificationSpec, 
    ConfigBuilder, 
    input_vars, 
    output_vars
)

from .certifier_base import BaseCertifier
from .models import LyapunovMultiOutputVerifier

__logger__ = logging.getLogger(__name__)

class _ABCrownModelWrapper(nn.Module):
    """
    A wrapper that freezes the dynamic rho parameter
    so that ABCrownSolver can evaluate a clean model x -> y.
    """
    def __init__(self, verifier: nn.Module, device: th.device):
        super().__init__()
        self.verifier = verifier
        self.register_buffer("rho", th.tensor(0.0, dtype=th.float32, device=device))

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.verifier(x, self.rho)

class ABCrownCertifier(BaseCertifier):
    """
    Lyapunov certifier using the full Alpha-Beta-CROWN framework.

    The main rho-search path delegates input-space branching to ABCrown's
    internal ``input_bab`` machinery. External recursive splitting is kept only
    for the diagnostic detail path in ``BaseCertifier.collect_certification_details``.
    """

    def __init__(
        self,
        policy_model,
        lyap_model,
        dyn_model,
        config,
        device: th.device = th.device("cpu"),
    ):
        super().__init__(policy_model, lyap_model, dyn_model, config, device)
        self.abcrown_config = None
        self.wrapped_model = None

    def setup_backend(self) -> None:
        """Set up the ABCrownSolver and its configuration."""
        self.abcrown_config = (
            ConfigBuilder.from_defaults()
            .set(general__device=self.device.type)
            .set(general__complete_verifier="input_bab")
            .set(general__enable_incomplete_verification=False)
            .set(solver__batch_size=int(self.config.batch_size))
            .set(solver__bound_prop_method="crown")
            .set(bab__branching__method="sb")
            .set(bab__branching__input_split__enable=True)
            .set(bab__branching__input_split__ibp_enhancement=True)
            .set(bab__branching__input_split__compare_with_old_bounds=True)
            .set(bab__branching__input_split__adv_check=-1)
            .set(bab__branching__input_split__split_partitions=2)
            .set(attack__pgd_order="before") # TODO: maybe switch to "skip" or "after" if PGD is found to be too aggressive in finding counterexamples
            .set(bab__decision_thresh=-float(self.config.condition_tolerance))
            ()
        )

        __logger__.info(
            "Configured ABCrown backend with solver_batch_size=%d on device=%s.",
            int(self.config.batch_size),
            self.device.type,
        )
        
        self.verifier = self._setup_verifier()
        self.wrapped_model = _ABCrownModelWrapper(self.verifier, self.device)
        self.wrapped_model.eval()

    def _build_regions(self) -> th.Tensor:
        """Build only the minimal root decomposition needed for the origin hole."""
        outer_lb = self.bounds[0]
        outer_ub = self.bounds[1]
        origin_exclusion = self._resolve_origin_exclusion()

        inner_lb = th.maximum(outer_lb, -origin_exclusion)
        inner_ub = th.minimum(outer_ub, origin_exclusion)
        has_hole = bool((inner_ub > inner_lb).all().item())

        if not has_hole:
            __logger__.info(
                "Origin exclusion resolved to %s; built 1 root region without additional hole split.",
                [float(value) for value in origin_exclusion.detach().cpu().tolist()],
            )
            return self._pack_regions(outer_lb.unsqueeze(0), outer_ub.unsqueeze(0))

        region_lbs: list[th.Tensor] = []
        region_ubs: list[th.Tensor] = []
        for dim in range(self.config.state_dim):
            prefix_lb = outer_lb.clone()
            prefix_ub = outer_ub.clone()
            if dim > 0:
                prefix_lb[:dim] = inner_lb[:dim]
                prefix_ub[:dim] = inner_ub[:dim]

            lower_lb = prefix_lb.clone()
            lower_ub = prefix_ub.clone()
            lower_ub[dim] = inner_lb[dim]
            if bool((lower_ub > lower_lb).all().item()):
                region_lbs.append(lower_lb)
                region_ubs.append(lower_ub)

            upper_lb = prefix_lb.clone()
            upper_ub = prefix_ub.clone()
            upper_lb[dim] = inner_ub[dim]
            if bool((upper_ub > upper_lb).all().item()):
                region_lbs.append(upper_lb)
                region_ubs.append(upper_ub)

        __logger__.info(
            "Origin exclusion resolved to %s; decomposed root box into %d regions around the centered exclusion box.",
            [float(value) for value in origin_exclusion.detach().cpu().tolist()],
            len(region_lbs),
        )
        return self._pack_regions(th.stack(region_lbs, dim=0), th.stack(region_ubs, dim=0))

    def is_rho_certified(self, rho: float) -> bool:
        """Check rho using only the root regions and ABCrown's internal input split.

        Unlike LiRPA, ABCrown already performs input-space branching internally.
        For rho search we therefore only certify the root regions returned by
        ``_build_regions`` and allow ABCrown to split within each root box.
        External recursive splitting is reserved for the diagnostic path that
        collects region details.
        """
        if self.regions is None:
            raise RuntimeError("Certification regions are not initialized.")

        candidate_regions = self.regions
        if self.config.use_ibp_filter and self.negative_filter is not None:
            candidate_regions, _ = self._filter_sublevel_regions(candidate_regions, rho)

        if len(candidate_regions) == 0:
            __logger__.info(
                "ABCrown root certification at rho=%.6f is vacuous: all root regions were proven outside V(x) <= rho.",
                float(rho),
            )
            return False

        return self._solve_root_regions_batched(candidate_regions, rho)

    @staticmethod
    def _is_verified_status(status: str) -> bool:
        normalized = str(status).strip().lower()
        return normalized == "verified" or normalized.startswith("safe")

    def _setup_verifier(self):
        lbx_batched = self.bounds[0].unsqueeze(0)
        ubx_batched = self.bounds[1].unsqueeze(0)
        
        return LyapunovMultiOutputVerifier(
            policy_model=self.policy_model,
            lyap_model=self.lyap_model,
            dyn_model=self.dyn_model,
            lbx=lbx_batched,
            ubx=ubx_batched,
            kappa=self.config.kappa,
            sublevel_tolerance=self.config.sublevel_tolerance,
            condition_margin=self.config.condition_margin,
        )

    @staticmethod
    def _format_region_bounds(lb: th.Tensor, ub: th.Tensor) -> str:
        """Return a compact string representation of a region box."""
        lb_str = ", ".join(f"{float(value):.4g}" for value in lb.detach().cpu().tolist())
        ub_str = ", ".join(f"{float(value):.4g}" for value in ub.detach().cpu().tolist())
        return f"lb=[{lb_str}], ub=[{ub_str}]"

    @staticmethod
    def _format_state_vector(x: th.Tensor) -> str:
        values = x.detach().cpu().reshape(-1).tolist()
        return "[" + ", ".join(f"{float(value):.4g}" for value in values) + "]"

    def _coerce_counterexample_candidates(self, candidates: Any) -> th.Tensor | None:
        if candidates is None:
            return None

        if isinstance(candidates, dict):
            for value in candidates.values():
                points = self._coerce_counterexample_candidates(value)
                if points is not None:
                    return points
            return None

        if isinstance(candidates, (list, tuple)):
            for value in candidates:
                points = self._coerce_counterexample_candidates(value)
                if points is not None:
                    return points
            return None

        try:
            points = th.as_tensor(candidates, dtype=th.float32, device=self.device)
        except (TypeError, ValueError, RuntimeError):
            return None

        if points.ndim == 0 or points.numel() < self.config.state_dim:
            return None

        if points.shape[-1] == self.config.state_dim:
            return points.reshape(-1, self.config.state_dim)

        if points.numel() % self.config.state_dim != 0:
            return None

        return points.reshape(-1, self.config.state_dim)

    @staticmethod
    def _find_containing_region_index(
        point: th.Tensor,
        lbs: th.Tensor,
        ubs: th.Tensor,
        tolerance: float = 1e-6,
    ) -> int | None:
        tol = float(tolerance)
        inside = ((point.unsqueeze(0) >= (lbs - tol)) & (point.unsqueeze(0) <= (ubs + tol))).all(dim=1)
        matches = inside.nonzero(as_tuple=False).flatten()
        if matches.numel() == 0:
            return None
        return int(matches[0].item())

    def _summarize_counterexample_violation(
        self,
        points: th.Tensor,
        lbs: th.Tensor,
        ubs: th.Tensor,
        rho: float,
    ) -> str | None:
        if points.numel() == 0:
            return None

        rho_tensor = th.full(
            (points.shape[0], 1),
            float(rho),
            dtype=points.dtype,
            device=points.device,
        )
        with th.no_grad():
            outputs = self.verifier(points, rho_tensor)

        decrease_margin = outputs[:, 0]
        v_curr = outputs[:, 1]
        x_next = outputs[:, 2:]

        sublevel_threshold = float(rho + self.config.sublevel_tolerance)
        condition_threshold = -float(self.config.condition_tolerance)
        bound_tolerance = float(self.config.condition_tolerance)

        outside_sublevel = v_curr > sublevel_threshold
        positivity_ok = v_curr > condition_threshold
        decrease_ok = decrease_margin > condition_threshold

        global_lb = self.bounds[0].to(device=points.device, dtype=points.dtype)
        global_ub = self.bounds[1].to(device=points.device, dtype=points.dtype)
        lower_slack = x_next - (global_lb - bound_tolerance)
        upper_slack = (global_ub + bound_tolerance) - x_next
        invariance_ok = (lower_slack >= 0.0).all(dim=1) & (upper_slack >= 0.0).all(dim=1)

        safe = outside_sublevel | (positivity_ok & decrease_ok & invariance_ok)
        failing = (~safe).nonzero(as_tuple=False).flatten()

        if failing.numel() == 0:
            return (
                "returned witness could not be reproduced against the current verifier: "
                f"x={self._format_state_vector(points[0])}"
            )

        idx = int(failing[0].item())
        reason_parts: list[str] = []

        if not bool(positivity_ok[idx].item()):
            reason_parts.append("positivity")
        if not bool(decrease_ok[idx].item()):
            reason_parts.append("decrease")

        lower_failed_dims = (lower_slack[idx] < 0.0).nonzero(as_tuple=False).flatten().tolist()
        upper_failed_dims = (upper_slack[idx] < 0.0).nonzero(as_tuple=False).flatten().tolist()
        if lower_failed_dims:
            reason_parts.append(f"x_next<lb dims={lower_failed_dims}")
        if upper_failed_dims:
            reason_parts.append(f"x_next>ub dims={upper_failed_dims}")

        region_idx = self._find_containing_region_index(points[idx], lbs, ubs)
        region_label = "?" if region_idx is None else f"{region_idx + 1}/{len(lbs)}"
        failure = ", ".join(reason_parts) if reason_parts else "unclassified"

        return (
            f"region={region_label}, x={self._format_state_vector(points[idx])}, "
            f"V={float(v_curr[idx].item()):.6g}, rho={float(rho):.6g}, "
            f"decrease_margin={float(decrease_margin[idx].item()):.6g}, "
            f"x_next={self._format_state_vector(x_next[idx])}, failure={failure}"
        )

    def _get_abcrown_config_value(self, *path: str, default: Any = None) -> Any:
        value: Any = self.abcrown_config
        for key in path:
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return value

    def _log_solver_diagnostics(
        self,
        res: Any,
        lbs: th.Tensor,
        ubs: th.Tensor,
        rho: float,
    ) -> None:
        stats = getattr(res, "stats", {}) or {}
        reference = getattr(res, "reference", {}) or {}
        bab_stats = stats.get("bab") or []
        complete_verifier = self._get_abcrown_config_value(
            "general",
            "complete_verifier",
            default="unknown",
        )
        input_split_enabled = bool(
            self._get_abcrown_config_value(
                "bab",
                "branching",
                "input_split",
                "enable",
                default=False,
            )
        )
        pgd_order = self._get_abcrown_config_value("attack", "pgd_order", default="before")
        status = getattr(res, "status", "unknown")

        if bab_stats:
            _, global_lb, visited_domains, bab_elapsed = bab_stats[-1]
            __logger__.info(
                "ABCrown complete verifier entered %s with input_split=%s, visited_domains=%s, global_lb=%.6g, bab_time=%.2fs.",
                complete_verifier,
                input_split_enabled,
                int(visited_domains),
                float(global_lb),
                float(bab_elapsed),
            )
        elif str(status).strip().lower() == "unsafe-pgd":
            __logger__.info(
                "ABCrown did not enter %s; PGD found a counterexample first (input_split_configured=%s, pgd_order=%s).",
                complete_verifier,
                input_split_enabled,
                pgd_order,
            )
        else:
            __logger__.info(
                "ABCrown complete verifier did not enter %s; input_split_configured=%s, status=%s, pgd_order=%s.",
                complete_verifier,
                input_split_enabled,
                status,
                pgd_order,
            )

        if self._is_verified_status(status):
            return

        candidate_sources = (
            ("attack_examples", stats.get("attack_examples")),
            ("all_adv_candidates", stats.get("all_adv_candidates")),
            ("reference.attack_examples", reference.get("attack_examples")),
        )
        for source_name, raw_candidates in candidate_sources:
            points = self._coerce_counterexample_candidates(raw_candidates)
            if points is None:
                continue

            summary = self._summarize_counterexample_violation(points, lbs, ubs, rho)
            if summary is None:
                continue

            __logger__.info("ABCrown violation witness from %s: %s", source_name, summary)
            return

        __logger__.info(
            "ABCrown returned status=%s without an inspectable counterexample payload.",
            status,
        )

    def _solve_box_with_model(
        self,
        lb: th.Tensor,
        ub: th.Tensor,
    ) -> bool:
        if lb.ndim > 1:
            lb = lb.squeeze(0)
            ub = ub.squeeze(0)

        rho_value = float(self.wrapped_model.rho.item())

        with self._get_suppress_ctx():
            x = input_vars(self.config.state_dim)
            y = output_vars(2 + self.config.state_dim)

            input_constraint = (x >= lb) & (x <= ub)
            output_constraint = self._build_safe_output_constraint(
                y=y,
                rho=rho_value,
            )

            spec = VerificationSpec.build_spec(
                input_vars=x,
                output_vars=y,
                input_constraint=input_constraint,
                output_constraint=output_constraint,
            )

            solver = ABCrownSolver(
                spec=spec,
                computing_graph=self.wrapped_model,
                config=self.abcrown_config,
            )

            res = solver.solve()

        return self._is_verified_status(res.status)

    def _build_safe_output_constraint(
        self,
        y,
        rho: float,
    ):
        """Build the safe-region predicate for the multi-output verifier.

        The safe condition enforces the paper-style implication over the
        global certification box ``B``:

        ``V(x) > rho + tol_sublevel`` OR
        ``(V(x) >= -tol_cond) AND (decrease margin >= -tol_cond) AND (x_next in B)``.
        """
        safe_outside_sublevel = y[1] > (rho + self.config.sublevel_tolerance)
        safe_positive = y[1] > (-self.config.condition_tolerance)
        safe_decrease = y[0] > (-self.config.condition_tolerance)

        global_lb = self.bounds[0]
        global_ub = self.bounds[1]
        safe_x_next = None
        for idx in range(self.config.state_dim):
            coord_safe = (y[idx + 2] > (float(global_lb[idx]) - self.config.condition_tolerance)) & (
                y[idx + 2] < (float(global_ub[idx]) + self.config.condition_tolerance)
            )
            safe_x_next = coord_safe if safe_x_next is None else (safe_x_next & coord_safe)

        return safe_outside_sublevel | (safe_positive & safe_decrease & safe_x_next)

    def _build_unsafe_output_clauses(
        self,
        rho: float,
    ) -> list[tuple[th.Tensor, th.Tensor]]:
        """Build bounds-mode clauses describing the unsafe output set.

        The current safe predicate is

        ``V(x) > rho + tol_sublevel`` OR
        ``(V(x) > -tol_cond) AND (dV(x) > -tol_cond) AND (x_next in B)``.

        ABCrown's bounds mode expects the negated unsafe set as an OR-of-ANDs
        over linear output constraints. Each returned tuple corresponds to one
        unsafe clause ``C y <= rhs``.
        """
        output_dim = 2 + self.config.state_dim
        sublevel_threshold = float(rho + self.config.sublevel_tolerance)
        condition_threshold = -float(self.config.condition_tolerance)
        global_lb = self.bounds[0]
        global_ub = self.bounds[1]

        def _row(index: int, coefficient: float, rhs: float) -> tuple[th.Tensor, th.Tensor]:
            coeffs = th.zeros((1, output_dim), dtype=th.float32)
            coeffs[0, index] = float(coefficient)
            rhs_tensor = th.tensor([float(rhs)], dtype=th.float32)
            return coeffs, rhs_tensor

        sublevel_row = _row(1, 1.0, sublevel_threshold)
        clauses: list[tuple[th.Tensor, th.Tensor]] = []

        failure_rows = [
            _row(1, 1.0, condition_threshold),
            _row(0, 1.0, condition_threshold),
        ]
        for idx in range(self.config.state_dim):
            failure_rows.append(
                _row(idx + 2, 1.0, float(global_lb[idx]) - float(self.config.condition_tolerance))
            )
            failure_rows.append(
                _row(idx + 2, -1.0, -(float(global_ub[idx]) + float(self.config.condition_tolerance)))
            )

        for failure_c, failure_rhs in failure_rows:
            clause_c = th.cat([sublevel_row[0], failure_c], dim=0)
            clause_rhs = th.cat([sublevel_row[1], failure_rhs], dim=0)
            clauses.append((clause_c, clause_rhs))

        return clauses

    def _solve_root_regions_batched(
        self,
        bs: th.Tensor,
        rho: float,
    ) -> bool:
        """Solve all root regions in one ABCrown bounds-mode call.

        This path is used for rho search and intentionally avoids extra wrapper-
        side splitting or conservative pre-verification. The only external
        decomposition is the origin-hole root partition built by ``_build_regions``.
        """
        if self.wrapped_model is None or self.abcrown_config is None:
            raise RuntimeError("ABCrownCertifier backend is not properly initialized.")

        num_regions = len(bs)
        if num_regions == 0:
            return False

        __logger__.info(
            "ABCrown root certification start: rho=%.6f, %d root regions in one batched solve.",
            float(rho),
            num_regions,
        )
        lbs, ubs = self._unpack_regions(bs)
        self.wrapped_model.rho.fill_(rho)
        clauses = self._build_unsafe_output_clauses(rho)

        solve_start = perf_counter()
        with self._get_suppress_ctx():
            spec = VerificationSpec.build_spec(
                lower=lbs,
                upper=ubs,
                clauses=clauses,
            )
            solver = ABCrownSolver(
                spec=spec,
                computing_graph=self.wrapped_model,
                config=self.abcrown_config,
            )
            res = solver.solve()

        solve_elapsed = perf_counter() - solve_start
        is_verified = self._is_verified_status(res.status)
        __logger__.info(
            "ABCrown root batched solve finished in %.2fs with status=%s.",
            solve_elapsed,
            res.status,
        )
        self._log_solver_diagnostics(res=res, lbs=lbs, ubs=ubs, rho=rho)
        return is_verified


    def _certify_batched_regions(
            self,
            bs: th.Tensor,
            rho: float,
            early_exit: bool = True,
        ) -> th.Tensor:
        """
        Certifies a batch of regions using the ABCrown solver.

        Parameters
        ----------
        bs : th.Tensor
            Packed region bounds with shape ``(n, 2, state_dim)``.
        rho : float
            The rho parameter for the certification.
        early_exit : bool, optional
            Whether to exit early if a region is not certified. Defaults to True.

        Returns
        -------
        th.Tensor
            Boolean tensor indicating whether each region is certified.
        """
        if self.wrapped_model is None or self.abcrown_config is None:
            raise RuntimeError("ABCrownCertifier backend is not properly initialized.")
        
        num_regions = len(bs)
        if num_regions == 0:
            return th.empty((0,), dtype=th.bool, device=self.device)

        __logger__.info(
            "ABCrown diagnostic batch start: rho=%.6f, %d regions.",
            float(rho),
            num_regions,
        )

        is_certified = th.zeros(num_regions, dtype=th.bool, device=self.device)
        lbs, ubs = self._unpack_regions(bs)
        self.wrapped_model.rho.fill_(rho)

        abcrown_start = perf_counter()
        total_remaining = num_regions

        for position in range(num_regions):
            lb = lbs[position]
            ub = ubs[position]
            solve_start = perf_counter()
            is_certified[position] = self._solve_box_with_model(lb=lb, ub=ub)
            solve_elapsed = perf_counter() - solve_start

            if not bool(is_certified[position].item()):
                __logger__.info(
                    "ABCrown diagnostic region %d/%d failed in %.2fs: %s",
                    position + 1,
                    total_remaining,
                    solve_elapsed,
                    self._format_region_bounds(lb, ub),
                )
            
            if early_exit and not bool(is_certified[position].item()):
                __logger__.info(
                    "ABCrown early exit after region %d/%d failed. Certified overall: %d/%d.",
                    position + 1,
                    total_remaining,
                    int(is_certified.sum().item()),
                    num_regions,
                )
                break

        __logger__.info(
            "ABCrown batch complete in %.2fs: %d/%d regions certified.",
            perf_counter() - abcrown_start,
            int(is_certified.sum().item()),
            num_regions,
        )

        return is_certified