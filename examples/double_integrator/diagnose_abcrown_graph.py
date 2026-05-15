from __future__ import annotations

import argparse
import logging
import tempfile

from dataclasses import dataclass
from pathlib import Path

import torch as th

from auto_LiRPA import BoundedModule

from lcil.certification.models import LyapunovCoreVerifier
from lcil.lyapunov_learning import LyapunovTrainingConfig
from lcil.utils.base_config import ArgumentParserConfig, config_field

from . import (
    DoubleIntegratorDynamics,
    discover_latest_lyapunov_dir,
    discover_latest_policy_dir,
    load_lyapunov_model,
    load_policy_model,
)

__logger__ = logging.getLogger("lcil.examples.double_integrator.diagnose_abcrown_graph")

_DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results" / "double_integrator"


class _LogCaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@dataclass(frozen=True)
class DiagnoseABCrownScriptConfig(ArgumentParserConfig):
    policy_dir: str = config_field(help="Policy run directory containing model.pt.")
    lyapunov_dir: str = config_field(help="Lyapunov run directory containing lyapunov_model.pt.")
    device: str = config_field(default="cpu", help="Torch device string used for the BoundedModule conversion attempt.")
    include_onnx_ops: bool = config_field(
        default=True,
        help="Whether to export the verifier once to ONNX and summarize node types.",
    )
    fail_on_conversion_error: bool = config_field(
        default=False,
        help="Return a non-zero exit code when BoundedModule conversion fails.",
    )


def _build_script_defaults() -> DiagnoseABCrownScriptConfig:
    default_policy_dir = discover_latest_policy_dir(_DEFAULT_RESULTS_ROOT)
    default_lyapunov_dir = discover_latest_lyapunov_dir(default_policy_dir)
    return DiagnoseABCrownScriptConfig(
        policy_dir=str(default_policy_dir),
        lyapunov_dir=str(default_lyapunov_dir),
    )


def parse_args() -> DiagnoseABCrownScriptConfig:
    script_defaults = _build_script_defaults()
    parser = argparse.ArgumentParser(
        description="Diagnose ABCrown/auto_LiRPA compatibility of the double-integrator closed-loop verifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    script_defaults.add_to_argparse(parser)
    args = parser.parse_args()
    return script_defaults.from_namespace(args)


def _configure_logging_capture() -> tuple[_LogCaptureHandler, list[tuple[logging.Logger, int]]]:
    capture_handler = _LogCaptureHandler()
    target_loggers = [
        logging.getLogger(),
        logging.getLogger("auto_LiRPA"),
        logging.getLogger("auto_LiRPA.bound_general"),
        logging.getLogger("abcrown"),
        logging.getLogger("abcrown.auto_LiRPA"),
    ]

    previous_levels: list[tuple[logging.Logger, int]] = []
    for logger in target_loggers:
        previous_levels.append((logger, logger.level))
        logger.addHandler(capture_handler)
        logger.setLevel(logging.DEBUG)
    return capture_handler, previous_levels


def _restore_logging_capture(
    capture_handler: _LogCaptureHandler,
    previous_levels: list[tuple[logging.Logger, int]],
) -> None:
    for logger, previous_level in previous_levels:
        logger.removeHandler(capture_handler)
        logger.setLevel(previous_level)


def _extract_unsupported_ops(messages: list[str]) -> list[str]:
    unsupported_ops: list[str] = []
    for message in messages:
        if "The node has an unsupported operation:" in message:
            unsupported_ops.append(message.split("operation:", maxsplit=1)[1].strip())
            continue
        if message.startswith("Name: "):
            unsupported_ops.append(message.strip())

    seen: set[str] = set()
    unique_unsupported_ops: list[str] = []
    for op in unsupported_ops:
        if op in seen:
            continue
        seen.add(op)
        unique_unsupported_ops.append(op)
    return unique_unsupported_ops


def _trace_graph_node_kinds(model: th.nn.Module, x: th.Tensor) -> list[str]:
    traced = th.jit.trace(model, x, strict=False)
    return sorted({node.kind() for node in traced.inlined_graph.nodes()})


def _export_onnx_node_types(model: th.nn.Module, x: th.Tensor) -> list[str]:
    import onnx

    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_path = Path(tmp_dir) / "diagnose_abcrown_graph.onnx"
        th.onnx.export(
            model,
            x,
            onnx_path,
            input_names=["x"],
            output_names=["y"],
            opset_version=17,
            do_constant_folding=True,
        )
        onnx_model = onnx.load(onnx_path)
    return sorted({node.op_type for node in onnx_model.graph.node})


def main() -> int:
    script_config = parse_args()
    device = th.device(script_config.device)

    policy_dir = Path(script_config.policy_dir).resolve()
    lyapunov_dir = Path(script_config.lyapunov_dir).resolve()

    policy_model = load_policy_model(policy_dir, device)
    lyap_model = load_lyapunov_model(lyapunov_dir, device)
    training_config = LyapunovTrainingConfig.load(lyapunov_dir / "training_config.json")

    dyn_model = DoubleIntegratorDynamics(
        dt=policy_model.global_config.dt,
        abcrown_compatible_ops=True,
    ).to(device)
    dyn_model.eval()

    verifier = LyapunovCoreVerifier(
        policy_model=policy_model,
        lyap_model=lyap_model,
        dyn_model=dyn_model,
        kappa=float(training_config.kappa),
        condition_margin=float(training_config.condition_margin),
    ).to(device)
    verifier.eval()

    x = th.zeros((1, int(policy_model.global_config.nx)), dtype=th.float32, device=device)

    __logger__.info("Policy checkpoint: %s", policy_dir)
    __logger__.info("Lyapunov checkpoint: %s", lyapunov_dir)
    __logger__.info(
        "Verifier summary: policy=%s lyapunov=%s state_dim=%d input_shape=%s device=%s.",
        type(policy_model).__name__,
        type(lyap_model).__name__,
        int(policy_model.global_config.nx),
        tuple(x.shape),
        device,
    )
    __logger__.info(
        "Policy metadata: max_seq_len=%s output_mode=%s.",
        getattr(policy_model, "max_seq_len", "n/a"),
        getattr(policy_model, "output_mode", "n/a"),
    )

    capture_handler, previous_levels = _configure_logging_capture()
    conversion_error: Exception | None = None
    try:
        BoundedModule(verifier, x, device=device)
    except Exception as exc:  # noqa: BLE001 - diagnosis script should surface the actual verifier failure.
        conversion_error = exc
    finally:
        _restore_logging_capture(capture_handler, previous_levels)

    unsupported_ops = _extract_unsupported_ops(capture_handler.messages)

    if conversion_error is None:
        __logger__.info("BoundedModule conversion succeeded.")
    else:
        __logger__.error(
            "BoundedModule conversion failed with %s: %s",
            type(conversion_error).__name__,
            conversion_error,
        )
        if unsupported_ops:
            __logger__.error("Unsupported ops reported by auto_LiRPA:")
            for op in unsupported_ops:
                __logger__.error("  %s", op)
        else:
            __logger__.error("No explicit unsupported-op log lines were captured.")

    verifier_cpu = verifier.to("cpu")
    x_cpu = x.detach().cpu()

    try:
        graph_node_kinds = _trace_graph_node_kinds(verifier_cpu, x_cpu)
        __logger__.info("TorchScript graph node kinds (%d): %s", len(graph_node_kinds), graph_node_kinds)
    except Exception as exc:  # noqa: BLE001 - diagnosis should continue even if tracing fails.
        __logger__.warning("TorchScript tracing failed with %s: %s", type(exc).__name__, exc)

    if script_config.include_onnx_ops:
        try:
            onnx_node_types = _export_onnx_node_types(verifier_cpu, x_cpu)
            __logger__.info("ONNX node types (%d): %s", len(onnx_node_types), onnx_node_types)
        except Exception as exc:  # noqa: BLE001 - optional diagnostics should not mask the primary result.
            __logger__.warning("ONNX export failed with %s: %s", type(exc).__name__, exc)

    if conversion_error is not None and script_config.fail_on_conversion_error:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())