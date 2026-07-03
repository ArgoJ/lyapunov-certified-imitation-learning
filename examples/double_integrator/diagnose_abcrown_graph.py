from __future__ import annotations

import argparse
import logging
import tempfile
import numpy as np

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch as th

from auto_LiRPA import BoundedModule
from mpc_datagen import MPCConfig

from lcil.imitation_learning import BoundedPolicy, TransformerPolicy
from lcil.certification.models import LyapunovCoreVerifier
from lcil.lyapunov_learning import NeuralLyapunovCandidate, QuadraticLyapunovCandidate
from lcil.utils.base_models import MLP
from lcil.utils.base_config import ArgumentParserConfig, config_field

from .double_integrator_dyn import DoubleIntegratorDynamics

__logger__ = logging.getLogger("lcil.examples.double_integrator.diagnose_abcrown_graph")

_SUSPICIOUS_TORCHSCRIPT_OP_KEYWORDS = (
    "scaled_dot_product_attention",
    "unflatten",
    "If",
    "Mod",
)
_SUSPICIOUS_ONNX_OP_TYPES = ("If", "Mod")



class _LogCaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@dataclass(frozen=True)
class _TraceNodeLocation:
    kind: str
    scope: str
    source: str
    matched_modules: tuple[str, ...]


@dataclass(frozen=True)
class _OnnxNodeLocation:
    op_type: str
    path: str
    name: str
    source: str
    matched_modules: tuple[str, ...]


@dataclass(frozen=True)
class DiagnoseABCrownScriptConfig(ArgumentParserConfig):
    device: str = config_field(default="cpu", help="Torch device string used for the BoundedModule conversion attempt.")
    policy_arch: Literal["transformer", "mlp"] = config_field(
        default="transformer",
        help="Policy architecture used when model_source=fresh.",
    )
    lyapunov_arch: Literal["neural", "quadratic"] = config_field(
        default="neural",
        help="Lyapunov architecture used when model_source=fresh.",
    )
    state_dim: int = config_field(default=2, help="State dimension for fresh models.")
    control_dim: int = config_field(default=1, help="Control dimension for fresh models.")
    dt: float = config_field(default=0.1, help="Sampling time used for fresh dynamics and policy metadata.")
    state_abs_bound: float = config_field(default=1.0, help="Symmetric state bound used in the synthetic MPCConfig for fresh models.")
    control_abs_bound: float = config_field(default=1.0, help="Symmetric control bound used in the synthetic MPCConfig for fresh models.")
    kappa: float = config_field(default=0.01, help="Lyapunov decay factor used for fresh verifier construction.")
    condition_margin: float = config_field(default=1e-5, help="Lyapunov verifier condition margin used for fresh models.")
    policy_hidden_sizes: tuple[int, ...] = config_field(
        default=(32, 32),
        help="Hidden layer sizes for a fresh MLP policy.",
    )
    policy_activation: str = config_field(
        default="relu",
        help="Activation used in fresh policy feedforward blocks.",
    )
    policy_dropout: float = config_field(
        default=0.0,
        help="Dropout used in fresh policy models.",
    )
    policy_normalization: Literal["none", "layer_norm"] = config_field(
        default="none",
        help="Hidden normalization used in a fresh MLP policy.",
    )
    transformer_d_model: int = config_field(default=32, help="Embedding width for a fresh transformer policy.")
    transformer_nhead: int = config_field(default=4, help="Number of attention heads for a fresh transformer policy.")
    transformer_num_layers: int = config_field(default=2, help="Number of encoder layers for a fresh transformer policy.")
    transformer_dim_feedforward: int = config_field(default=64, help="Feedforward width inside a fresh transformer policy.")
    transformer_max_seq_len: int = config_field(default=5, help="Maximum sequence length for a fresh transformer policy.")
    transformer_causal: bool = config_field(default=True, help="Whether the fresh transformer policy uses a causal mask.")
    transformer_output_mode: Literal["last", "per_step"] = config_field(
        default="last",
        help="Output reduction mode for a fresh transformer policy.",
    )
    lyapunov_hidden_sizes: tuple[int, ...] = config_field(
        default=(32, 32),
        help="Hidden layer sizes for the fresh neural Lyapunov feature net.",
    )
    lyapunov_activation: str = config_field(
        default="relu",
        help="Activation used in the fresh neural Lyapunov feature net.",
    )
    lyapunov_dropout: float = config_field(
        default=0.0,
        help="Dropout used in the fresh neural Lyapunov feature net.",
    )
    lyapunov_normalization: Literal["none", "layer_norm"] = config_field(
        default="none",
        help="Hidden normalization used in the fresh neural Lyapunov feature net.",
    )
    lyapunov_eps: float = config_field(default=1e-3, help="Positive-definite epsilon used in fresh Lyapunov models.")
    include_onnx_ops: bool = config_field(
        default=True,
        help="Whether to export the verifier once to ONNX and summarize node types.",
    )
    fail_on_conversion_error: bool = config_field(
        default=False,
        help="Return a non-zero exit code when BoundedModule conversion fails.",
    )


def _build_script_defaults() -> DiagnoseABCrownScriptConfig:
    return DiagnoseABCrownScriptConfig()


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


def _get_trace_graph(model: th.nn.Module, x: th.Tensor):
    get_trace_graph = getattr(th.jit, "_get_trace_graph", None)
    if callable(get_trace_graph):
        graph, _ = get_trace_graph(model, (x,))
        return graph

    traced = th.jit.trace(model, x, strict=False)
    graph_for = getattr(traced, "graph_for", None)
    if callable(graph_for):
        return graph_for(x)

    graph = getattr(traced, "inlined_graph", None)
    if graph is None:
        graph = getattr(traced, "graph", None)
    if graph is None:
        raise AttributeError("The traced module does not expose an inlined_graph or graph attribute.")
    return graph


def _stringify_source_range(source_range: object | None) -> str:
    if source_range is None:
        return "<unknown>"
    source_text = str(source_range).strip()
    if not source_text:
        return "<unknown>"
    return source_text.splitlines()[0]


def _get_node_scope_name(node) -> str:
    scope_name = getattr(node, "scopeName", None)
    if callable(scope_name):
        resolved = scope_name()
        return resolved if resolved else "<unknown>"
    return "<unknown>"


def _get_node_source(node) -> str:
    source_range = getattr(node, "sourceRange", None)
    if callable(source_range):
        return _stringify_source_range(source_range())
    return _stringify_source_range(source_range)


def _matches_keyword(value: str, keywords: tuple[str, ...]) -> bool:
    normalized_value = value.lower()
    return any(keyword.lower() in normalized_value for keyword in keywords)


def _tokenize_scope(scope_text: str) -> list[str]:
    normalized_scope = (
        scope_text.replace("/", ".")
        .replace(":", ".")
        .replace("[", ".")
        .replace("]", ".")
    )
    return [
        token
        for token in normalized_scope.split(".")
        if token and token != "__module"
    ]


def _contains_token_subsequence(tokens: list[str], candidate: list[str]) -> bool:
    if not candidate or len(candidate) > len(tokens):
        return False
    candidate_length = len(candidate)
    for start_idx in range(len(tokens) - candidate_length + 1):
        if tokens[start_idx : start_idx + candidate_length] == candidate:
            return True
    return False


def _resolve_scope_modules(scope_text: str, model: th.nn.Module) -> tuple[str, ...]:
    scope_tokens = _tokenize_scope(scope_text)
    if not scope_tokens:
        return ()

    matches: list[tuple[int, str, str]] = []
    for module_name, module in model.named_modules():
        if not module_name:
            continue
        module_tokens = module_name.split(".")
        if _contains_token_subsequence(scope_tokens, module_tokens):
            matches.append((len(module_tokens), module_name, type(module).__name__))

    matches.sort(key=lambda item: (item[0], item[1]))
    return tuple(f"{module_name} ({module_type})" for _, module_name, module_type in matches[-3:])


def _trace_graph_node_kinds(model: th.nn.Module, x: th.Tensor) -> list[str]:
    graph = _get_trace_graph(model, x)
    return sorted({node.kind() for node in graph.nodes()})


def _trace_suspicious_graph_nodes(
    model: th.nn.Module,
    x: th.Tensor,
    suspicious_keywords: tuple[str, ...],
) -> list[_TraceNodeLocation]:
    graph = _get_trace_graph(model, x)
    suspicious_nodes: list[_TraceNodeLocation] = []
    for node in graph.nodes():
        kind = node.kind()
        if not _matches_keyword(kind, suspicious_keywords):
            continue

        scope = _get_node_scope_name(node)
        suspicious_nodes.append(
            _TraceNodeLocation(
                kind=kind,
                scope=scope,
                source=_get_node_source(node),
                matched_modules=_resolve_scope_modules(scope, model),
            )
        )
    return suspicious_nodes


def _export_onnx_model(model: th.nn.Module, x: th.Tensor):
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
        return onnx.load(onnx_path)


def _export_onnx_node_types(onnx_model) -> list[str]:
    return sorted({node.op_type for _, node in _iter_onnx_nodes(onnx_model.graph)})


def _normalize_unsupported_op_name(op_name: str) -> str:
    normalized = op_name.strip()
    if "op='" in normalized:
        normalized = normalized.split("op='", maxsplit=1)[1].split("'", maxsplit=1)[0].strip()
    if normalized.startswith("Name: "):
        normalized = normalized.split(",", maxsplit=1)[0].replace("Name:", "").strip()
    if "::" in normalized:
        normalized = normalized.split("::", maxsplit=1)[1]
    return normalized


def _iter_onnx_nodes(graph, prefix: str = "graph"):
    for node_idx, node in enumerate(graph.node):
        node_prefix = f"{prefix}[{node_idx}]"
        yield node_prefix, node
        for attr in node.attribute:
            if attr.type == attr.GRAPH:
                yield from _iter_onnx_nodes(attr.g, prefix=f"{node_prefix}.{attr.name}")
            elif attr.type == attr.GRAPHS:
                for graph_idx, subgraph in enumerate(attr.graphs):
                    yield from _iter_onnx_nodes(
                        subgraph,
                        prefix=f"{node_prefix}.{attr.name}[{graph_idx}]",
                    )


def _trace_suspicious_onnx_nodes(
    onnx_model,
    model: th.nn.Module,
    suspicious_op_types: tuple[str, ...],
) -> list[_OnnxNodeLocation]:
    suspicious_nodes: list[_OnnxNodeLocation] = []
    for node_path, node in _iter_onnx_nodes(onnx_model.graph):
        if not _matches_keyword(node.op_type, suspicious_op_types):
            continue

        node_name = node.name or (node.output[0] if node.output else "<unnamed>")
        suspicious_nodes.append(
            _OnnxNodeLocation(
                op_type=node.op_type,
                path=node_path,
                name=node_name,
                source=_stringify_source_range(node.doc_string),
                matched_modules=_resolve_scope_modules(node_name, model),
            )
        )
    return suspicious_nodes


def _format_matched_modules(matched_modules: tuple[str, ...]) -> str:
    if not matched_modules:
        return "<unknown>"
    return " -> ".join(matched_modules)


def _build_synthetic_global_config(script_config: DiagnoseABCrownScriptConfig) -> MPCConfig:
    global_config = MPCConfig(
        T_sim=10,
        N=5,
        nx=int(script_config.state_dim),
        nu=int(script_config.control_dim),
        dt=float(script_config.dt),
    )
    global_config.constraints.lbx = np.full((int(script_config.state_dim),), -float(script_config.state_abs_bound), dtype=float)
    global_config.constraints.ubx = np.full((int(script_config.state_dim),), float(script_config.state_abs_bound), dtype=float)
    global_config.constraints.lbu = np.full((int(script_config.control_dim),), -float(script_config.control_abs_bound), dtype=float)
    global_config.constraints.ubu = np.full((int(script_config.control_dim),), float(script_config.control_abs_bound), dtype=float)
    return global_config


def _build_fresh_policy_model(
    script_config: DiagnoseABCrownScriptConfig,
    device: th.device,
) -> tuple[BoundedPolicy | TransformerPolicy, MPCConfig]:
    global_config = _build_synthetic_global_config(script_config)

    if script_config.policy_arch == "mlp":
        hidden_sizes = tuple(int(size) for size in script_config.policy_hidden_sizes)
        layer_sizes = [int(script_config.state_dim), *hidden_sizes, int(script_config.control_dim)]
        activations = [str(script_config.policy_activation)] * len(hidden_sizes) + ["identity"]
        policy_model: BoundedPolicy | TransformerPolicy = BoundedPolicy(
            feature_net=MLP(
                layer_dims=layer_sizes,
                activations=activations,
                dropout=float(script_config.policy_dropout),
                normalization=str(script_config.policy_normalization),
            ),
            u_min=global_config.constraints.lbu,
            u_max=global_config.constraints.ubu,
        )
    else:
        policy_model = TransformerPolicy(
            input_dim=int(script_config.state_dim),
            output_dim=int(script_config.control_dim),
            d_model=int(script_config.transformer_d_model),
            nhead=int(script_config.transformer_nhead),
            num_encoder_layers=int(script_config.transformer_num_layers),
            dim_feedforward=int(script_config.transformer_dim_feedforward),
            dropout=float(script_config.policy_dropout),
            activation=str(script_config.policy_activation),
            max_seq_len=int(script_config.transformer_max_seq_len),
            causal=bool(script_config.transformer_causal),
            output_mode=str(script_config.transformer_output_mode),
            u_min=global_config.constraints.lbu,
            u_max=global_config.constraints.ubu,
        )

    return policy_model.to(device).eval(), global_config


def _build_fresh_lyapunov_model(
    script_config: DiagnoseABCrownScriptConfig,
    device: th.device,
) -> NeuralLyapunovCandidate | QuadraticLyapunovCandidate:
    state_dim = int(script_config.state_dim)
    lyapunov_eps = float(script_config.lyapunov_eps)

    if script_config.lyapunov_arch == "quadratic":
        lyap_model: NeuralLyapunovCandidate | QuadraticLyapunovCandidate = QuadraticLyapunovCandidate(
            state_dim=state_dim,
            eps=lyapunov_eps,
        )
        return lyap_model.to(device).eval()

    hidden_sizes = tuple(int(size) for size in script_config.lyapunov_hidden_sizes)
    layer_dims = [state_dim, *hidden_sizes, 1]
    activations = [str(script_config.lyapunov_activation)] * len(hidden_sizes) + ["identity"]
    feature_net = MLP(
        layer_dims=layer_dims,
        activations=activations,
        dropout=float(script_config.lyapunov_dropout),
        normalization=str(script_config.lyapunov_normalization),
    )
    lyap_model = NeuralLyapunovCandidate(
        feature_net=feature_net,
        state_dim=state_dim,
        eps=lyapunov_eps,
    )
    return lyap_model.to(device).eval()


def main() -> int:
    script_config = parse_args()
    device = th.device(script_config.device)

    policy_model, mpc_cfg = _build_fresh_policy_model(script_config, device)
    lyap_model = _build_fresh_lyapunov_model(script_config, device)
    kappa = float(script_config.kappa)
    condition_margin = float(script_config.condition_margin)
    policy_source = f"fresh:{script_config.policy_arch}"
    lyapunov_source = f"fresh:{script_config.lyapunov_arch}"

    dyn_model = DoubleIntegratorDynamics(
        dt=mpc_cfg.dt,
        abcrown_compatible_ops=True,
    ).to(device)
    dyn_model.eval()

    verifier = LyapunovCoreVerifier(
        policy_model=policy_model,
        lyap_model=lyap_model,
        dyn_model=dyn_model,
        kappa=kappa,
        condition_margin=condition_margin,
    ).to(device)
    verifier.eval()

    x = th.zeros((1, int(mpc_cfg.nx)), dtype=th.float32, device=device)

    __logger__.info("Policy source: %s", policy_source)
    __logger__.info("Lyapunov source: %s", lyapunov_source)
    __logger__.info(
        "Verifier summary: policy=%s lyapunov=%s state_dim=%d input_shape=%s device=%s.",
        type(policy_model).__name__,
        type(lyap_model).__name__,
        int(mpc_cfg.nx),
        tuple(x.shape),
        device,
    )
    __logger__.info(
        "Policy metadata: max_seq_len=%s output_mode=%s kappa=%.6f condition_margin=%.6g.",
        getattr(policy_model, "max_seq_len", "n/a"),
        getattr(policy_model, "output_mode", "n/a"),
        kappa,
        condition_margin,
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

        suspicious_graph_nodes = _trace_suspicious_graph_nodes(
            verifier_cpu,
            x_cpu,
            suspicious_keywords=_SUSPICIOUS_TORCHSCRIPT_OP_KEYWORDS,
        )
        if suspicious_graph_nodes:
            __logger__.info("Suspicious TorchScript nodes (%d):", len(suspicious_graph_nodes))
            for node in suspicious_graph_nodes:
                __logger__.info(
                    "  kind=%s | scope=%s | modules=%s | source=%s",
                    node.kind,
                    node.scope,
                    _format_matched_modules(node.matched_modules),
                    node.source,
                )
        else:
            __logger__.info("No suspicious TorchScript nodes matched %s.", _SUSPICIOUS_TORCHSCRIPT_OP_KEYWORDS)
    except Exception as exc:  # noqa: BLE001 - diagnosis should continue even if tracing fails.
        __logger__.warning("TorchScript tracing failed with %s: %s", type(exc).__name__, exc)

    if script_config.include_onnx_ops:
        try:
            onnx_model = _export_onnx_model(verifier_cpu, x_cpu)
            onnx_node_types = _export_onnx_node_types(onnx_model)
            __logger__.info("ONNX node types (%d): %s", len(onnx_node_types), onnx_node_types)

            suspicious_onnx_nodes = _trace_suspicious_onnx_nodes(
                onnx_model,
                verifier_cpu,
                suspicious_op_types=_SUSPICIOUS_ONNX_OP_TYPES,
            )
            if suspicious_onnx_nodes:
                __logger__.info("Suspicious ONNX nodes (%d):", len(suspicious_onnx_nodes))
                for node in suspicious_onnx_nodes:
                    __logger__.info(
                        "  op=%s | path=%s | name=%s | modules=%s | source=%s",
                        node.op_type,
                        node.path,
                        node.name,
                        _format_matched_modules(node.matched_modules),
                        node.source,
                    )
            else:
                __logger__.info("No suspicious ONNX nodes matched %s.", _SUSPICIOUS_ONNX_OP_TYPES)

            unsupported_onnx_ops = sorted({_normalize_unsupported_op_name(op) for op in unsupported_ops})
            exported_onnx_ops = set(_export_onnx_node_types(onnx_model))
            missing_from_export = [op for op in unsupported_onnx_ops if op and op not in exported_onnx_ops]
            if missing_from_export:
                __logger__.warning(
                    "Unsupported ops reported by auto_LiRPA but not present in the exported ONNX graph: %s. "
                    "Those ops are likely introduced during auto_LiRPA import or later decomposition, so the nearest module clues are the suspicious TorchScript nodes above.",
                    missing_from_export,
                )
        except Exception as exc:  # noqa: BLE001 - optional diagnostics should not mask the primary result.
            __logger__.warning("ONNX export failed with %s: %s", type(exc).__name__, exc)

    if conversion_error is not None and script_config.fail_on_conversion_error:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())