import torch as th
import torch.nn as nn

from pathlib import Path
from mpc_datagen import MPCConfig

from lcil.utils.base_models import load_feature_net, save_feature_net


def get_mpc_cfg_from_policy_model(policy_model: nn.Module) -> MPCConfig:
    global_config = getattr(policy_model, "global_config", None)
    if global_config is None:
        global_config = getattr(policy_model, "net", None)
        if global_config is not None:
            global_config = getattr(global_config, "global_config", None)

    if global_config is None:
        raise ValueError("Could not find a 'global_config' attribute in the policy model or its 'net' submodule.")

    if not isinstance(global_config, MPCConfig):
        raise TypeError(f"Expected 'global_config' to be an instance of MPCConfig, but got {type(global_config)}.")

    return global_config


class CartpoleAngleWrapper(nn.Module):
    """Feature net wrapper for the cartpole angle."""

    def __init__(
        self,
        feature_net: nn.Module,
    ):
        """Wraps the third state (angle) in sin/cos before passing to the feature net.

        Parameters
        ----------
        feature_net : nn.Module
            The underlying feature net that takes in the transformed state.
        """
        super().__init__()
        self.net = feature_net

    def __getattr__(self, name: str):
        """Delegate missing attributes to the underlying feature net."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.net, name)

    @staticmethod
    def _transform_inputs(x: th.Tensor) -> th.Tensor:
        cart_pos_vel = x[:, :2]
        theta = x[:, 2:3]
        theta_dot = x[:, 3:]

        return th.cat([
            cart_pos_vel,
            th.sin(theta),
            th.cos(theta),
            theta_dot
        ], dim=-1)

    def forward_raw(self, x: th.Tensor) -> th.Tensor:
        features = self._transform_inputs(x)
        raw_forward = getattr(self.net, "forward_raw", None)
        if callable(raw_forward):
            return raw_forward(features)
        return self.net(features)
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        features = self._transform_inputs(x)
        return self.net(features)

    def save(self, path: str | Path, **feature_net_save_kwargs) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        feature_net_path = checkpoint_path.with_name(
            checkpoint_path.stem + "_feature_net.pt"
        )
        save_feature_net(self.net, feature_net_path, **feature_net_save_kwargs)
        th.save({"feature_net_path": feature_net_path.name}, checkpoint_path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        map_location: th.device | str = "cpu",
        strict: bool = True,
        feature_net_cls: type[nn.Module] | None = None,
        feature_net_args: tuple | None = None,
        feature_net_kwargs: dict | None = None,
    ) -> "CartpoleAngleWrapper":
        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'.")

        payload = th.load(checkpoint_path, map_location=map_location, weights_only=True)
        feature_net_path = checkpoint_path.with_name(payload["feature_net_path"])
        feature_net = load_feature_net(
            feature_net_path,
            map_location=map_location,
            strict=strict,
            feature_net_cls=feature_net_cls,
            feature_net_args=feature_net_args,
            feature_net_kwargs=feature_net_kwargs,
        )
        model = cls(feature_net=feature_net)
        model.eval()
        return model
