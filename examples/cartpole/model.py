import torch as th
import torch.nn as nn

from pathlib import Path

from lcil.utils.base_models import load_feature_net, save_feature_net


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
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        cart_pos_vel = x[:, :2]
        theta = x[:, 2:3]
        theta_dot = x[:, 3:]
        
        features = th.cat([
            cart_pos_vel,
            th.sin(theta),
            th.cos(theta),
            theta_dot
        ], dim=-1)
        
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
