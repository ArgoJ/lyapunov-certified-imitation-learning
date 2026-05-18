import unittest

import torch as th
import torch.nn as nn

from lcil.utils import CertifiableTransformerEncoder, CertifiableTransformerEncoderLayer


class TestCertifiableTransformerEncoderLayer(unittest.TestCase):
    def test_preserves_shapes_for_supported_layouts(self) -> None:
        batch_first_layer = CertifiableTransformerEncoderLayer(
            d_model=8,
            nhead=2,
            dim_feedforward=16,
            dropout=0.0,
            activation="relu",
            batch_first=True,
        )
        sequence_first_layer = CertifiableTransformerEncoderLayer(
            d_model=8,
            nhead=2,
            dim_feedforward=16,
            dropout=0.0,
            activation="relu",
            batch_first=False,
        )

        batch_first_input = th.randn(2, 5, 8)
        sequence_first_input = th.randn(5, 2, 8)

        batch_first_output = batch_first_layer(batch_first_input, is_causal=True)
        sequence_first_output = sequence_first_layer(sequence_first_input, is_causal=True)

        self.assertEqual(tuple(batch_first_output.shape), (2, 5, 8))
        self.assertEqual(tuple(sequence_first_output.shape), (5, 2, 8))

    def test_causal_mask_changes_first_token_attention(self) -> None:
        layer = CertifiableTransformerEncoderLayer(
            d_model=4,
            nhead=2,
            dim_feedforward=8,
            dropout=0.0,
            activation="identity",
            batch_first=True,
        )
        layer.norm1 = nn.Identity()
        layer.norm2 = nn.Identity()

        with th.no_grad():
            for projection in (layer.q_proj, layer.k_proj, layer.v_proj, layer.out_proj):
                projection.weight.copy_(th.eye(4))
                projection.bias.zero_()
            layer.linear1.weight.zero_()
            layer.linear1.bias.zero_()
            layer.linear2.weight.zero_()
            layer.linear2.bias.zero_()

        src = th.tensor([[[1.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0]]])
        causal_mask = th.triu(th.ones(2, 2, dtype=th.bool), diagonal=1)

        masked_output = layer(src, src_mask=causal_mask)
        unmasked_output = layer(src)

        self.assertFalse(th.allclose(masked_output[:, 0, :], unmasked_output[:, 0, :]))


class TestCertifiableTransformerEncoder(unittest.TestCase):
    def test_stacks_layers_and_applies_optional_norm(self) -> None:
        layer = CertifiableTransformerEncoderLayer(
            d_model=8,
            nhead=2,
            dim_feedforward=16,
            dropout=0.0,
            activation="relu",
            batch_first=True,
        )
        encoder = CertifiableTransformerEncoder(
            encoder_layer=layer,
            num_layers=3,
            norm=nn.LayerNorm(8),
        )

        src = th.randn(2, 6, 8)
        output = encoder(src, mask=th.triu(th.ones(6, 6, dtype=th.bool), diagonal=1))

        self.assertEqual(tuple(output.shape), (2, 6, 8))


if __name__ == "__main__":
    unittest.main(verbosity=2)