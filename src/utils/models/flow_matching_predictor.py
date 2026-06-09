from typing import List, Optional, Tuple

import torch
from torch import nn

from utils.config import Config, FlowMatchingPredictorConfig
from utils.models import AdaLN, GatedMLP, TemporalLayerCache, TemporalRoPEAttention, init_weights
from utils.motion_utils import FeatureNormalizer, flow_output_to_positions


class SinusoidalEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp: nn.Sequential = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        init_weights(self, linear_init="xavier_normal")

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        if half == 0:
            return torch.zeros((t.shape[0], dim), device=t.device, dtype=torch.float32)

        max_period_tensor = torch.tensor(max_period, device=t.device, dtype=torch.float32)
        freqs = torch.exp(
            -torch.log(max_period_tensor) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        out = self.mlp(t_freq)
        return out


class PredictorMLP(GatedMLP):
    def __init__(self, config: FlowMatchingPredictorConfig):
        super().__init__(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=config.mlp_bias,
            activation="silu",
            dropout=0.0,
        )


class PredictorRopeCrossAttention(TemporalRoPEAttention):
    def __init__(self, config: Config) -> None:
        super().__init__(config.predictor_config)
        self.config = config

        self.k_proj = nn.Linear(config.encoder_config.hidden_size, config.predictor_config.hidden_size, bias=True)
        self.v_proj = nn.Linear(config.encoder_config.hidden_size, config.predictor_config.hidden_size, bias=True)

        self.kv_cache = TemporalLayerCache()
        self._init_weights()

    def _init_weights(self) -> None:
        init_weights(self, linear_init="xavier_normal")

    def attn_weights(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim**0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)

        return attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, N, H)
        encoder_hidden_states: torch.Tensor,  # (B, M, H_enc)
        _encoder_cache: Optional[TemporalLayerCache] = None,
        output_attentions: bool = False,
    ):
        batch_size, seq_len, _ = hidden_states.shape

        encoder_cache = self.kv_cache if _encoder_cache is None else _encoder_cache

        if encoder_cache.key is None or encoder_cache.value is None:
            key = self._reshape_heads(self.k_proj(encoder_hidden_states))
            value = self._reshape_heads(self.v_proj(encoder_hidden_states))
            encoder_cache.key = key
            encoder_cache.value = value

        query = self._reshape_heads(self.q_proj(hidden_states))
        key = encoder_cache.key
        value = encoder_cache.value

        cos, sin = self._build_rope(
            seq_len=seq_len,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        query = self._apply_rope(query, cos, sin)

        attn_output = nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        if output_attentions:
            attn_weights = self.attn_weights(query, key, value)
            return attn_output, attn_weights

        return attn_output


class PredictorLayer(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        pred_config = config.predictor_config

        self.adaln_attn = AdaLN(d_model=pred_config.hidden_size, d_cond=pred_config.hidden_size * 2)

        self.cross_attn = PredictorRopeCrossAttention(config)

        self.adaln_mlp = AdaLN(d_model=pred_config.hidden_size, d_cond=pred_config.hidden_size * 2)

        self.mlp = PredictorMLP(pred_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        adaln_cond: torch.Tensor,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states, attn_gate = self.adaln_attn(hidden_states, adaln_cond)
        hidden_states, attn_weights = self.cross_attn(
            hidden_states, encoder_hidden_states, output_attentions=output_attentions
        )
        hidden_states = residual + attn_gate * hidden_states

        residual = hidden_states
        hidden_states, mlp_gate = self.adaln_mlp(hidden_states, adaln_cond)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + mlp_gate * hidden_states

        if output_attentions:
            return hidden_states, attn_weights

        return hidden_states, None


class FlowMatchingPredictor(nn.Module):
    def __init__(
        self,
        config: Config,  # Model configuration
        **kwargs,
    ):
        super().__init__()
        self.config = config
        pred_config = config.predictor_config

        self.text_proj = nn.Linear(config.text_embedding_dim, pred_config.hidden_size, bias=True)

        # Time embedding for denoising timestep
        self.time_embedder = SinusoidalEmbedder(pred_config.hidden_size)

        # Transformer layers with AdaLN
        self.layers = nn.ModuleList([PredictorLayer(config) for _ in range(pred_config.num_hidden_layers)])

        self.latent_in_proj = nn.Linear(config.encoder_config.hidden_size, pred_config.hidden_size, bias=True)
        self.latent_out_proj = nn.Linear(pred_config.hidden_size, config.encoder_config.hidden_size, bias=True)
        # Output prediction head
        self.output_adaln = nn.Sequential(
            nn.Linear(pred_config.hidden_size * 2, config.encoder_config.hidden_size * 2, bias=True),
            nn.SiLU(),
            nn.Linear(config.encoder_config.hidden_size * 2, config.encoder_config.hidden_size * 2, bias=True),
        )
        self.output_norm = nn.LayerNorm(config.encoder_config.hidden_size)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        init_weights(self, linear_init="xavier_normal")

    def forward(
        self,
        noisy_states: torch.Tensor,  # (B, N, H_enc) - noised reduced-state features
        timesteps: torch.Tensor,  # (B,) or (B, 1) - denoising timesteps in [0,1]
        encoder_hidden_states: torch.Tensor,  # (B, M, H_enc) - from MotionHistoryEncoder
        text_embedding: torch.Tensor,  # (B, F) - Text embedding for global conditioning
        output_attentions: bool = False,
        **kwargs,  # Ignore attention_mask, position_ids, etc.
    ) -> tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        # Build static per-joint kinematic tokens once and reuse across layers.
        B, N, H = noisy_states.shape

        text_cond = self.text_proj(text_embedding)  # (B, H)
        time_cond = self.time_embedder(timesteps.squeeze(-1) if timesteps.dim() > 1 else timesteps)  # (B, H)

        adaln_cond = torch.cat([text_cond, time_cond], dim=-1)  # (B, 2H)

        # 4. Transformer processing (NO ATTENTION MASKING)
        all_cross_attns: list[torch.Tensor] = []

        hidden_states = self.latent_in_proj(noisy_states)  # (B, N, H) — projected into predictor hidden dim

        for layer_idx, layer in enumerate(self.layers):
            # Bounded signed gate allows add/subtract structural prior per layer.
            hidden_states = hidden_states

            hidden_states, attn_weights = layer(
                hidden_states,
                encoder_hidden_states,
                adaln_cond=adaln_cond,
                output_attentions=output_attentions,
            )

            if output_attentions:
                all_cross_attns.append(attn_weights)

        hidden_states = self.latent_out_proj(hidden_states)  # (B, N, H_enc)

        output_shift, output_scale = self.output_adaln(adaln_cond).chunk(2, dim=-1)
        output_shift = output_shift.unsqueeze(1)  # (B, 1, H_enc)
        output_scale = output_scale.unsqueeze(1)  # (B, 1, H_enc)
        flow_prediction = self.output_norm(output_shift + output_scale * hidden_states)

        return (
            flow_prediction,
            all_cross_attns if output_attentions else None,
        )


class LatentDecoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.decoder = GatedMLP(
            hidden_size=config.encoder_config.hidden_size,
            intermediate_size=config.encoder_config.hidden_size * 4,
            bias=True,
            activation="silu",
            dropout=0.0,
        )
        self.out_proj = nn.Linear(config.encoder_config.hidden_size, 68, bias=True)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        init_weights(self, linear_init="xavier_normal")

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        pred = self.decoder(latent)
        pred = self.out_proj(pred)
        return pred

    def decode(
        self, latent: torch.Tensor, prev_pos: torch.Tensor, prev_frame: torch.Tensor, normalizer: FeatureNormalizer
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes the predicted flow output into new joint positions and relative shifts.
        Args:
            latent: The latent representation from the predictor (B, N, H_enc).
            prev_pos: The previous joint positions (B, 22, 3) - only the first joint is used for flow decoding.
            prev_frame: The previous frame's full features (B, 271) - used for denormalization.
            normalizer: The feature normalizer to denormalize the outputs.
        """

        pred = self.forward(latent)
        pred_raw = normalizer.denormalize_flow_output(pred)
        prev_frame_raw = normalizer.denormalize(prev_frame)

        new_pos = flow_output_to_positions(pred_raw, prev_pos[:, 0], prev_frame_raw[:, 69:75])

        relative_shift = new_pos - prev_pos

        return new_pos, relative_shift
