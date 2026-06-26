from typing import List, Optional, Tuple

import torch
from torch import nn

from utils.config import Config, FlowMatchingPredictorConfig
from utils.models import AdaLN, GatedMLP, TemporalLayerCache, TemporalRoPEAttention, init_weights
from utils.motion_utils import FeatureNormalizer, positions_to_x271, x68_to_positions


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


class PredictorSelfAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        pred_config = config.predictor_config

        self.q_proj = nn.Linear(pred_config.hidden_size, pred_config.hidden_size, bias=pred_config.attention_bias)
        self.k_proj = nn.Linear(pred_config.hidden_size, pred_config.hidden_size, bias=pred_config.attention_bias)
        self.v_proj = nn.Linear(pred_config.hidden_size, pred_config.hidden_size, bias=pred_config.attention_bias)
        self.out_proj = nn.Linear(pred_config.hidden_size, pred_config.hidden_size, bias=pred_config.attention_bias)

        self.encoder_cond_proj = nn.Linear(config.encoder_config.hidden_size, pred_config.hidden_size, bias=True)
        self.gate_proj = nn.Linear(3 * pred_config.hidden_size, 1, bias=True)

        self.num_heads = pred_config.num_attention_heads
        self.head_dim = pred_config.hidden_size // pred_config.num_attention_heads
        self.hidden_size = pred_config.hidden_size
        self.attention_dropout = pred_config.attention_dropout

        self._init_weights()

    def _init_weights(self) -> None:
        init_weights(self, linear_init="xavier_normal")
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def attn_weights(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim**0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        return attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        masked_cond: torch.Tensor,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.shape

        query = self._reshape_heads(self.q_proj(hidden_states))
        key = self._reshape_heads(self.k_proj(hidden_states))
        value = self._reshape_heads(self.v_proj(hidden_states))

        attn_output = nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        attn_output = self.out_proj(attn_output)

        if output_attentions:
            attn_w = self.attn_weights(query, key)
            return attn_output, attn_w

        return attn_output, None


class PredictorLayer(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        pred_config = config.predictor_config

        self.adaln_self_attn = AdaLN(d_model=pred_config.hidden_size, d_cond=pred_config.hidden_size * 2)

        self.self_attn = PredictorSelfAttention(config)

        self.adaln_cross_attn = AdaLN(d_model=pred_config.hidden_size, d_cond=pred_config.hidden_size * 2)

        self.cross_attn = PredictorRopeCrossAttention(config)

        self.adaln_mlp = AdaLN(d_model=pred_config.hidden_size, d_cond=pred_config.hidden_size * 2)

        self.mlp = PredictorMLP(pred_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        adaln_cond: torch.Tensor,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        B, N, H = hidden_states.shape
        masked_cond_raw = encoder_hidden_states[:, -N:, :]

        masked_cond = self.self_attn.encoder_cond_proj(masked_cond_raw)
        gate_input = torch.cat([masked_cond, adaln_cond.unsqueeze(1).expand(-1, N, -1), hidden_states], dim=-1)
        gate = 1 + 0.5 * torch.tanh(self.self_attn.gate_proj(gate_input))

        hidden_states = hidden_states + gate * masked_cond

        residual = hidden_states
        hidden_states, self_attn_gate = self.adaln_self_attn(hidden_states, adaln_cond)
        hidden_states, self_attn_weights = self.self_attn(hidden_states, output_attentions=False)
        hidden_states = residual + self_attn_gate * hidden_states

        residual = hidden_states
        hidden_states, cross_attn_gate = self.adaln_cross_attn(hidden_states, adaln_cond)
        hidden_states, cross_attn_weights = self.cross_attn(
            hidden_states, encoder_hidden_states, output_attentions=output_attentions
        )
        hidden_states = residual + cross_attn_gate * hidden_states

        residual = hidden_states
        hidden_states, mlp_gate = self.adaln_mlp(hidden_states, adaln_cond)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + mlp_gate * hidden_states

        if output_attentions:
            return hidden_states, self_attn_weights, cross_attn_weights

        return hidden_states, None, None


class FlowMatchingPredictor(nn.Module):
    def __init__(
        self,
        config: Config,
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

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        init_weights(self, linear_init="xavier_normal")

    def forward(
        self,
        noisy_states: torch.Tensor,
        timesteps: torch.Tensor,
        track_features: torch.Tensor,
        text_embedding: torch.Tensor,
        output_attentions: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
        """
        Args:
            noisy_states: [B, N, H_enc]
            timesteps: [B] or [B, 1]
            track_features: [B, M+N, H_enc]
            text_embedding: [B, D]
            output_attentions: bool
        Returns:
            flow_prediction: [B, N, H_enc]
            all_cross_attns: List of attention weights from each layer if output_attentions is True, else None
        """
        B, N, H = noisy_states.shape

        text_cond = self.text_proj(text_embedding)  # [B, H_pred]
        time_cond = self.time_embedder(timesteps.squeeze(-1) if timesteps.dim() > 1 else timesteps)  # [B, H_pred]

        adaln_cond = torch.cat([text_cond, time_cond], dim=-1)  # [B, 2*H_pred]

        all_self_attns: list[torch.Tensor] = []
        all_cross_attns: list[torch.Tensor] = []

        hidden_states = self.latent_in_proj(noisy_states)  # (B, N, H_pred)

        for layer_idx, layer in enumerate(self.layers):
            hidden_states = hidden_states

            hidden_states, self_attn_weights, cross_attn_weights = layer(
                hidden_states,
                track_features,
                adaln_cond=adaln_cond,
                output_attentions=output_attentions,
            )

            if output_attentions:
                all_self_attns.append(self_attn_weights)
                all_cross_attns.append(cross_attn_weights)

        hidden_states = self.latent_out_proj(hidden_states)  # (B, N, H_enc)

        output_shift, output_scale = self.output_adaln(adaln_cond).chunk(2, dim=-1)
        output_shift = output_shift.unsqueeze(1)
        output_scale = output_scale.unsqueeze(1)
        flow_prediction = output_shift + output_scale * hidden_states

        return (
            flow_prediction,
            all_self_attns if output_attentions else None,
            all_cross_attns if output_attentions else None,
        )


class DecoderMLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.norm = nn.RMSNorm(config.decoder_config.hidden_size, eps=config.predictor_config.rms_norm_eps)

        self.mlp = nn.Sequential(
            nn.Linear(config.decoder_config.hidden_size, config.decoder_config.intermediate_size, bias=True),
            nn.GELU(),
            nn.Dropout(config.decoder_config.dropout),
            nn.Linear(config.decoder_config.intermediate_size, config.decoder_config.hidden_size, bias=True),
            nn.Dropout(config.decoder_config.dropout),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        init_weights(self, linear_init="xavier_normal")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x))


class LatentDecoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        H = config.decoder_config.hidden_size

        self.down_proj = nn.Linear(config.encoder_config.hidden_size, H, bias=True)
        self.blocks = nn.Sequential(*[DecoderMLP(config) for _ in range(config.decoder_config.num_layers)])

        self.head_norm = nn.RMSNorm(H, eps=config.predictor_config.rms_norm_eps)

        self.root_head_xz = nn.Sequential(
            nn.Linear(H, H),
            nn.GELU(),
            nn.Linear(H, 2),
        )

        self.root_head_y = nn.Sequential(
            nn.Linear(H, H),
            nn.GELU(),
            nn.Linear(H, H // 2),
            nn.GELU(),
            nn.Linear(H // 2, 1),
        )

        self.yaw_head = nn.Sequential(
            nn.Linear(H, H // 2),
            nn.GELU(),
            nn.Linear(H // 2, 2),
        )  # delta_yaw sin, cos

        self.ric_head = nn.Sequential(
            nn.Linear(H, 2 * H),
            nn.GELU(),
            nn.Linear(2 * H, 2 * H),
            nn.GELU(),
            nn.Linear(2 * H, 63),
        )  # 21 joint_ric_vel * 3

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        init_weights(self, linear_init="xavier_normal")

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        latent: (B, H_encoder) - the latent representation from the predictor
        output: (B, 68) - the predicted reduced features (root_y, root_xz_vel, delta_yaw sin, cos, joint_ric_vel * 3)
        """
        x = self.down_proj(latent)
        x = self.blocks(x)
        x = self.head_norm(x)

        root_xz = self.root_head_xz(x)  # (..., 2) - root_xz_vel
        root_y = self.root_head_y(x)  # (..., 1) - root_y
        yaw = self.yaw_head(x)  # (..., 2) - delta_yaw sin, cos
        yaw = nn.functional.normalize(yaw, dim=-1)  # normalize to unit vector
        ric = self.ric_head(x)  # (..., 63) - 21 joint_ric_vel * 3

        return torch.cat([root_y, root_xz, yaw, ric], dim=-1)

    def decode(
        self, latent: torch.Tensor, prev_pos: torch.Tensor, prev_frame: torch.Tensor, normalizer: FeatureNormalizer
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decodes the predicted flow output into new joint positions and relative shifts.
        Args:
            latent: The latent representation from the predictor (B, H_enc).
            prev_pos: The previous joint positions (B, 22, 3) - only the first joint is used for flow decoding.
            prev_frame: The previous frame's full features (B, 271) - normalized.
            normalizer: The feature normalizer to denormalize the outputs.
        """

        pred = self.forward(latent)

        new_pos = x68_to_positions(
            pred,
            normalizer,
            prev_frame,
            prev_pos,
        )
        new_frame, _ = positions_to_x271(new_pos, prev_pos, normalizer)

        relative_shift = new_pos - prev_pos

        return new_pos, relative_shift, new_frame
