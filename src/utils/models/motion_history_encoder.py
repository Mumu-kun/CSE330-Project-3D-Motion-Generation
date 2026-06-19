from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import Config, MotionHistoryEncoderConfig
from utils.models import AdaLN, GatedMLP, TemporalCacheState, TemporalLayerCache, TemporalRoPEAttention, init_weights


class EncoderRoPEAttention(TemporalRoPEAttention):
    def __init__(self, config: MotionHistoryEncoderConfig) -> None:
        super().__init__(config)
        self.num_registers = config.num_registers
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"TemporalRoPEAttention requires an even per-head dimension, got head_dim={self.head_dim}."
            )

    def forward(self, hidden_states: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        query = self._reshape_heads(self.q_proj(hidden_states))
        key = self._reshape_heads(self.k_proj(hidden_states))
        value = self._reshape_heads(self.v_proj(hidden_states))

        cos, sin = self._build_rope(
            seq_len=seq_len - self.num_registers,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        query[:, :, self.num_registers :, :] = self._apply_rope(query[:, :, self.num_registers :, :], cos, sin)
        key[:, :, self.num_registers :, :] = self._apply_rope(key[:, :, self.num_registers :, :], cos, sin)

        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.out_proj(attn_output)


class EncoderMLP(GatedMLP):
    def __init__(self, config: MotionHistoryEncoderConfig) -> None:
        super().__init__(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=config.mlp_bias,
            activation=config.hidden_act,
            dropout=config.dropout,
        )


class EncoderLayer(nn.Module):
    def __init__(self, config: MotionHistoryEncoderConfig) -> None:
        super().__init__()

        self.adaln_attn = AdaLN(d_model=config.hidden_size, d_cond=config.text_embedding_dim)

        self.self_attn = EncoderRoPEAttention(config)

        self.adaln_mlp = AdaLN(d_model=config.hidden_size, d_cond=config.text_embedding_dim)

        self.mlp = EncoderMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        text_emb: torch.Tensor,
        is_causal: bool = True,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, attn_gate = self.adaln_attn(hidden_states, text_emb)
        hidden_states = self.self_attn(hidden_states, is_causal=is_causal)
        hidden_states = residual + attn_gate * hidden_states

        residual = hidden_states
        hidden_states, mlp_gate = self.adaln_mlp(hidden_states, text_emb)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + mlp_gate * hidden_states

        return hidden_states


class MotionHistoryEncoder(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self._config = config
        self.config = config.encoder_config
        enc_config = config.encoder_config

        self.frame_projection = nn.Linear(config.motion_dim, enc_config.hidden_size, bias=True)

        self.layers = nn.ModuleList([EncoderLayer(enc_config) for _ in range(enc_config.num_hidden_layers)])

        self.final_norm = nn.LayerNorm(enc_config.hidden_size, eps=enc_config.layer_norm_eps)

        self.register_tokens = nn.Parameter(torch.empty(enc_config.num_registers, enc_config.hidden_size))
        self.mask_token = nn.Parameter(torch.empty(1, enc_config.hidden_size))

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.register_tokens, mean=0.0, std=0.02)
        nn.init.normal_(self.mask_token, mean=0.0, std=0.01)
        init_weights(self, linear_init="trunc_normal", linear_std=0.02)

    def _empty_cache_state(self) -> TemporalCacheState:
        return TemporalCacheState(layers=[TemporalLayerCache() for _ in range(self.config.num_hidden_layers)])

    def forward(
        self,
        motion_seq: torch.Tensor,
        text_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_layer_outputs: bool = False,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if motion_seq.ndim != 3 or motion_seq.shape[-1] != self._config.motion_dim:
            raise ValueError(
                f"Expected motion_seq shape (B, T, {self._config.motion_dim}), got {tuple(motion_seq.shape)}"
            )
        if text_emb.ndim != 2 or text_emb.shape[-1] != self._config.text_embedding_dim:
            raise ValueError(
                f"Expected text_emb shape (B, {self._config.text_embedding_dim}), got {tuple(text_emb.shape)}"
            )
        if text_emb.shape[0] != motion_seq.shape[0]:
            raise ValueError(
                "Batch size mismatch between motion_seq and text_emb: "
                f"{tuple(motion_seq.shape)} vs {tuple(text_emb.shape)}"
            )

        if mask is not None and mask.shape != motion_seq.shape[:2]:
            if mask.ndim != 2:
                raise ValueError(
                    "Expected mask shape (B, T) matching motion_seq, got "
                    f"{tuple(mask.shape)} vs {tuple(motion_seq.shape[:2])}"
                )

        # --------------------------

        batch_size, seq_len, _ = motion_seq.shape
        if seq_len == 0:
            raise ValueError("Expected motion_seq with at least one timestep.")

        hidden_states: torch.Tensor = self.frame_projection(motion_seq)

        if mask is not None:
            mask_flat = mask.flatten()  # (B*T,)
            hidden_states_flat = hidden_states.flatten(0, 1)  # (B*T, H)
            hidden_states_flat[mask_flat] = self.mask_token.to(hidden_states_flat)
            hidden_states = hidden_states_flat.view_as(hidden_states)  # (B, T, H)

        register_tokens = self.register_tokens.unsqueeze(0).expand(batch_size, -1, -1)

        hidden_states = torch.cat([register_tokens, hidden_states], dim=1)

        all_hidden_states: list[torch.Tensor] = []

        for layer in self.layers:
            hidden_states = layer(hidden_states, text_emb, is_causal=is_causal)

            if return_layer_outputs:
                all_hidden_states.append(hidden_states[:, self.config.num_registers :, :])

        hidden_states = self.final_norm(hidden_states)
        hidden_states = hidden_states[:, self.config.num_registers :, :]

        if return_layer_outputs:
            all_hidden_states[-1] = hidden_states
            return torch.stack(all_hidden_states, dim=2)  # (B, N, L, H)

        return hidden_states  # (B, N, H)

    def step(
        self,
        x_t: torch.Tensor,
        text_emb: torch.Tensor,
        frame_buffer: Optional[torch.Tensor],
        cache_state: Optional[TemporalCacheState] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, TemporalCacheState]:
        if x_t.ndim != 2 or x_t.shape[-1] != self._config.motion_dim:
            raise ValueError(f"Expected x_t shape (B, {self._config.motion_dim}), got {tuple(x_t.shape)}")

        if frame_buffer is None:
            next_frame_buffer = x_t.unsqueeze(1)
        else:
            if frame_buffer.ndim != 3 or frame_buffer.shape[-1] != self._config.motion_dim:
                raise ValueError(
                    f"Expected frame_buffer shape (B, T, {self._config.motion_dim}), got {tuple(frame_buffer.shape)}"
                )
            if frame_buffer.shape[0] != x_t.shape[0]:
                raise ValueError(
                    "Batch size mismatch between x_t and frame_buffer: "
                    f"{tuple(x_t.shape)} vs {tuple(frame_buffer.shape)}"
                )
            next_frame_buffer = torch.cat([frame_buffer, x_t.unsqueeze(1)], dim=1)

        next_cache_state = cache_state or self._empty_cache_state()
        if len(next_cache_state.layers) != self.config.num_hidden_layers:
            raise ValueError(
                "cache_state layer count mismatch: "
                f"expected {self.config.num_hidden_layers}, got {len(next_cache_state.layers)}"
            )
        return (
            self.forward(next_frame_buffer, text_emb),
            next_frame_buffer,
            next_cache_state,
        )


class LinearProbe(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        text_embedding_dim: int = 512,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, text_embedding_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.zeros_(self.linear.bias)
        nn.init.trunc_normal_(self.linear.weight, std=0.02)
        if self.norm.weight is not None:
            nn.init.ones_(self.norm.weight)
        if self.norm.bias is not None:
            nn.init.zeros_(self.norm.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(f"Expected hidden_states shape (B, T, D), got {tuple(hidden_states.shape)}")
        seq_mean = hidden_states.mean(dim=1)
        x = self.norm(seq_mean)
        x = self.linear(x)
        return F.normalize(x, dim=-1)


class JepaMLPBlock(nn.Module):
    def __init__(self, config: MotionHistoryEncoderConfig) -> None:
        super().__init__()
        self.config = config

        self.norm = nn.LayerNorm(config.jp_config.hidden_size, eps=config.layer_norm_eps)

        self.mlp = GatedMLP(
            hidden_size=config.jp_config.hidden_size,
            intermediate_size=config.jp_config.intermediate_size,
            bias=True,
            activation=config.hidden_act,
            dropout=config.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.mlp(x)
        x = residual + x
        return x


class JepaPredictor(nn.Module):
    def __init__(self, config: MotionHistoryEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.jp_config.hidden_size
        self.intermediate_size = config.jp_config.intermediate_size
        self.num_layers = config.jp_config.num_hidden_layers

        self.z_proj = nn.Linear(self.hidden_size // 4, self.hidden_size)

        self.input_mlp = GatedMLP(
            hidden_size=config.num_hidden_layers * config.hidden_size,
            intermediate_size=2 * self.hidden_size,
            output_size=self.hidden_size,
            bias=True,
            activation=config.hidden_act,
            dropout=config.dropout,
        )

        self.input_proj = nn.Linear(2 * self.hidden_size, self.hidden_size)

        self.blocks = nn.ModuleList([JepaMLPBlock(config) for _ in range(self.num_layers)])

        self.final_norm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)

        self.output_mlp = GatedMLP(
            hidden_size=self.hidden_size,
            intermediate_size=4 * self.hidden_size,
            output_size=self.hidden_size * config.num_hidden_layers,
            bias=True,
            activation=config.hidden_act,
            dropout=config.dropout,
        )

        self.output_proj = nn.ModuleList(
            [nn.Linear(self.hidden_size, config.hidden_size) for _ in range(config.num_hidden_layers)]
        )

        self._init_weights()

    def _init_weights(self) -> None:
        init_weights(self, linear_init="xavier_normal")

    def forward(self, motion_history_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            motion_history_emb: (B, T, L, H) where
                B = batch size,
                T = sequence length,
                L = config.num_hidden_layers,
                H = config.hidden_size.

        Returns:
            (B, T, L, H) where
                B = batch size,
                T = sequence length,
                L = config.num_hidden_layers,
                H = config.hidden_size.
        """

        if motion_history_emb.ndim != 4 or motion_history_emb.shape[-1] != self.config.hidden_size:
            raise ValueError(
                "Expected motion_history_emb shape "
                f"(B, T, L, {self.config.hidden_size}), got {tuple(motion_history_emb.shape)}"
            )

        B, T, L, H_enc = motion_history_emb.shape

        motion_history_emb = motion_history_emb.reshape(B, T, -1)  # (B, T, L*H_enc)
        motion_history_emb = self.input_mlp(motion_history_emb)  # (B, T, H_enc)

        z = torch.randn(B, self.z_proj.in_features, device=motion_history_emb.device)  # (B, H//4)
        # VJEPA-style latent noise — random noise injected as a learnable conditioning signal.

        z_proj = self.z_proj(z).unsqueeze(1).expand(-1, T, -1)  # (B, T, H)
        combined = torch.cat([motion_history_emb, z_proj], dim=-1)
        x = self.input_proj(combined)  # (B, T, H)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        x = self.output_mlp(x)  # (B, T, L*H)

        x = x.view(B, T, L, -1)  # (B, T, L, H)

        layer_outputs = []
        for i in range(self.config.num_hidden_layers):
            layer_output = self.output_proj[i](x[:, :, i, :])  # (B, T, H_enc)
            layer_outputs.append(layer_output)

        output = torch.stack(layer_outputs, dim=2)  # (B, T, L, H_enc)

        return output
