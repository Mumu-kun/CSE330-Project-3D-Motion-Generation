def _get_default_predictor_config():
    """Deferred import to avoid circular dependency with models.py"""
    from models import FlowMatchingPredictorConfig
    return FlowMatchingPredictorConfig(
        hidden_size=256,
        intermediate_size=768,
        num_hidden_layers=4,
        num_attention_heads=8,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        attention_bias=True,
        attention_dropout=0.1,
        mlp_bias=True,
        track_dimensionality=22,
        head_dim=None,
    )
