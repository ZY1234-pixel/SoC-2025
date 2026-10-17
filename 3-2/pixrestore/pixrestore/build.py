"""Construct the restoration network."""
from .models import LightningDiT_PixelDiffusion


def build_model(config):
    return LightningDiT_PixelDiffusion(
        input_size=config.resolution,
        patch_size=config.patch_size,
        in_channels=6,
        out_channels=3,
        hidden_size=config.hidden_size,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        z_dims=config.encoder_dim,
        encdim_ratio=config.encoder_dim_ratio,
        use_qknorm=config.use_qknorm,
        use_swiglu=config.use_swiglu,
        use_rope=config.use_rope,
        use_rmsnorm=config.use_rmsnorm,
        adain_single=True,
        num_fused_layers=len(config.encoder_layers),
        pca_dim=config.bottleneck_dim,
        use_bottleneck_patch_embed=True,
        use_dino_layer_router=config.use_dino_layer_router,
        dino_layer_router_mode=config.dino_layer_router_mode,
        same_condition_layer_weights=config.same_condition_layer_weights,
        gate_temperature=config.gate_temperature,
        feature_norm=config.feature_norm,
        use_lq_degradation_token=config.use_lq_degradation_token,
    )
