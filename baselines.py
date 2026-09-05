"""
Full baseline suite for the head-to-head comparison table, built on MONAI
(https://monai.io) rather than hand-rolled, since UNETR/Swin UNETR are
intricate enough (window attention, relative position bias, patch
merging/expanding) that hand-writing them without being able to execute
and test the code here would be a real correctness risk. MONAI is the
standard PyTorch medical-imaging library and reviewers will recognize it;
using its validated implementations is the safer and more defensible
choice for a Q1 submission.

Registered models (name -> factory):

  "unet"            MONAI UNet, channels matched to the ~6M-parameter CNN
                     budget used throughout this project (32,64,128,192,256).
  "unet_plus_plus"  MONAI BasicUNetPlusPlus (nested skip pathways), same
                     channel budget - the paper's second-best baseline.
  "attention_unet"  MONAI AttentionUnet - attention-gated skip connections,
                     a very standard additional CNN comparator, same budget.
  "segresnet"       MONAI SegResNet - modern residual encoder-decoder
                     (BraTS-winning family), included as a stronger modern
                     CNN reference point beyond the two the dataset paper used.
  "unetr"           MONAI UNETR (ViT encoder + CNN decoder), spatial_dims=2,
                     standard published config. One of the dataset paper's
                     own four baselines.
  "swin_unetr"      MONAI SwinUNETR (hierarchical shifted-window attention),
                     spatial_dims=2, standard published config. The other
                     transformer baseline the dataset paper used.
  "anato_unetpp"    The proposed model from model.py (hand-written on
                     purpose, since the ventricle-channel wiring is the
                     actual contribution and needs to stay auditable).

CNN baselines are capacity-matched to the proposed model (~6M params) so
that architecture comparisons are fair. The two transformer baselines use
their standard published configurations rather than being shrunk to 6M -
this mirrors what the dataset paper itself did, and shrinking a ViT/Swin
encoder to 6M params would no longer be a faithful UNETR/Swin UNETR anyway.
Expect them to need smaller batch sizes; see README.md.

IMPORTANT: MONAI's exact constructor keyword names for UNETR/SwinUNETR
(`pos_embed` vs `proj_type`, 2D support, etc.) have changed across
versions, and this module cannot be executed here to verify them against
whatever MONAI build is installed on the training machine. Always run
`python sanity_check.py` first - it builds every model, runs one dummy
forward pass through the exact adapter used in training, and reports
which ones actually work on your installed MONAI/PyTorch before you spend
any GPU time.
"""

from __future__ import annotations

CNN_CHANNELS = (32, 64, 128, 192, 256)
IMG_SIZE = (256, 256)


def unpack_output(out):
    """Normalizes every model's forward() return value to (main_logits, [aux_logits...])."""
    if isinstance(out, dict):
        return out["logits"], out.get("aux_logits", [])
    if isinstance(out, (list, tuple)):
        return out[0], list(out[1:])
    return out, []


def build_model(name: str, in_channels: int, num_classes: int = 4,
                 base_channels: int = 32, deep_supervision: bool = True):
    if name == "anato_unetpp":
        from model import AnatoUNetPP
        return AnatoUNetPP(in_channels=in_channels, num_classes=num_classes,
                            base=base_channels, deep_supervision=deep_supervision)

    import monai.networks.nets as nets

    if name == "unet":
        return nets.UNet(spatial_dims=2, in_channels=in_channels, out_channels=num_classes,
                          channels=CNN_CHANNELS, strides=(2, 2, 2, 2), num_res_units=2)

    if name == "unet_plus_plus":
        return nets.BasicUNetPlusPlus(spatial_dims=2, in_channels=in_channels, out_channels=num_classes,
                                       features=(32, 32, 64, 128, 256, 32),
                                       deep_supervision=deep_supervision)

    if name == "attention_unet":
        return nets.AttentionUnet(spatial_dims=2, in_channels=in_channels, out_channels=num_classes,
                                   channels=CNN_CHANNELS, strides=(2, 2, 2, 2))

    if name == "segresnet":
        return nets.SegResNet(spatial_dims=2, in_channels=in_channels, out_channels=num_classes,
                               init_filters=32, blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1))

    if name == "unetr":
        return nets.UNETR(in_channels=in_channels, out_channels=num_classes, img_size=IMG_SIZE,
                           feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12,
                           norm_name="instance", spatial_dims=2)

    if name == "swin_unetr":
        # MONAI >=1.5 removed the `img_size` argument (the model became
        # input-size-agnostic, as long as H/W are divisible by 32); older
        # MONAI still requires it. Try without it first, since that's what
        # your installed version (1.6.0, confirmed via sanity_check.py) needs.
        try:
            return nets.SwinUNETR(in_channels=in_channels, out_channels=num_classes,
                                   feature_size=24, spatial_dims=2)
        except TypeError:
            return nets.SwinUNETR(img_size=IMG_SIZE, in_channels=in_channels, out_channels=num_classes,
                                   feature_size=24, spatial_dims=2)

    raise ValueError(f"Unknown model name: {name}. Known: {sorted(MODEL_NAMES)}")


MODEL_NAMES = {
    "unet", "unet_plus_plus", "attention_unet", "segresnet",
    "unetr", "swin_unetr", "anato_unetpp",
}

# Rough VRAM/throughput guidance for a 16GB card at 256x256, informing the
# per-model default batch size in run_all_folds.py / the notebook. Not
# hard limits - just sane starting points to avoid the first OOM.
RECOMMENDED_BATCH_SIZE = {
    "unet": 16,
    "unet_plus_plus": 12,
    "attention_unet": 12,
    "segresnet": 12,
    "unetr": 6,
    "swin_unetr": 6,
    "anato_unetpp": 12,
}
