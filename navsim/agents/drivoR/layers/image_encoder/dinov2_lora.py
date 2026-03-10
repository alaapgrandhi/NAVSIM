# code adapted from https://github.com/mrabiabrn/robustbev
#

import math
import numpy as np
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.vision_transformer import VisionTransformer
from timm.models.convnext import ConvNeXt
from einops import rearrange
from safetensors import safe_open
from safetensors.torch import save_file
from .grid_mask import GridMask
from navsim.agents.drivoR.utils import pylogger
import torchvision.transforms as T
import inspect
log = pylogger.get_pylogger(__name__)

class timm_ViT(VisionTransformer):

    def _pos_embed(self, x: torch.Tensor, scene_tokens: torch.Tensor = None) -> torch.Tensor:
        """Apply positional embedding to input."""
        if self.pos_embed is None:
            return x.view(x.shape[0], -1, x.shape[-1])

        if self.dynamic_img_size:
            B, H, W, C = x.shape
            prev_grid_size = self.patch_embed.grid_size
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                new_size=(H, W),
                old_size=prev_grid_size,
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        # concatenate the scene tokens
        if scene_tokens is not None:
            x = torch.cat([scene_tokens,x], dim=1)


        return self.pos_drop(x)

    def _pos_embed_2(self, x: torch.Tensor, scene_tokens: torch.Tensor = None) -> torch.Tensor:
        if self.dynamic_img_size:
            B, H, W, C = x.shape
            if self.pos_embed is not None:
                prev_grid_size = self.patch_embed.grid_size
                pos_embed = resample_abs_pos_embed(
                    self.pos_embed,
                    new_size=(H, W),
                    old_size=prev_grid_size,
                    num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
                )
            else:
                pos_embed = None
            x = x.view(B, -1, C)
            rot_pos_embed = self.rope.get_embed(shape=(H, W)) if self.rope is not None else None
        else:
            pos_embed = self.pos_embed
            rot_pos_embed = self.rope.get_embed() if self.rope is not None else None

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            # position embedding does not overlap with class / reg token
            if pos_embed is not None:
                x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # pos_embed has entry for class / reg token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            if pos_embed is not None:
                x = x + pos_embed

        if scene_tokens is not None:
            x = torch.cat([scene_tokens,x], dim=1)
            base_npt = self.blocks[0].attn.num_prefix_tokens   # e.g., 5

            # bump prefix count to include scene tokens as prefix too
            new_npt = base_npt + scene_tokens.shape[1]
            for blk in self.blocks:
                # blk.attn.num_prefix_tokens = new_npt
                blk.attn.num_prefix_tokens = self.num_prefix_tokens + scene_tokens.shape[1]
        x = self.pos_drop(x)
        # print("\n\n\n")
        # print(to_cat[0].shape)
        # print(len(to_cat))
        # print(scene_tokens.shape)
        # print(self.num_prefix_tokens)
        # print(x.shape)
        # print("\n\n\n")
        # a=1/0

        # apply patch dropout to patches and rotary position embedding
        if self.patch_drop is not None:
            x, keep_indices = self.patch_drop(x)
            if rot_pos_embed is not None and keep_indices is not None:
                rot_pos_embed = apply_keep_indices_nlc(x, rot_pos_embed, keep_indices)
                # After applying keep indices to rope embeds, batch dim is added
                if getattr(self, 'rope_mixed', False):
                    # B, D, nH, N, dim -> D, B, nH, N, dim. For consistent iteration over depth at index 0.
                    rot_pos_embed = rot_pos_embed.transpose(0, 1)
                else:
                    # B, N, dim -> B, 1, N, dim.  Need head dim singleton for correct dim alignment in axial mode.
                    rot_pos_embed = rot_pos_embed.unsqueeze(1)

        return x, rot_pos_embed

    def forward_features(self, x: torch.Tensor, scene_tokens: torch.Tensor = None, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through feature layers (embeddings, transformer blocks, post-transformer norm)."""
        x = self.patch_embed(x)
        
        if self.patch_drop is not None:
            x = self._pos_embed(x, scene_tokens)
            x = self.patch_drop(x)
            x = self.norm_pre(x)

            if attn_mask is not None:
                # If mask provided, we need to apply blocks one by one
                for blk in self.blocks:
                    x = blk(x, attn_mask=attn_mask)
            elif self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint_seq(self.blocks, x)
            else:
                x = self.blocks(x)
        else:
            x, rope = self._pos_embed_2(x, scene_tokens)
            x = self.norm_pre(x)
            if attn_mask is not None:
                # If mask provided, we need to apply blocks one by one
                for blk in self.blocks:
                    print("Block forward:", inspect.signature(blk.forward))
                    x = blk(x, attn_mask=attn_mask, rope=rope)
            elif self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint_seq(self.blocks, x)
            else:
                # for blk in self.blocks:
                #     # print("Block forward:", inspect.signature(blk.forward))
                #     # print(x.shape)
                #     # print(rope.shape)
                for blk in self.blocks:
                    x = blk(x, rope=rope)
                # x = self.blocks(x, rope=rope)

        x = self.norm(x)
        return x


class _LoRA_qkv_timm(nn.Module):
    """In timm it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    """
    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
        linear_a_k: nn.Module,
        linear_b_k: nn.Module,
        layer_norm_q: nn.Module = None,
        layer_norm_v: nn.Module = None,
        layer_norm_k: nn.Module = None,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.linear_a_k = linear_a_k
        self.linear_b_k = linear_b_k
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

        self.layernorm_q = layer_norm_q
        self.layernorm_v = layer_norm_v
        self.layernorm_k = layer_norm_k

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(self.layernorm_q(x)))
        new_v = self.linear_b_v(self.linear_a_v(self.layernorm_v(x)))
        #new_k = self.linear_b_k(self.linear_a_k(self.layernorm_k(x)))
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v
        #qkv[:, :, self.dim : 2 * self.dim] += new_k
        return qkv


class LoRA_ViT_timm(nn.Module):
    def __init__(self, vit_model: timm_ViT, r: int, lora_layer=None, use_layer_norm=False, use_qkv=False):
        super(LoRA_ViT_timm, self).__init__()

        if r == 0:
            for param in vit_model.parameters():
                param.requires_grad = False
            self.lora_vit = vit_model
            
        else:
            if lora_layer:
                self.lora_layer = lora_layer
            else:
                self.lora_layer = list(range(len(vit_model.blocks)))

            # dim = vit_model.head.in_features
            # create for storage, then we can init them or load weights
            self.w_As = []  # These are linear layers
            self.w_Bs = []

            # lets freeze first
            for param in vit_model.parameters():
                param.requires_grad = False

            # Here, we do the surgery
            for t_layer_i, blk in enumerate(vit_model.blocks):
                # If we only want few lora layer instead of all
                if t_layer_i not in self.lora_layer:
                    continue
                w_qkv_linear = blk.attn.qkv
                self.dim = w_qkv_linear.in_features
                w_a_linear_q = nn.Linear(self.dim, r, bias=False)
                w_b_linear_q = nn.Linear(r, self.dim, bias=False)
                w_a_linear_v = nn.Linear(self.dim, r, bias=False)
                w_b_linear_v = nn.Linear(r, self.dim, bias=False)
                w_a_linear_k = nn.Identity()
                w_b_linear_k = nn.Identity()
                if use_qkv:
                    w_a_linear_k = nn.Linear(self.dim, r, bias=False)
                    w_b_linear_k = nn.Linear(r, self.dim, bias=False)
                layer_norm_q = nn.Identity()
                layer_norm_v = nn.Identity()
                layer_norm_k = nn.Identity()
                if use_layer_norm:
                    layer_norm_q = nn.LayerNorm(self.dim)
                    layer_norm_v = nn.LayerNorm(self.dim)
                    if use_qkv:
                        layer_norm_k = nn.LayerNorm(self.dim)
                self.w_As.append(w_a_linear_q)
                self.w_Bs.append(w_b_linear_q)
                self.w_As.append(w_a_linear_v)
                self.w_Bs.append(w_b_linear_v)
                blk.attn.qkv = _LoRA_qkv_timm(
                    w_qkv_linear,
                    w_a_linear_q,
                    w_b_linear_q,
                    w_a_linear_v,
                    w_b_linear_v,
                    w_a_linear_k,
                    w_b_linear_k,
                    layer_norm_q,
                    layer_norm_v,
                    layer_norm_k,
                )
            self.reset_parameters()
            self.lora_vit = vit_model

    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        
        save both lora and fc parameters.
        """

        assert filename.endswith(".safetensors")

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        
        merged_dict = {**a_tensors, **b_tensors}
        save_file(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\
            
        load both lora and fc parameters.
        """

        assert filename.endswith(".safetensors")

        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.w_As):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_A_linear.weight = Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.w_Bs):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_B_linear.weight = Parameter(saved_tensor)
                

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x: Tensor, scene_tokens: torch.Tensor = None) -> Tensor:
        return self.lora_vit.forward_features(x, scene_tokens)


class timm_ConvNeXt(ConvNeXt):
    """
    ConvNeXt variant whose forward_features returns token sequence (B, T, C)
    and accepts scene_tokens, analogous to timm_ViT in your code.
    """

    def forward_features(self, x: Tensor, scene_tokens: Tensor = None) -> Tensor:
        # Call the original ConvNeXt implementation
        fmap = super().forward_features(x)  # usually (B, C, H, W) for convnext_tiny.dinov3

        if fmap.ndim == 4:
            B, C, H, W = fmap.shape
            tokens = fmap.flatten(2).transpose(1, 2)  # (B, H*W, C)
        elif fmap.ndim == 2:
            # If it's already pooled: treat as single token
            tokens = fmap.unsqueeze(1)  # (B, 1, C)
        else:
            raise RuntimeError(f"Unexpected ConvNeXt feature shape {fmap.shape}")

        if scene_tokens is not None:
            if scene_tokens.shape[-1] != tokens.shape[-1]:
                raise ValueError(
                    f"scene_tokens dim {scene_tokens.shape[-1]} != ConvNeXt token dim {tokens.shape[-1]}"
                )
            tokens = torch.cat([scene_tokens, tokens], dim=1)

        return tokens

    def forward(self, x: Tensor, scene_tokens: Tensor = None) -> Tensor:
        # Mirror your timm_ViT: forward just delegates to forward_features
        return self.forward_features(x, scene_tokens)


class _LoRA_fc_timm(nn.Module):
    """
    LoRA wrapper for a *single* module (fc1 or fc2), analogous to _LoRA_qkv_timm.

    base:      original nn.Linear or nn.Conv2d (usually 1x1 conv in ConvNeXt MLP)
    linear_a:  low-rank 'A' (dim -> r)
    linear_b:  low-rank 'B' (r -> dim_out)
    layer_norm: optional LayerNorm (like your use_layer_norm flag)
    """
    def __init__(
        self,
        base: nn.Module,
        linear_a: nn.Module,
        linear_b: nn.Module,
        layer_norm: nn.Module = None,
    ):
        super().__init__()
        self.base = base
        self.linear_a = linear_a
        self.linear_b = linear_b
        self.layernorm = layer_norm if layer_norm is not None else nn.Identity()

        # Freeze base (ConvLoRA / LoRA style)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        # Exactly like your _LoRA_qkv_timm: base(x) + low-rank correction
        delta = self.linear_b(self.linear_a(self.layernorm(x)))
        return self.base(x) + delta
    

class LoRA_ConvNeXt_timm(nn.Module):
    """
    LoRA for ConvNeXt, analogous to LoRA_ViT_timm for ViT.

    - convnext_model: timm_ConvNeXt instance (so forward_features(x, scene_tokens) returns tokens)
    - r: LoRA rank
    - lora_layer: list[int] of block indices (flattened over all stages). If None -> all blocks.
    - use_layer_norm: apply LayerNorm before A/B for Linear MLPs.
    """
    def __init__(
        self,
        convnext_model: timm_ConvNeXt,
        r: int,
        lora_layer=None,
        use_layer_norm: bool = False,
    ):
        super().__init__()

        self.w_As = []
        self.w_Bs = []

        if r == 0:
            # no LoRA, just freeze
            for p in convnext_model.parameters():
                p.requires_grad = False
            self.lora_convnext = convnext_model
            self.num_features = getattr(convnext_model, "num_features", None)
            return

        # ----- flatten ConvNeXt blocks like vit_model.blocks -----
        stages = getattr(convnext_model, "stages", None)
        if stages is None:
            raise ValueError("Expected ConvNeXt model with `.stages` attribute")

        block_list = []  # (flat_idx, block)
        flat_idx = 0
        for stage in stages:
            blocks = getattr(stage, "blocks", None)
            if blocks is None:
                continue
            for blk in blocks:
                block_list.append((flat_idx, blk))
                flat_idx += 1

        num_blocks = len(block_list)
        if lora_layer is None:
            # default: all blocks get LoRA (like list(range(len(vit_model.blocks)))
            lora_layer = list(range(num_blocks))
        self.lora_layer = set(lora_layer)

        # freeze all original params
        for p in convnext_model.parameters():
            p.requires_grad = False

        # ----- surgery: replace blk.mlp.fc1 / fc2 in-place -----
        for t_layer_i, blk in block_list:
            if t_layer_i not in self.lora_layer:
                continue

            mlp = getattr(blk, "mlp", None)
            if mlp is None:
                continue

            for fc_name in ["fc1", "fc2"]:
                if not hasattr(mlp, fc_name):
                    continue

                fc = getattr(mlp, fc_name)

                if isinstance(fc, nn.Linear):
                    in_dim = fc.in_features
                    out_dim = fc.out_features
                    w_a = nn.Linear(in_dim, r, bias=False)
                    w_b = nn.Linear(r, out_dim, bias=False)
                    ln = nn.LayerNorm(in_dim) if use_layer_norm else nn.Identity()
                elif isinstance(fc, nn.Conv2d):
                    # MLP conv path: should be 1x1 conv
                    if fc.kernel_size != (1, 1):
                        raise ValueError(f"LoRA only implemented for 1x1 convs; got {fc.kernel_size}")
                    in_dim = fc.in_channels
                    out_dim = fc.out_channels
                    w_a = nn.Conv2d(in_dim, r, kernel_size=1, bias=False)
                    w_b = nn.Conv2d(r, out_dim, kernel_size=1, bias=False)
                    ln = nn.Identity()
                else:
                    raise TypeError(f"Unsupported MLP layer type {type(fc)} for {fc_name}")

                self.w_As.append(w_a)
                self.w_Bs.append(w_b)

                # in-place replacement, just like blk.attn.qkv = _LoRA_qkv_timm(...)
                setattr(mlp, fc_name, _LoRA_fc_timm(fc, w_a, w_b, ln))

        self.reset_parameters()
        self.lora_convnext = convnext_model
        self.num_features = getattr(convnext_model, "num_features", None)

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def save_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".safetensors")
        num_layer = len(self.w_As)
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        save_file({**a_tensors, **b_tensors}, filename)

    def load_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".safetensors")
        with safe_open(filename, framework="pt") as f:
            for i, w_A in enumerate(self.w_As):
                w_A.weight = Parameter(f.get_tensor(f"w_a_{i:03d}"))
            for i, w_B in enumerate(self.w_Bs):
                w_B.weight = Parameter(f.get_tensor(f"w_b_{i:03d}"))

    def forward(self, x: Tensor, scene_tokens: Tensor = None) -> Tensor:
        """
        EXACT same calling convention as LoRA_ViT_timm:
            tokens = model(x, scene_tokens)
        We just delegate to ConvNeXt's forward_features, which we've made
        ViT-like via timm_ConvNeXt.
        """
        return self.lora_convnext.forward_features(x, scene_tokens)


class ImgEncoder(torch.nn.Module):
    """Extract embeddings from images using timm's Dinov2 models"""
    model_names = (
        "timm/vit_small_patch14_dinov2.lvd142m",
        "timm/vit_base_patch14_dinov2.lvd142m",
        "timm/vit_large_patch14_dinov2.lvd142m",
        "timm/vit_giant_patch14_dinov2.lvd142m",
        "timm/vit_small_patch14_reg4_dinov2.lvd142m",
        "timm/vit_base_patch14_reg4_dinov2.lvd142m",
        "timm/vit_large_patch14_reg4_dinov2.lvd142m",
        "timm/vit_giant_patch14_reg4_dinov2.lvd142m",
        "timm/vit_small_patch16_dinov3.lvd1689m",
        "timm/vit_large_patch16_dinov3.lvd1689m",
        "timm/vit_large_patch16_dinov3.lvd1689m",
        "timm/convnext_tiny.dinov3_lvd1689m"
        )

    def __init__(self, 
                 config,
    ):
        super().__init__()


        model_name = config.model_name
        self.num_prefix_tokens = config.num_scene_tokens
        self.h = config.image_size[1]
        self.w = config.image_size[0]
        if model_name not in self.model_names:
            raise ValueError(f"Unknown model name: {repr(model_name)}")
        else:
            print("loading ", model_name)
        pretrained_cfg_overlay = {
            "file": config.model_weights,
        }
        
        in_chans = config.in_chans if "in_chans" in config else 3
        
        # HACK: to deal with new numpy version that does not allow pickle by default
        # Create a context manager to temporarily modify np.load
        np_load_old = np.load
        np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
        try:
            if "convnext" in model_name:
                self.model = timm.create_model(
                    model_name,
                    pretrained=True,
                    pretrained_cfg_overlay=pretrained_cfg_overlay,
                    num_classes=0,
                    in_chans=in_chans)
            else:
                self.model = timm.create_model(
                    model_name,
                    pretrained=True,
                    pretrained_cfg_overlay=pretrained_cfg_overlay,
                    img_size=(config.image_size[1], config.image_size[0]),
                    num_classes=0,
                    in_chans=in_chans)
        except:
            if "convnext" in model_name:
                self.model = timm.create_model(
                    model_name,
                    pretrained=True,
                    num_classes=0,
                    in_chans=in_chans)
            else:
                self.model = timm.create_model(
                    model_name,
                    pretrained=True,
                    img_size=(config.image_size[1], config.image_size[0]),
                    num_classes=0,
                    in_chans=in_chans)
        np.load = np_load_old
        
        if "convnext" in model_name:
            self.model.__class__ = timm_ConvNeXt
            try:
                self.patch_size = int(self.model.stem[0].stride[0])
            except Exception:
                self.patch_size = 4
        else:
            self.model.__class__ = timm_ViT
            self.patch_size = self.model.patch_embed.patch_size[0]
        
        self.use_lora = config.use_lora
        self.finetune = config.finetune

        self.neck = torch.nn.Linear(self.model.num_features,config.tf_d_model)

        self.num_features = self.model.num_features

        # Adaptation Setting
        if self.use_lora:
            if "convnext" in model_name:
                self.model = LoRA_ConvNeXt_timm(self.model, r=config.lora_rank)
            else:
                self.model = LoRA_ViT_timm(self.model, r=config.lora_rank)
        # Finetuning
        elif self.finetune:
            for param in self.model.parameters():
                param.requires_grad = True
            self.model.train()
        # Frozen
        else:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
            # self.model.train() # here we let it in train mode

        # train the patch embedder
        if in_chans != 3:
            log.info("Training patch embed and pos embed as in channels != 3")
            for name, param in self.model.named_parameters():
                if "convnext" in model_name:
                    if "stem" in name:
                        param.requires_grad = True
                if "patch_embed" in name:
                    param.requires_grad = True
                elif "pos_embed" in name:
                    param.requires_grad = True

        self.grid_mask = GridMask( True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = True

        # feature pooling
        self.use_feature_pooling = config.use_feature_pooling
        if self.use_feature_pooling:
            self.pool_proj = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool1d(self.num_prefix_tokens)
            )
        
        # focus front cam
        self.focus_front_cam = config.focus_front_cam
        self.compress_fc = config.compress_fc
        if self.compress_fc:
            self.compress_fc_layer = torch.nn.Linear(3957, self.num_prefix_tokens)


    
    # def forward(self, data_dict):
    def forward(self, img, scene_tokens):


        B, N, C, H, W = img.size()
        # print("img.shape ", img.shape)
        img = rearrange(img, 'b n c h w -> (b n) c h w')
        # img = img.reshape(B * N, C, H, W)
        if self.use_grid_mask:
            img = self.grid_mask(img)

        scene_tokens = rearrange(scene_tokens, 'b n t c -> (b n) t c')

        # model inference
        if self.use_lora:
            tokens = self.model(img, scene_tokens)
        elif self.finetune:
            tokens = self.model.forward_features(img, scene_tokens)
        else:
            # self.model.eval()
            # with torch.no_grad():
            tokens = self.model.forward_features(img, scene_tokens)

        if self.use_feature_pooling:
            B_, T, D = tokens.shape  # (B*N, num_tokens, dim)
            # Project the sequence of tokens to `num_prefix_tokens` summary tokens
            tokens = self.pool_proj(tokens.transpose(1, 2))  # shape: (B*N, D, T) → (B*N, D, num_prefix_tokens)
            # print("self.use_feature_pooling: ", self.use_feature_pooling)
            tokens = tokens.transpose(1, 2)  # → (B*N, num_prefix_tokens, D)
        elif self.focus_front_cam:
            B_, T, D = tokens.shape  # (B*N, num_tokens, dim)
            tokens = rearrange(tokens, '(b n) t c -> b n t c', b=B, n=N)
            # all front-cam tokens (camera index 0): [B, T, D]
            front =  tokens[:, 0,  :,                       :] 
            if self.compress_fc:
                # print("before front.shape ", front.shape)  
                front = self.compress_fc_layer(front.transpose(1,2)).transpose(1,2)
                # print("front.shape ", front.shape)                     
            # first K tokens from every other camera: [B, N-1, K, D] -> [B, (N-1)*K, D]
            others = tokens[:, 1:, :self.num_prefix_tokens, :].reshape(B, -1, D)
            # concatenate per batch, preserving order: front first, then cam1..camN-1 prefixes
            tokens = torch.cat([front, others], dim=1)  # [B, (N-1)*K, D]
        elif self.num_prefix_tokens > 0:
            tokens = tokens[:,:self.num_prefix_tokens]
        else:
            tokens = tokens

        tokens = self.neck(tokens)
        if not self.focus_front_cam:
            tokens = rearrange(tokens, '(b n) t c -> b (n t) c', b=B, n=N)

        return tokens