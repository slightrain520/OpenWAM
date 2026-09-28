"""WAM（世界与动作联合建模）的架构基类。

这里统一管理视频/动作骨干网络、训练时的流匹配损失和推理时的联合去噪；
各子类实现 ``forward``，决定两路特征如何交互。以下 B 为批大小，
视频潜变量为 [B, C, T_v, H, W]，动作序列为 [B, T_a, D_a]。

三类子架构共用本文件的训练和推理流程：
1. 单系统：动作 token 与视频 token 一起经过视频 DiT；可选普通前馈层或 MoE 专家层。
2. 双系统：视频 DiT 与单独的 ActionDiT 协作；可在完整视频前向后做交叉注意力，
   也可由 ``DualSystemMoTDriver`` 在各层做联合自注意力。
3. 三系统：在视频和动作分支外增加冻结的视觉语言模型（VLM）理解分支。
具体 token 交互方式由子类 ``forward()`` 定义，本基类只约定输入、输出和损失。
"""

import functools
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn

from openwam.model.compile_options import compile_enabled


def _wrap_single_forward(module: nn.Module) -> None:
    """把单个模块的 forward 包在 ``torch.no_grad`` 中；重复调用不会重复包装。

    ``module`` 是待冻结的 nn.Module；原地替换其 forward，无返回值。
    ``no_grad`` 关闭反向图记录，但参数是否可训练仍由 ``requires_grad_`` 控制。
    """
    # 标记位使该操作幂等，避免 forward 被一层层重复包装。
    if getattr(module, "_openwam_no_grad_wrapped", False):
        return
    original_forward = module.forward

    # wraps 保留原 forward 的名称和元数据，便于调试/检查。
    @functools.wraps(original_forward)
    def wrapped(*args, **kwargs):
        with torch.no_grad():
            return original_forward(*args, **kwargs)

    # 只替换当前模块的入口；子模块由 _wrap_forward_in_no_grad 递归处理。
    module.forward = wrapped
    module._openwam_no_grad_wrapped = True


def _wrap_forward_in_no_grad(module: nn.Module) -> None:
    """关闭整个模块子树的前向梯度记录，入参为根模块，无返回值。

    ``modules()`` 遍历根模块及其子模块，因为调用者可能直接调用内部模块。
    注意：冻结父模块也会切断其可训练子模块的梯度，不能用于只训练其内部 LoRA 的场景。
    包装具有幂等标记；已处在 no_grad 中时嵌套调用也不会额外建立梯度图。
    """
    for sub in module.modules():
        _wrap_single_forward(sub)


logger = logging.getLogger(__name__)

# VLM 权重另存为独立目录；此键名前缀用于从架构 safetensors 中排除它们。
VLM_STATE_DICT_PREFIX = "vlm_backbone."


def _exclude_vlm_from_state_dict(state_dict: dict[str, "Tensor"]) -> dict[str, "Tensor"]:
    """按 ``vlm_backbone.`` 前缀过滤参数字典，返回不含 VLM 权重的新字典。

    注意：若未来把可训练 LoRA 挂在 VLM 子树内，也会被过滤；它应注册在架构顶层。
    """
    return {k: v for k, v in state_dict.items() if not k.startswith(VLM_STATE_DICT_PREFIX)}


def _assert_decode_video_supported(vb) -> None:
    """检查视频骨干 ``vb`` 的编码器能否还原像素，不能则抛出 ValueError。

    部分外部表征编码器只提供特征，没有像素解码器；decode_video=True 时应尽早报错。
    """
    enc = vb.external_encoder
    if enc is not None and not enc.properties.pixel_decode:
        raise ValueError(
            f"generate(decode_video=True) but the configured encoder "
            f"({type(enc).__name__}) is irreversible (properties.pixel_decode=False). "
            "Pass decode_video=False to retrieve raw latents."
        )


if TYPE_CHECKING:
    from openwam.model.action_backbone.base import ActionDiTBackbone, SharedActionBackbone
    from openwam.model.video_backbone.base import VideoBackbone

    AnyActionBackbone = Union["ActionDiTBackbone", "SharedActionBackbone"]


@dataclass
class ActionState:
    """联合自注意力路径中逐层传递的可变动作状态。

    ``action_latents``：当前带噪动作 [B,T_a,D_a]；``timestep``：动作噪声时刻；
    ``payload``：动作骨干内部逐层状态。MoT 驱动器在层间更新此容器。
    它仅用于双系统联合自注意力路径；其他架构不需要此容器。
    """

    action_latents: Optional[Tensor] = None
    timestep: Optional[Tensor] = None
    payload: Optional[Any] = None


class BaseWAMArchitecture(ABC, nn.Module):
    """WAM 各架构共用的训练与推理接口；``nn.Module`` 负责注册参数和调用钩子。

    ``cfg`` 是架构配置；基类创建视频骨干，具体子类创建动作骨干并实现 forward。
    训练主线：prepare_inputs → compute_loss → forward → 两路掩码损失；
    推理主线：generate → 多步联合 forward → 更新视频/动作潜变量。

    可选增加 VLM 等骨干；cfg 可为 OmegaConf DictConfig 或普通字典。
    """

    # --- Construction & config resolution ---

    def __init__(self, cfg=None):
        # 初始化 nn.Module，后续赋值的子模块才会被自动注册进参数树。
        super().__init__()
        self.cfg = cfg
        # 基类先占位；视频骨干在下方创建，动作骨干由子类按架构类型创建。
        self.video_backbone: Optional["VideoBackbone"] = None
        self.action_backbone: Optional["AnyActionBackbone"] = None
        # 记录模型默认设备/精度；set_dtype_device 可在构造后同步修改各骨干。
        self._device = torch.device("cuda")
        self._dtype = torch.bfloat16

        # 训练器调用 set_training_runtime 写入这些开关；prepare_inputs 后续读取它们。
        # 两个 boundary 是视频噪声时刻采样区间的比例边界，默认覆盖完整时刻表。
        self._use_gradient_checkpointing = False
        self._use_gradient_checkpointing_offload = False
        self._max_timestep_boundary = 1.0
        self._min_timestep_boundary = 0.0

        # 部署归一化器：输入状态转到训练数值范围，输出动作转回物理量纲。
        self.normalizer = None

        if cfg is not None:
            self._init_video_backbone(cfg)

    @staticmethod
    def _cfg_get(cfg, key, default=None):
        """统一读取字典或对象形式的配置；缺字段时返回 ``default``。"""
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def _init_video_backbone(self, cfg):
        """按配置构造视频骨干网络；``cfg`` 指定骨干名称或部署权重来源。

        返回值为 ``None``，构造结果保存在 ``self.video_backbone``。
        ``_source`` 是部署/恢复已有组件的入口，``name`` 是从注册表选择类型的入口。
        更换外部编码器会改变视频潜空间，只有 from_scratch=True 时才能同步重建 DiT。
        """
        from openwam.model.video_backbone import build_video_backbone

        # vb_cfg 只保存视频骨干的配置；两种配置容器都支持。
        vb_cfg = cfg.get("video_backbone", {}) if isinstance(cfg, dict) else getattr(cfg, "video_backbone", None)
        if vb_cfg is None:
            return

        # source 指已有权重/组件，name 指注册表中的骨干类型；from_scratch 决定是否重建 DiT。
        source = vb_cfg.get("_source") if isinstance(vb_cfg, dict) else getattr(vb_cfg, "_source", None)
        vb_name = vb_cfg.get("name") if isinstance(vb_cfg, dict) else getattr(vb_cfg, "name", None)
        from_scratch = bool(self._cfg_get(vb_cfg, "from_scratch", False))

        # 四种组合：有 encoder 且从头建 DiT → 使用外部编码器；有 encoder 但复用 DiT
        # → 保持原生 VAE（非默认编码器会在下方报错）；无 encoder 且从头建 DiT
        # → 仅重置 DiT；两项均无 → 直接使用预训练骨干。
        enc_cfg = vb_cfg.get("encoder") if isinstance(vb_cfg, dict) else getattr(vb_cfg, "encoder", None)
        external_encoder = None
        # 只有从头训练 DiT 才允许替换视频编码器：换潜空间后原预训练输入层不再匹配。
        if enc_cfg is not None and from_scratch:
            if source is None:
                # 训练：按配置创建外部编码器，随后由骨干接入其潜空间。
                from openwam.model.video_backbone.encoder import build_video_encoder

                external_encoder = build_video_encoder(enc_cfg)
            else:
                # 部署：只按 checkpoint 元信息搭出空结构，权重随后由 load_checkpoint 填入。
                # _ckpt_dir 指向保存的附加文件（例如编码器 manifest），无需原训练机路径。
                ckpt_dir_for_encoder = (
                    vb_cfg.get("_ckpt_dir") if isinstance(vb_cfg, dict) else getattr(vb_cfg, "_ckpt_dir", None)
                )
                external_encoder = self._build_external_encoder_skeleton(enc_cfg, source, ckpt_dir=ckpt_dir_for_encoder)
        elif enc_cfg is not None and source is None:
            # 配置包含 encoder 但未从头建 DiT：默认 wan22_vae 仅作为模板提示，
            # 仍用原生 VAE；其他编码器会改变潜变量通道数，必须直接报错。
            enc_name = ""
            if isinstance(enc_cfg, dict):
                enc_name = str(enc_cfg.get("name", ""))
            else:
                enc_name = str(getattr(enc_cfg, "name", ""))
            if enc_name and enc_name != "wan22_vae":
                raise ValueError(
                    f"video_backbone.encoder.name='{enc_name}' is incompatible "
                    "with from_scratch=false: the pre-trained DiT's first conv "
                    "channels are bound to native Wan VAE's z_dim and cannot "
                    "consume a different encoder's latent space. Set "
                    "from_scratch=true to activate the encoder swap (and re-init "
                    "the DiT), or remove the encoder block to keep the native "
                    "Wan VAE path."
                )
            logger.info(
                "video_backbone.encoder is set but from_scratch=false; "
                "encoder block IGNORED, using native pipe.vae. Set from_scratch=true "
                "to activate the encoder swap."
            )

        text_dim = self._cfg_get(cfg, "text_dim", None)
        text_dim = None if text_dim in (None, 0) else int(text_dim)
        if source is not None:
            # 从已有组件恢复骨干；这里先建结构，具体权重由恢复流程管理。
            ckpt_dir = vb_cfg.get("_ckpt_dir") if isinstance(vb_cfg, dict) else getattr(vb_cfg, "_ckpt_dir", None)
            # materialize_weights 决定恢复时是否立即分配真实权重存储；
            # 这里不会直接加载 state_dict，恢复逻辑由训练器另行完成。
            materialize = bool(
                vb_cfg.get("_materialize_weights")
                if isinstance(vb_cfg, dict)
                else getattr(vb_cfg, "_materialize_weights", False)
            )
            self.video_backbone = build_video_backbone(
                vb_name,
                cfg,
                source=source,
                device="cpu",
                ckpt_dir=ckpt_dir,
                materialize_weights=materialize,
                external_encoder=external_encoder,
                text_dim=text_dim,
            )
        elif vb_name is not None:
            # 常规训练路径：由注册表名称选择视频骨干实现。
            self.video_backbone = build_video_backbone(
                vb_name, cfg, external_encoder=external_encoder, text_dim=text_dim
            )

        # 配置声明的时间压缩率/因果性必须与实际骨干一致。否则数据加载器可能使用错误的
        # 帧数整除规则，视频有效帧掩码也会与潜变量时间轴错位，因此构造时就报错。
        if self.video_backbone is not None:
            # 配置声明必须与编码器实际时间压缩率一致，否则视频掩码长度会错位。
            declared_tc = self._cfg_get(vb_cfg, "temporal_compression", 4)
            declared_causal = self._cfg_get(vb_cfg, "causal_temporal", True)
            actual_tc = self.video_backbone.temporal_compression
            actual_causal = self.video_backbone.causal_temporal
            if external_encoder is not None:
                encoder_src = f"external encoder {type(external_encoder).__name__}"
            else:
                encoder_src = "native VAE"
            if (declared_tc, declared_causal) != (actual_tc, actual_causal):
                raise ValueError(
                    f"video_backbone.temporal_compression / causal_temporal yaml "
                    f"({declared_tc}, {declared_causal}) does not match {encoder_src} "
                    f"({actual_tc}, {actual_causal}). Update the yaml fields to match."
                )

        # 从头训练由骨干的 reinit_for_from_scratch 统一处理：训练路径重置 DiT 权重，
        # 部署路径若有 source 则只对齐结构，之后由 checkpoint 填入训练好的权重。
        # VAE/文本编码器是否冻结由训练配置负责，随机种子在构造前由训练器设置。
        if self.video_backbone is not None and from_scratch:
            # 训练时重新初始化 DiT；部署时 source 已指定，只恢复匹配的结构。
            self.video_backbone.reinit_for_from_scratch(
                external_encoder=external_encoder,
                source=source,
            )

    @staticmethod
    def _build_external_encoder_skeleton(enc_cfg, source, *, ckpt_dir=None):
        """部署时按保存的组件信息重建外部编码器空壳。

        ``enc_cfg`` 给出编码器类型，``source['components']`` 中的 vae 项描述结构，
        ``ckpt_dir`` 指向编码器可能依赖的附加文件。返回编码器模块；权重稍后加载。
        缺少类型或组件时立即报错，避免误用原生 VAE。
    """
        from openwam.model.video_backbone.encoder import _VIDEO_ENCODER_REGISTRY

        # 先从注册表校验名字；不存在的编码器不能凭空恢复。
        enc_name = enc_cfg["name"] if isinstance(enc_cfg, dict) else enc_cfg.name
        if enc_name not in _VIDEO_ENCODER_REGISTRY:
            available = ", ".join(sorted(_VIDEO_ENCODER_REGISTRY)) or "(none)"
            raise KeyError(f"Unknown video encoder '{enc_name}'. Available: {available}")

        # components 是 checkpoint 记录的各模块结构，vae 项代表视频编码器位置。
        components = (source or {}).get("components") if isinstance(source, dict) else None
        if not components:
            raise RuntimeError(
                "Deploy with encoder block but saved config has no "
                "video_backbone.components — cannot reconstruct encoder skeleton. "
                "Re-save the checkpoint with the current code, or strip the "
                "encoder block from config.yaml to fall back to native VAE."
            )
        # next(..., None) 取第一个 vae 描述项，找不到时给出明确错误。
        vae_entry = next((e for e in components if e.get("attr") == "vae"), None)
        if vae_entry is None:
            raise RuntimeError(
                "Deploy with encoder block but components list has no attr=vae "
                "entry to construct the encoder skeleton from."
            )
        encoder_cls = _VIDEO_ENCODER_REGISTRY[enc_name]
        return encoder_cls.from_skeleton(vae_entry, encoder_cfg=enc_cfg, ckpt_dir=ckpt_dir)

    def _resolve_video_dim(self, cfg) -> int:
        """确定视频特征维度：优先读配置，否则读骨干的 ``dim``，仍缺失则报错。"""
        dim = int(cfg.get("video_dim", 0)) if isinstance(cfg, dict) else int(getattr(cfg, "video_dim", 0))
        if dim == 0 and self.video_backbone is not None:
            dim = self.video_backbone.dim
        if not dim:
            raise ValueError("video_dim must be specified in config or inferred from video_backbone")
        return dim

    # --- Backbone composition ---

    @property
    def backbones(self) -> dict[str, nn.Module]:
        """返回当前架构已创建的骨干字典，供设备迁移、调度器和保存逻辑统一遍历。

        子类如三系统可重写此属性，把额外的 VLM 骨干也纳入管理。
        """
        result = {}
        if self.video_backbone is not None:
            result["video_backbone"] = self.video_backbone
        if self.action_backbone is not None:
            result["action_backbone"] = self.action_backbone
        return result

    # --- Action-side properties (delegate to action_backbone) ---

    @property
    def action_scheduler(self):
        """返回动作骨干的流匹配调度器；骨干未创建时抛错。"""
        if self.action_backbone is None:
            raise RuntimeError("action_backbone is not initialized")
        return self.action_backbone.scheduler

    @property
    def video_scheduler(self):
        """返回视频骨干的流匹配调度器；骨干未创建时抛错。"""
        if self.video_backbone is None:
            raise RuntimeError("video_backbone is not initialized")
        return self.video_backbone.scheduler

    @property
    def external_encoder(self):
        """返回外部视频编码器；使用原生 VAE 或无视频骨干时返回 None。
        架构层对外统一提供此属性，调用方不必直接访问骨干内部。"""
        vb = self.video_backbone
        return vb.external_encoder if vb is not None else None

    @property
    def action_dim(self) -> int:
        """动作向量最后一维 D_a；尚无动作骨干时为 0。"""
        return self.action_backbone.action_dim if self.action_backbone is not None else 0

    @property
    def bridge_layers(self) -> tuple:
        """返回视频/动作交互层的编号；无动作骨干时返回空元组。"""
        return self.action_backbone.bridge_layers if self.action_backbone is not None else ()

    @property
    def uses_proprioception(self) -> bool:
        """是否由主干上下文或动作骨干消费机器人本体状态。"""
        return bool(getattr(self, "_use_proprioception_context", False)) or (
            self.action_backbone is not None and self.action_backbone.uses_proprioception
        )

    # --- Proprio-as-context conditioning ---

    def _init_proprio_context(self, cfg, *, text_dim: int = 4096) -> None:
        """按配置创建本体状态投影层，输入维 D_s=state_dim，输出维 D_c=text_dim。

        状态被编码为一个与文本同宽的 token；未启用时不创建 nn.Linear。
        """
        enabled = bool(self._cfg_get(cfg, "use_proprioception", False))
        self._use_proprioception_context = enabled
        self.proprio_encoder: Optional[nn.Module] = None
        self.proprio_dim = 0
        self.context_dim = int(text_dim)
        if not enabled:
            return
        state_dim = int(self._cfg_get(cfg, "state_dim", 0) or 0)
        if state_dim <= 0:
            raise ValueError("use_proprioception=True requires explicit state_dim for context-token proprio.")
        self.proprio_dim = state_dim
        # nn.Linear 执行 y=xW^T+b，把每个样本的 [D_s] 状态投影成 [D_c] 条件特征。
        self.proprio_encoder = nn.Linear(state_dim, self.context_dim)

    def _append_proprio_context_token(self, pipeline_inputs: dict, proprio: Optional[Tensor]) -> dict:
        """把本体状态编码成一个上下文 token，并与文本 token 拼接。

        ``pipeline_inputs['context']`` 为 [B, L, D_c]；``proprio`` 为 [B, D_s]
        或 [B, 1, D_s]。返回新字典，其中 context 为 [B, L+1, D_c]，
        context_mask 为 [B, L+1]；被掩码屏蔽的状态不会成为有效 token，
        启用该条件却未提供状态时会报错。

        可选 ``_proprio_sample_mask`` 指示哪些样本的状态有效；被屏蔽样本的
        token 数值置零，注意力掩码也为 False，避免其状态影响前向和梯度。
        """
        # 复制字典避免改动调用方；内部掩码只在本函数使用，不能传入视频骨干。
        pipeline_inputs = dict(pipeline_inputs)
        sample_mask = pipeline_inputs.pop("_proprio_sample_mask", None)

        # 配置关闭时，原样返回其余条件；打开后必须有编码层和实际状态。
        if not bool(getattr(self, "_use_proprioception_context", False)):
            return pipeline_inputs
        if self.proprio_encoder is None:
            raise RuntimeError("proprio context is enabled but proprio_encoder is not initialized.")
        if proprio is None:
            raise ValueError("use_proprioception=True requires `proprio` from sample['proprio'] or obs['state'].")
        # 统一为 [B,D_s]：单样本 [D_s] 补批轴，[B,1,D_s] 去掉长度为 1 的中间轴。
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        elif proprio.ndim == 3 and proprio.shape[1] == 1:
            proprio = proprio[:, 0, :]
        if proprio.ndim != 2:
            raise ValueError(f"proprio must be [B, D] or [B, 1, D], got shape {tuple(proprio.shape)}")
        if proprio.shape[1] != self.proprio_dim:
            raise ValueError(f"proprio last dim must be {self.proprio_dim}, got {proprio.shape[1]}")

        # context 是文本条件 [B,L,D_c]；单条状态可在 B 个文本样本间复用。
        context = pipeline_inputs["context"]
        if context.shape[0] != proprio.shape[0]:
            if proprio.shape[0] == 1 and context.shape[0] > 1:
                # expand 只创建广播视图，不复制 B 份状态数据。
                proprio = proprio.expand(context.shape[0], -1)
            else:
                raise ValueError(
                    f"Batch mismatch between context and proprio: {context.shape[0]} vs {proprio.shape[0]}"
                )

        # 统一状态掩码为与新增 token 对应的 [B,1] 布尔张量。
        # 输入可为 [B]、[B,1] 或逐维的 [B,1,D_s]；后者沿状态维做 any。
        if sample_mask is None:
            # 未给掩码时，默认所有样本的状态都有效，形状 [B,1]。
            sample_mask = torch.ones(
                (proprio.shape[0], 1),
                dtype=torch.bool,
                device=context.device,
            )
        else:
            sample_mask = sample_mask.to(device=context.device, dtype=torch.bool)
            if sample_mask.ndim == 3:
                # [B,1,D_s] → [B,1]：只要存在一个真实维度，就启用该样本的状态 token。
                sample_mask = sample_mask.any(dim=-1)
            elif sample_mask.ndim == 1:
                # [B] → [B,1]，与一个新增 token 的注意力掩码对应。
                sample_mask = sample_mask.unsqueeze(-1)
            if sample_mask.shape != (proprio.shape[0], 1):
                raise ValueError(
                    f"_proprio_sample_mask shape {tuple(sample_mask.shape)} must be ({proprio.shape[0]}, 1)"
                )

        # nn.Linear 将 [B, D_s] 投影为 [B, D_c]；unsqueeze(1) 插入长度为 1 的序列轴。
        proprio_token = (
            self.proprio_encoder(proprio.to(device=context.device, dtype=self.proprio_encoder.weight.dtype))
            .to(dtype=context.dtype)
            .unsqueeze(1)
        )
        # 布尔掩码转成浮点并补特征轴 [B,1,1]，无效样本的整个 token 被乘为 0；
        # 该样本也不会经此 token 向投影层贡献梯度。
        sample_mask_f = sample_mask.to(proprio_token.dtype).unsqueeze(-1)  # (B, 1, 1)
        proprio_token = proprio_token * sample_mask_f

        context_mask = pipeline_inputs.get("context_mask")
        if context_mask is None:
            # 如果只有每条文本的有效长度 seq_lens [B]，构造 [B,L] 布尔掩码。
            seq_lens = pipeline_inputs.get("seq_lens")
            if seq_lens is not None:
                seq_lens = seq_lens.to(device=context.device)
                # arange 产生位置 0..L-1；广播比较得到各样本的有效 token 位置。
                positions = torch.arange(context.shape[1], device=context.device).unsqueeze(0)
                context_mask = positions < seq_lens.unsqueeze(1)
            else:
                # 连长度都没有时，认为全部 L 个文本 token 有效。
                context_mask = torch.ones((context.shape[0], context.shape[1]), dtype=torch.bool, device=context.device)
        else:
            context_mask = context_mask.to(device=context.device, dtype=torch.bool)

        updated = dict(pipeline_inputs)
        # torch.cat 沿 token 轴拼接，批大小和特征维不变；掩码也须增加对应的一列。
        updated["context"] = torch.cat([context, proprio_token], dim=1)
        updated["context_mask"] = torch.cat([context_mask, sample_mask], dim=1)
        # 状态 token 位于所有文本位置之后，可能落在文本 padding 后面，
        # 因而有效位置不一定连续；后续应以 context_mask 而非 seq_lens 为准。
        return updated

    # --- Device / dtype (top-level authority) ---

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def set_dtype_device(self, dtype: torch.dtype, device: torch.device) -> None:
        """记录目标精度/设备，并让状态投影层与各骨干自行迁移参数；无返回值。"""
        self._dtype = dtype
        self._device = device
        proprio_encoder = getattr(self, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.to(dtype=dtype, device=device)
        for bb in self.backbones.values():
            bb.set_dtype_device(dtype, device)

    # --- Normalizer (deployment) ---

    def attach_normalizer(self, normalizer) -> None:
        """挂载训练时相同的归一化器，供状态预处理和生成动作反归一化使用。

        ``normalizer=None`` 会清除已有归一化器；无返回值。
        """
        self.normalizer = normalizer

    def normalize_deploy_proprio(self, proprio):
        """部署时把原始机器人状态转成 float32 CPU 张量；输入 None 则返回 None。

        若已挂载训练用 normalizer，先把物理单位状态映射到训练时的数值范围。
        返回形状与输入一致，generate 会再转换到模型设备和精度。

        """
        if proprio is None:
            return None

        import numpy as np
        import torch

        # np.asarray 接受列表/NumPy/CPU 张量，并统一为 float32。
        arr = np.asarray(proprio, dtype=np.float32)
        normalizer = getattr(self, "normalizer", None)
        if normalizer is not None:
            arr = normalizer.normalize(arr)
        # from_numpy 与 NumPy 数组共享 CPU 内存，此处不在 GPU 上分配。
        return torch.from_numpy(arr)

    # --- Checkpoint save / load ---

    def save_checkpoint(self, path: str, *, state_dict: dict | None = None) -> None:
        """把模型参数保存为 safetensors 文件，VLM 骨干另行保存。

        ``path`` 是输出文件路径；``state_dict`` 可由分布式训练器先聚合后传入，
        省略时调用 nn.Module.state_dict() 获取当前参数/缓冲区；无返回值。

        """
        from safetensors.torch import save_file

        if state_dict is None:
            state_dict = self.state_dict()
        # 按键名前缀剔除 VLM，避免在 safetensors 中重复保存其共享权重。
        state_dict = _exclude_vlm_from_state_dict(state_dict)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        save_file(state_dict, path)

    def load_checkpoint(self, path: str, strict: bool = True) -> None:
        """从 safetensors 恢复参数；strict=True 时校验非 VLM 参数是否完整。

        ``path`` 为权重文件；返回 None。meta 参数是只记录形状、尚未分配真实存储的
        空壳，需用 assign=True 将加载的张量绑定到参数槽位。

        """
        from safetensors.torch import load_file

        state_dict = load_file(path)
        has_vlm = getattr(self, "vlm_backbone", None) is not None
        # 检测是否存在 meta 参数；普通真实参数仍按原位拷贝，以保持优化器引用。
        has_meta = any(p.device.type == "meta" for p in self.parameters())
        missing, unexpected = self.load_state_dict(state_dict, strict=False, assign=has_meta)
        # VLM 本来就不在此文件中，只允许它缺失；其他缺失/多余键仍视为错误。
        if strict and not has_vlm:
            if missing or unexpected:
                raise RuntimeError(f"Strict load failed: missing={missing}, unexpected={unexpected}")
        elif strict and has_vlm:
            non_vlm_missing = [k for k in missing if not k.startswith(VLM_STATE_DICT_PREFIX)]
            if non_vlm_missing or unexpected:
                raise RuntimeError(
                    f"Strict load failed (VLM keys excluded): missing={non_vlm_missing}, unexpected={unexpected}"
                )

    # --- Training: module management ---

    def init_training_schedulers(self, num_timesteps: int = 1000) -> None:
        """为视频/动作调度器建立训练时刻表及对应噪声强度，返回 None。

        ``num_timesteps`` 是离散时刻数；各路 shift 来自所属骨干，后续
        compute_loss 从这些时刻及 sigma 表采样，推理也沿相同规则构造轨迹。

        未显式设置 shift 时使用调度器默认值；训练和推理读同一骨干配置，
        避免训练噪声分布与推理轨迹不一致。
        """
        # getattr 在骨干无 shift 属性时返回 None，由调度器采用默认 shift。
        for name, bb in self.backbones.items():
            if not hasattr(bb, "scheduler"):
                continue
            kwargs = {"training": True}
            # 视频/动作分别使用自身 shift；未配置时让调度器采用内置默认值。
            shift = getattr(bb, "shift_video" if name == "video_backbone" else "shift_action", None)
            if shift is not None:
                kwargs["shift"] = float(shift)
            bb.scheduler.set_timesteps(num_timesteps, **kwargs)

    def freeze_modules(self, names: list[str]) -> list[str]:
        """按模块路径冻结权重与前向图，返回实际找到并冻结的名称列表。

        例如 names=['video_backbone.vae']。requires_grad_(False) 防止权重更新；
        no_grad 包装使冻结分支不保存反传激活；training=False 关闭训练行为。

        ``nn.Module.get_submodule`` 支持点号路径；不存在的路径会跳过。
        文本编码器/VAE 本就处在 prepare_inputs 的 no_grad 中，额外包装不改变结果；
        冻结 VLM 等训练前向中的分支时，包装可减少激活内存。
        """
        frozen = []
        for name in names:
            try:
                # get_submodule 识别带点号的 nn.Module 子模块路径。
                module = self.get_submodule(name)
            except (AttributeError, KeyError):
                module = None
            if module is not None:
                # 递归关闭模块内参数的 requires_grad；优化器不再更新这些权重。
                module.requires_grad_(False)
                # 遍历 modules() 设置 eval 标志；避免有自引用别名时递归调用 eval()。
                for sub in module.modules():
                    sub.training = False
                _wrap_forward_in_no_grad(module)
                frozen.append(name)
        return frozen

    def get_trainable_modules(self, freeze_list: list[str] = ()) -> dict[str, nn.Module]:
        """返回顶层可训练子模块字典，供优化器收集参数。

        ``freeze_list`` 排除指定名称；仅含至少一个 requires_grad=True 参数的模块入选。

        ``named_children()`` 只遍历顶层模块；实际参数组再由优化器构造代码收集。
        """
        result = {}
        freeze_set = set(freeze_list)
        for name, mod in self.named_children():
            if name in freeze_set:
                continue
            if any(p.requires_grad for p in mod.parameters()):
                result[name] = mod
        return result

    def move_frozen_to_device(self, device: torch.device, names: tuple[str, ...] = ("text_encoder", "vae")) -> None:
        """把指定冻结模块搬到 ``device``；先在架构下查找，再在各骨干下查找。
        """
        for name in names:
            mod = None
            try:
                mod = self.get_submodule(name)
            except (AttributeError, KeyError):
                pass
            if mod is None:
                for bb in self.backbones.values():
                    found = None
                    try:
                        found = bb.get_submodule(name)
                    except (AttributeError, KeyError):
                        found = None
                    if found is not None:
                        mod = found
                        break
            if mod is not None:
                mod.to(device=device)

    def save_assets_for_deployment(self, output_dir: str, cfg) -> None:
        """让各骨干把部署所需组件说明和 tokenizer 等文件写入输出目录。

        ``cfg`` 会被骨干更新，``output_dir`` 是 checkpoint 目录；无返回值。

        在保存 config.yaml 前调用，使部署端可依据 checkpoint 自带信息重建模块，
        不必依赖训练机器上的原始模型路径。
        """
        for bb in self.backbones.values():
            bb.save_deploy_assets(output_dir, cfg)

    # --- Training: preprocessing ---

    @torch.no_grad()
    def preprocess(self, **kwargs) -> dict:
        """把原始视频帧和文本交给视频骨干编码，返回视频潜变量和条件上下文。

        ``**kwargs`` 包含帧、文本及可选参考图；返回字典由具体视频骨干定义。
        ``@torch.no_grad`` 关闭该预处理过程的梯度记录，节省激活显存。
        """
        return self.video_backbone.preprocess_input_for_train(**kwargs)

    def set_training_runtime(
        self,
        *,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        max_timestep_boundary: float = 1.0,
        min_timestep_boundary: float = 0.0,
    ) -> None:
        """保存训练运行选项，随后由 prepare_inputs 传给 compute_loss。

        ``use_gradient_checkpointing`` 控制反传重算；``*_offload`` 控制卸载；
        两个 timestep_boundary 是 [0,1] 范围的采样边界比例；无返回值。
        """
        self._use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self._use_gradient_checkpointing_offload = bool(use_gradient_checkpointing_offload)
        self._max_timestep_boundary = float(max_timestep_boundary)
        self._min_timestep_boundary = float(min_timestep_boundary)

    @torch.no_grad()
    def prepare_inputs(self, batch: list[dict]) -> dict:
        """整理样本并编码条件，供 ``compute_loss(**inputs)`` 直接使用。

        参数 batch 是 B 个样本的列表，也接受单样本字典。每条动作 [T_a,D_a]
        合并为 [B,T_a,D_a]，每条状态 [D_s] 堆叠为 [B,D_s]。
        返回字典含干净视频潜变量 ``input_latents`` [B,C,T_v,H,W]、文本条件、
        动作、有效位置掩码和训练选项；``@torch.no_grad`` 使编码过程不进入反向图。
        """
        from openwam.dataloader.transforms.pipeline import FirstFrameConditioningTransform
        from openwam.model.architectures.utils.common import downsample_video_mask_to_latent

        if isinstance(batch, dict):
            # 单样本字典归一成长度 1 的列表，下面统一按批处理。
            batch = [batch]

        if not hasattr(self, "_pipeline_transform_instance"):
            self._pipeline_transform_instance = FirstFrameConditioningTransform()
        # 给每个样本补齐首帧条件等字段；samples 长度仍为 B。
        samples = [self._pipeline_transform_instance.apply(s) for s in batch]

        _dtype = self.dtype
        _device = self.device

        # 以下列表分别收集 B 条样本的原始视频、文本、可选条件和监督信号。
        all_frames: list = []
        all_prompts: list = []
        all_vace_videos: list = []
        all_ref_images: list = []
        all_actions: list = []
        all_proprios: list = []
        all_proprio_masks: list = []
        all_action_masks: list = []
        all_video_masks: list = []

        for sample in samples:
            # 视频/文本由骨干统一编码，参考图与 VACE 视频是可选条件。
            all_frames.append(sample["video"])
            all_prompts.append(sample["prompt"])
            all_vace_videos.append(sample.get("vace_video"))
            all_ref_images.append(sample.get("first_frame_image"))

            action = sample.get("action")
            if action is not None:
                if isinstance(action, np.ndarray):
                    # from_numpy 把 NumPy 动作数组转为张量；形状不变。
                    action = torch.from_numpy(action)
                # 单样本 [T_a,D_a] → [1,T_a,D_a]，便于随后沿批轴 cat。
                action = action.to(dtype=_dtype, device=_device).unsqueeze(0)
            all_actions.append(action)

            # 只要数据集给出本体状态就收集；主干上下文和动作骨干可各自决定是否使用。
            proprio = sample.get("proprio")
            if proprio is not None:
                if isinstance(proprio, np.ndarray):
                    proprio = torch.from_numpy(proprio)
                proprio = proprio.to(dtype=_dtype, device=_device)
                # 把 [1,D_s] 的单样本状态压成 [D_s]；其他形状直接报错。
                if proprio.ndim == 1:
                    pass
                elif proprio.ndim == 2 and proprio.shape[0] == 1:
                    proprio = proprio[0]
                else:
                    raise ValueError(f"sample['proprio'] must be [D] or [1, D], got shape {tuple(proprio.shape)}")
            all_proprios.append(proprio)

            # proprio_mask 可为 [1]（整条状态开关）或 [1,D_s]（逐维有效性）。
            # 缺省为 [True]；一个批次中两种形状混合时，下方会先统一维度。
            pmask = sample.get("proprio_mask")
            if pmask is None:
                # 没有提供掩码时默认该样本本体状态有效。
                pmask = torch.ones(1, dtype=torch.bool)
            else:
                if isinstance(pmask, np.ndarray):
                    pmask = torch.from_numpy(pmask)
                pmask = pmask.to(dtype=torch.bool)
                if pmask.ndim == 0:
                    pmask = pmask.unsqueeze(0)
            all_proprio_masks.append(pmask)

            amask = sample.get("action_mask", None)
            vmask = sample.get("video_mask", None)
            # 数据集掩码中的 True 表示有效；稍后取反成为 is_pad。
            if isinstance(amask, np.ndarray):
                amask = torch.from_numpy(amask)
            if isinstance(vmask, np.ndarray):
                vmask = torch.from_numpy(vmask)
            all_action_masks.append(amask)
            all_video_masks.append(vmask)

        ref_flags = [r is not None for r in all_ref_images]
        # 一个批次要么全部有参考图，要么全部没有，避免编码器收到混合条件。
        if any(ref_flags) and not all(ref_flags):
            raise ValueError("Mixed reference images in batch: all samples must be consistent.")

        # 编码后得到 input_latents [B,C,T_v,H,W] 和文本 context 等骨干条件。
        preprocessed = self.preprocess(
            frames=all_frames,
            text=all_prompts,
            vace_videos=all_vace_videos,
            ref_images=all_ref_images if ref_flags[0] else None,
        )

        # 前面为每个动作插入批轴 [1,T_a,D_a]，torch.cat 沿该轴得到 [B,T_a,D_a]。
        action_data = torch.cat(all_actions, dim=0) if all_actions[0] is not None else None

        # latents 暂未加噪，留给 compute_loss 创建；其余标志控制本次前向。
        inputs = {
            **preprocessed,
            "latents": None,
            "cfg_scale": 1,
            "cfg_merge": False,
            "tiled": False,
            "use_gradient_checkpointing": self._use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self._use_gradient_checkpointing_offload,
            "max_timestep_boundary": self._max_timestep_boundary,
            "min_timestep_boundary": self._min_timestep_boundary,
            "actions": action_data,
        }

        # 状态存在就写入 inputs；真正使用哪条状态注入路径，由具体架构决定。
        if all_proprios[0] is not None:
            # torch.stack 新建批轴；contiguous 保证结果在内存中连续，便于后续算子使用。
            inputs["proprio"] = torch.stack(all_proprios, dim=0).contiguous()
            # 混合批次必须先将 [1] 开关扩成 [1,D_s]，否则 torch.stack 会因形状不同报错。
            if len({m.ndim for m in all_proprio_masks}) > 1:
                # 混合 [1] 与 [1,D_s] 掩码时，把标量开关广播到每个状态维度。
                pdim = max((m.shape[-1] for m in all_proprio_masks if m.ndim == 2), default=1)
                all_proprio_masks = [
                    m if m.ndim == 2 else m.reshape(m.shape[0], 1).expand(m.shape[0], pdim) for m in all_proprio_masks
                ]
            inputs["proprio_mask"] = torch.stack(all_proprio_masks, dim=0).contiguous()

        if all_action_masks[0] is not None:
            # ~ 把“有效=True”翻成“填充=True”；stack 后形状 [B,T_a] 或 [B,T_a,D_a]。
            inputs["action_is_pad"] = torch.stack([~m for m in all_action_masks], dim=0).to(device=_device)
        if all_video_masks[0] is not None:
            # 仅当首个视频潜变量本身是干净条件帧时才跳过它：TI2V 由
            # first_frame_latents 或骨干 needs_first_frame_skip 标记。
            # Wan I2V 的首帧条件走旁路 y，VACE 走 vace_context；两者的潜变量首帧
            # 仍被加噪并监督，不能误跳过。普通 T2V 也不跳过。
            # 只有潜变量首帧本身被固定为干净条件时，才不对其计算视频损失。
            skip_first = inputs.get("first_frame_latents") is not None or self.video_backbone.needs_first_frame_skip
            # 用实际编码器的 temporal_compression 映射帧掩码；不能假定都是 Wan VAE 的 4 倍。
            temporal_factor = int(self.video_backbone.temporal_compression)
            # 视频帧掩码按编码器时间压缩率映射到潜变量帧轴，再逐样本堆叠。
            latent_masks = [
                downsample_video_mask_to_latent(~m, temporal_factor=temporal_factor, skip_first=skip_first)
                for m in all_video_masks
            ]
            inputs["video_is_pad"] = torch.stack(latent_masks, dim=0).to(device=_device)

        # 所有字段可通过 compute_loss(**inputs) 解包；掩码只用于损失归约。
        return inputs

    # --- Training: loss computation ---

    def compute_loss(
        self,
        *,
        actions: Optional[torch.Tensor] = None,
        lambda_video: float = 1.0,
        lambda_action: float = 1.0,
        **inputs,
    ) -> dict:
        """计算视频与动作两路的流匹配损失。

        输入：``actions`` 是干净动作 [B,T_a,D_a]，可显式传入或放在 inputs 中；
        ``inputs['input_latents']`` 是干净视频 [B,C,T_v,H,W]，其余 inputs 为文本、
        状态、掩码与运行选项，通常由 prepare_inputs 产生。lambda_video/action
        分别控制两项损失的权重。此函数采样时刻、加噪、调用子类 forward、计算损失。
        返回 dict：``loss`` 是可反传标量，``loss_video/action`` 是已 detach 的记录值。
        """
        vb = self.video_backbone
        # 两路调度器各自存离散时刻、噪声比例 sigma 和训练权重。
        action_scheduler = self.action_backbone.scheduler
        _dtype = self.dtype
        _device = self.device

        # actions 允许显式传参或放在 prepare_inputs 返回的字典里；pop 避免重复传给 forward。
        if actions is None:
            actions = inputs.pop("actions", None)
        else:
            inputs.pop("actions", None)

        # 时间边界用时刻表长度换算成整数索引，限定视频噪声水平的抽样范围。
        max_tb = int(inputs.pop("max_timestep_boundary", 1) * len(vb.scheduler.timesteps))
        min_tb = int(inputs.pop("min_timestep_boundary", 0) * len(vb.scheduler.timesteps))
        # input_latents 是干净视频潜变量，首轴长度给出本批样本数 B。
        B = inputs["input_latents"].shape[0]

        # --- Sample video timesteps ---
        # torch.randint 为每个样本独立抽一个离散视频时刻，索引形状 [B]。
        video_timestep_ids = torch.randint(min_tb, max_tb, (B,))

        # timestep 是模型看到的时间条件，sigma 是线性插值所用的噪声比例；均为 [B]。
        video_timesteps = vb.scheduler.timesteps[video_timestep_ids].to(dtype=_dtype, device=_device)
        video_sigmas = vb.scheduler.sigmas[video_timestep_ids].to(dtype=_dtype, device=_device)

        # --- Add video noise (flow-matching: linear interp + velocity target) ---
        # randn_like 生成与干净视频潜变量同形状的标准高斯噪声 [B,C,T_v,H,W]。
        video_noise = torch.randn_like(inputs["input_latents"])
        # view 仅改变形状为 [B,1,1,1,1]，广播到各样本的所有通道、帧和空间位置。
        sigma_bc = video_sigmas.view(B, 1, 1, 1, 1)
        # 逐元素插值：sigma=0 时为干净视频 x0，sigma=1 时为高斯噪声 eps。
        inputs["latents"] = (1 - sigma_bc) * inputs["input_latents"] + sigma_bc * video_noise
        # x_sigma=(1-sigma)x_0+sigma*eps，目标速度 d x_sigma/d sigma=eps-x_0。
        video_target = video_noise - inputs["input_latents"]

        if inputs.get("first_frame_latents") is not None:
            # TI2V 图文生成视频 条件帧保持干净；0:1 保留时间轴，形状仍为 [B,C,1,H,W]。
            inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]

        # --- Prepare action noise ---
        # 允许只训练视频：这时动作相关变量为 None，子类 forward 可跳过动作分支。
        noisy_actions, action_target, action_timesteps, action_timestep_ids, action_sigmas, a_sigma_bc = (
            None,
            None,
            None,
            None,
            None,
            None,
        )
        if lambda_action > 0 and actions is not None:
            # 动作时刻独立于视频时刻抽样，同一样本的两路噪声强度可以不同。
            action_timestep_ids = torch.randint(0, len(action_scheduler.timesteps), (B,))

            # 动作的模型时间条件和噪声比例也是 [B]，不与视频共享随机时刻。
            action_timesteps = action_scheduler.timesteps[action_timestep_ids].to(dtype=_dtype, device=_device)
            action_sigmas = action_scheduler.sigmas[action_timestep_ids].to(dtype=_dtype, device=_device)

            actions = actions.to(dtype=_dtype, device=_device)
            if actions.dim() == 2:
                # 兼容单样本输入 [T_a,D_a] → [1,T_a,D_a]。
                actions = actions.unsqueeze(0)

            # 动作噪声与动作同形 [B,T_a,D_a]；sigma 扩展为可按样本广播的形状。
            # 动作噪声各维度的含义是：B=批大小，T_a=动作时间步数，D_a=动作维度。
            action_noise = torch.randn_like(actions)
            if action_sigmas.dim() == 1:
                # [B] → [B,1,1]，沿动作时间和维度广播。
                a_sigma_bc = action_sigmas.view(B, 1, 1)
            else:
                # 调度器若给出逐时间步 sigma，则只需在末尾补动作维轴。
                a_sigma_bc = action_sigmas.unsqueeze(-1)
            # add_noise 与视频相同：a_sigma=(1-sigma)*a0+sigma*eps；目标为 eps-a0。
            noisy_actions = action_scheduler.add_noise(actions, action_noise, a_sigma_bc)
            action_target = action_scheduler.training_target(actions, action_noise)

        # --- Joint forward pass ---
        # 拷贝一份参数字典：损失侧仍需原始填充掩码，forward 只接收骨干需要的字段。
        forward_inputs = dict(inputs)
        proprio = forward_inputs.pop("proprio", None)  # 本体状态 [B,D_s]，由子类决定如何注入。
        proprio_mask = forward_inputs.pop("proprio_mask", None)
        # 梯度检查点在反传时重算部分前向结果，以减少激活显存。
        use_grad_ckpt = forward_inputs.pop("use_gradient_checkpointing", False)
        use_grad_ckpt_offload = forward_inputs.pop("use_gradient_checkpointing_offload", False)
        # 填充掩码保留在 inputs 中用于损失，不能泄漏给视频骨干的 prepare。
        # 当前架构的注意力本身不消费这种样本级 padding 掩码。
        forward_inputs.pop("action_is_pad", None)
        forward_inputs.pop("video_is_pad", None)

        # 本体状态掩码借内部键传给 _append_proprio_context_token，后者会将该键取出。
        if proprio_mask is not None:
            forward_inputs["_proprio_sample_mask"] = proprio_mask

        # self(...) 经 nn.Module.__call__ 调用子类 forward，也触发架构级前向钩子。
        # 调用 nn.Module.__call__ 以保留前向钩子；输出分别为
        # [B,C,T_v,H,W] 和可选的 [B,T_a,D_a]，与各自速度目标同形。
        # self(...)（即触发 __call__）：PyTorch 会在内部执行一系列复杂的系统框架操作。它会先触发所有的 前向预钩子（Forward Pre-hooks），然后调用 forward 函数，最后再触发 前向后钩子（Forward Post-hooks）。
        # self()函数会调用类的__call__方法
        video_noise_pred, action_noise_pred = self(
            noisy_actions if lambda_action > 0 else None,
            action_timesteps if lambda_action > 0 else None,
            proprio=proprio,
            use_gradient_checkpointing=use_grad_ckpt, #梯度检查点（Gradient Checkpointing） 是一项极度省显存的技术（它在前向传播时不保存中间激活值，反向传播时需要用到再临时重算）。
            use_gradient_checkpointing_offload=use_grad_ckpt_offload,
            **forward_inputs,
            timestep=video_timesteps,
        )

        # 视频预测与目标形状同为 [B,C,T_v,H,W]；内部跳过干净条件帧/填充帧。
        loss_video = self._compute_video_loss(
            video_noise_pred,
            video_target,
            video_timestep_ids,
            inputs,
            _device,
        )

        if lambda_action == 0 or action_noise_pred is None:
            # 仅有视频分支时也保持同一返回结构；detach 的项只用于记录，不反传。
            return {
                "loss": lambda_video * loss_video,
                "loss_video": lambda_video * loss_video.detach(),
                "loss_action": torch.tensor(0.0, device=loss_video.device),
            }

        # 动作预测与目标同为 [B,T_a,D_a]；按有效时刻和有效维度归约。
        loss_action = self._compute_action_loss(
            action_noise_pred,
            action_target,
            action_timestep_ids,
            action_scheduler,
            inputs,
            _device,
        )

        if lambda_video == 0:
            # 视频权重为零时直接采用动作损失；否则计算加权和供 backward() 使用。
            loss = lambda_action * loss_action
        else:
            loss = lambda_video * loss_video + lambda_action * loss_action

        # detach 切断记录值的梯度边，只有 loss 保留计算图。
        result = {
            "loss": loss,
            "loss_video": lambda_video * loss_video.detach(),
            "loss_action": lambda_action * loss_action.detach(),
        }

        return result

    def _compute_video_loss(self, noise_pred, target, timestep_ids, inputs, device):
        """视频逐样本加权均方误差。

        ``noise_pred/target`` 为 [B,C,T_v,H,W]，``timestep_ids`` 为 [B]；
        跳过干净条件帧与填充帧后，返回一个标量损失。
        """
        import torch.nn.functional as F

        num_clean_prefix = int(inputs.get("num_clean_prefix_frames", 0) or 0)
        # video_is_pad：True 表示该潜变量时间位置是填充，不应贡献视频误差。
        video_is_pad = inputs.get("video_is_pad")

        # n_skip 是视频时间轴开头不参与监督的潜变量帧数，依条件路径确定。
        n_skip = 0
        if inputs.get("first_frame_latents") is not None:
            # TI2V 的前缀是干净条件，不是预测目标。Wan 隐式首帧的计数可能为 0，
            # 因此至少跳过 1 帧；VACE 条件走旁路，不进入这个分支。
            n_skip = max(num_clean_prefix, 1)
        elif num_clean_prefix > 0:
            # 只有前缀计数而无显式首帧潜变量时，还要跳过一个编码器条件帧，
            # 与 prepare_inputs 生成的视频掩码保持一致。
            n_skip = num_clean_prefix + 1
        elif video_is_pad is not None and video_is_pad.shape[-1] < noise_pred.shape[2]:
            # 兼容只覆盖尾部帧的掩码：按长度差裁剪预测/目标，使时间轴与掩码对齐。
            n_skip = noise_pred.shape[2] - video_is_pad.shape[-1]

        if n_skip > 0:
            # 只裁剪时间轴 dim=2：[B,C,T_v,H,W] → [B,C,T_v-n_skip,H,W]。
            noise_pred = noise_pred[:, :, n_skip:]
            target = target[:, :, n_skip:]

        vb = self.video_backbone
        # 按每个样本抽中的 timestep_id 查询损失权重 tw，形状 [B]。
        tw = vb.scheduler.linear_timesteps_weights[timestep_ids].to(dtype=torch.float32, device=device)

        # F.mse_loss(reduction='none') 保留逐元素误差 [B,C,T',H,W]，便于先掩码再归约。
        per_element = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
        # mean 仅消去通道和空间轴，得到每帧误差 [B,T']。
        per_frame = per_element.mean(dim=(1, 3, 4))

        if video_is_pad is not None:
            # 掩码长度必须与裁剪后时间轴一致，否则可能把监督施加在错误的帧上。
            if video_is_pad.shape[-1] != noise_pred.shape[2]:
                raise ValueError(
                    f"video_is_pad length {video_is_pad.shape[-1]} does not match "
                    f"trimmed noise_pred T={noise_pred.shape[2]} (n_skip={n_skip}). "
                    "Expected mask sized to T_lat minus leading conditioning latents."
                )
            video_is_pad = video_is_pad.to(device=per_frame.device, dtype=torch.bool)
            # ~ 把“填充=True”翻成“有效=True”；逐帧误差形状始终为 [B,T']。
            valid_mask = ~video_is_pad
            per_frame = per_frame * valid_mask.float()
            # 按每个样本的有效帧数求平均；clamp(min=1) 防止全填充样本除零。
            valid_count = valid_mask.float().sum(dim=1).clamp(min=1)
            per_sample = per_frame.sum(dim=1) / valid_count
        else:
            # 无填充掩码时所有保留帧都参加均值，得到每样本 [B] 损失。
            per_sample = per_frame.mean(dim=1)

        # 先乘各样本的时刻权重，再沿批轴取均值，得到标量。
        return (per_sample * tw).mean()

    def _compute_action_loss(self, noise_pred, target, timestep_ids, scheduler, inputs, device):
        """动作逐样本加权均方误差。

        ``noise_pred/target`` 为 [B,T_a,D_a]，``timestep_ids`` 为 [B]；
        [B,T_a,D_a] 掩码逐元素排除无效维，分母是有效元素数；旧版 [B,T_a]
        掩码按时间步排除，先平均动作维再除以有效步数。两者在所有动作维同样
        有效时数学等价。返回乘以时刻权重后的批均值标量。
        """
        import torch.nn.functional as F

        tw = scheduler.training_weight(timestep_ids).to(dtype=torch.float32, device=device)
        # tw 必须是一条样本一个权重 [B]，不能混入时间/动作维。
        if tw.ndim != 1:
            raise ValueError(f"action loss weights must be per-sample [B], got shape {tuple(tw.shape)}")
        # 保留逐动作维误差 [B,T_a,D_a]，避免无效维度在求均值前混入损失。
        per_element = F.mse_loss(noise_pred.float(), target.float(), reduction="none")

        action_is_pad = inputs.get("action_is_pad")

        if action_is_pad is None:
            # 无掩码时同时平均时间和动作维：[B,T_a,D_a] → [B]。
            per_sample = per_element.mean(dim=(1, 2))
            return (per_sample * tw).mean()

        # 掩码中的 True 表示 padding；取反并转 float 才能与逐元素误差相乘。
        action_is_pad = action_is_pad.to(device=per_element.device, dtype=torch.bool)
        valid_mask_f = (~action_is_pad).float()

        # 新版掩码与逐元素误差同形 [B,T_a,D_a]，可分别筛掉时间位置和动作维。
        if valid_mask_f.shape == per_element.shape:
            # 精细掩码 [B,T_a,D_a]：每个时刻、每个动作维分别决定是否有效。
            weighted = per_element * valid_mask_f
            # 有效误差总和 / 有效元素数，得到 [B]；clamp 避免全无效样本除零。
            per_sample = weighted.sum(dim=(1, 2)) / valid_mask_f.sum(dim=(1, 2)).clamp(min=1)
            return (per_sample * tw).mean()

        # 兼容旧版 [B,T_a] 掩码，或末维 K 与预测 D_a 不同的 [B,T_a,K] 掩码。
        # 后者先用 any 折叠为逐时间步掩码，再计算每步平均误差。
        if valid_mask_f.ndim == 3:
            # 维度数与预测不匹配时按时间步折叠：任一维有效即保留该时间步。
            valid_mask_f = (valid_mask_f > 0).any(dim=-1).float()
        # 先平均动作维 [B,T_a,D_a] → [B,T_a]，再用逐时间步掩码求样本均值。
        per_step = per_element.mean(dim=2)
        per_step = per_step * valid_mask_f
        # 每条样本只除以有效时间步数，得到 [B]；最后乘训练时刻权重并平均为标量。
        valid_count = valid_mask_f.sum(dim=1).clamp(min=1)
        per_sample = per_step.sum(dim=1) / valid_count
        return (per_sample * tw).mean()

    # --- Inference: generation ---

    def _resolve_inactive_action_dims(
        self, active_action_mask: Optional[Tensor], device: torch.device
    ) -> Optional[Tensor]:
        """找出统一动作空间中未启用的维度。

        ``active_action_mask`` 是 [D_a] 布尔张量；返回反向的 [D_a] 掩码，
        全部启用时返回 ``None``。未启用维度在推理时沿解析噪声路径更新。
        显式掩码优先；省略时尝试从统一动作归一化器的目标索引推断。
        """
        if active_action_mask is None:
            # 调用方未给掩码时，从统一动作归一化器中读取本机器人的有效目标维索引。
            normalizer = getattr(self, "normalizer", None)
            active_action_indices = getattr(normalizer, "_dst_index", None)
            unified_action_dim = getattr(normalizer, "_unify_dim", None)
            if (
                active_action_indices is not None
                and unified_action_dim is not None
                and int(unified_action_dim) == self.action_dim
            ):
                # as_tensor 把索引转为当前设备上的整型张量；索引必须落在 [0,D_a)。
                active_action_indices = torch.as_tensor(active_action_indices, device=device, dtype=torch.long)
                if active_action_indices.numel() and (
                    int(active_action_indices.min()) < 0 or int(active_action_indices.max()) >= self.action_dim
                ):
                    raise ValueError(
                        f"Unified action indices must be within [0, {self.action_dim}); "
                        f"got {active_action_indices.tolist()}."
                    )
                # 先把 D_a 维全标为无效，再将该机器人使用的维度标为 True。
                active_action_mask = torch.zeros(self.action_dim, device=device, dtype=torch.bool)
                active_action_mask[active_action_indices] = True

        if active_action_mask is None:
            # 既无显式掩码也无可推断的映射：不固定任何维度。
            return None
        # 统一设备和 bool 类型，并严格检查长度，防止错误屏蔽动作维。
        active_action_mask = torch.as_tensor(active_action_mask, device=device, dtype=torch.bool)
        if active_action_mask.shape != (self.action_dim,):
            raise ValueError(
                f"active_action_mask must have shape ({self.action_dim},); got {tuple(active_action_mask.shape)}."
            )
        # 返回“未启用=True”的反向掩码，供动作去噪循环索引这些列。
        inactive_action_dims = ~active_action_mask
        if not bool(inactive_action_dims.any()):
            return None
        return inactive_action_dims

    @torch.no_grad()
    def generate(
        self,
        schedule,
        prompt: str,
        *,
        vace_video=None, #VACE视频生成多功能模型
        first_frame_image=None,
        num_frames: int = 49,
        action_num_frames: Optional[int] = None,
        height: int = 384,
        width: int = 320,
        seed: int = 42,
        tiled: bool = True,
        input_video_latents: Optional[Tensor] = None,
        num_inference_steps: int = 50,
        shift: float = 5.0,
        tile_size: tuple = None,
        tile_stride: tuple = None,
        dit_cache=None,
        decode_video: bool = True,
        profile: bool = False,
        vace_cache: Optional[dict] = None,
        prompt_embed_cache: Optional[dict] = None,
        proprio: Optional[Tensor] = None,
        cfg_scale: float = 1.0,
        cfg_merge: bool = False,
        active_action_mask: Optional[Tensor] = None,
        **extra_pipeline_inputs: Any,
    ) -> dict:
        """按联合时刻表迭代视频和动作的流匹配去噪。

        ``schedule`` 的每项是 (视频时刻, 动作时刻)；``prompt`` 和可选首帧
        提供条件，``action_num_frames-1`` 为输出动作步数。
        ``num_frames`` 是视频原始帧数，编码后潜变量时间长度 T_v 由骨干决定；
        ``first_frame_image`` 固定观测首帧，``proprio`` 是当前机器人状态；
        ``decode_video`` 控制是否把视频潜变量还原为可见帧，``seed`` 控制初始噪声。
        返回 ``{'video': 帧列表或 None, 'actions': [T_a,D_a] NumPy 数组}``。
        ``@torch.no_grad`` 关闭推理梯度；``cfg_scale>1`` 时融合有/无条件预测。
        ``active_action_mask`` 是 [D_a] 的可选有效维掩码；未给出时尝试从
        normalizer 推断。未启用维度保持在解析噪声轨迹上。
        """
        import time

        from tqdm import tqdm

        # 即使调用方绕过部署加载器，也在此切换推理模式；重复调用 eval() 安全。
        # eval() 切换整个模块树到推理模式，关闭 dropout 等训练专有行为。
        self.eval()

        # vb 负责条件编码和可选视频解码；device/dtype 要与生成张量一致。
        vb = self.video_backbone
        device = self.device
        dtype = self.dtype

        t0 = time.time()

        # CFG 系数小于 1 不被此接口接受；CFG 与当前 DiT 缓存不兼容，
        # 因为缓存键未区分有条件/无条件速度场，复用会得到错误预测。
        cfg_scale_f = float(cfg_scale)
        if cfg_scale_f < 1.0:
            raise ValueError(f"cfg_scale must be >= 1.0; got {cfg_scale!r}.")
        if cfg_scale_f > 1.0:
            # CFG 需有/无条件两次预测，旧缓存若不区分条件就会串用，故禁用。
            dit_cache = None

        # 默认视频和动作采样使用相同的原始帧窗口；动作预测长度少 1。
        action_num_frames = int(action_num_frames if action_num_frames is not None else num_frames)

        # 传入 CFG 参数让支持它的骨干准备 uncond_context；Wan 可忽略这些参数。
        # 真正的 CFG 速度融合由下方去噪循环完成。
        # 骨干把 prompt 编成文本条件、准备初始视频潜变量和可选首帧潜变量。
        # 返回的 latents 通常为 [1,C,T_v,H,W]，context 为 [1,L,D_c]。
        inputs_shared = vb.preprocess_input_for_inference(
            prompt=prompt,
            vace_video=vace_video,
            first_frame_image=first_frame_image,
            num_frames=num_frames,
            height=height,
            width=width,
            seed=seed,
            num_inference_steps=num_inference_steps,
            shift=shift,
            tiled=tiled,
            vace_cache=vace_cache,
            prompt_embed_cache=prompt_embed_cache,
            cfg_scale=cfg_scale,
            cfg_merge=cfg_merge,
        )

        if profile:
            if torch.cuda.is_available():
                # CUDA 操作异步；计时前同步，才能测到已完成的准备耗时。
                torch.cuda.synchronize()
            logger.info("[WAM_PROFILE] pipeline_prep: %.3fs", time.time() - t0)

        if input_video_latents is not None:
            # 调用方提供已有视频潜变量时，覆盖骨干刚生成的初始化潜变量。
            inputs_shared["latents"] = input_video_latents
        # 额外条件如三系统的 VLM 隐状态原样转交给相应子类 forward。
        for key, value in extra_pipeline_inputs.items():
            if value is not None:
                # 三系统等子类可通过这些额外键接收 VLM 特征；忽略 None。
                inputs_shared[key] = value
        ref_latents = inputs_shared.get("first_frame_latents")
        if ref_latents is not None:
            # clone 避免原地改写调用方提供的潜变量；把前若干帧固定为观测条件。
            latents = inputs_shared["latents"].clone()
            latents[:, :, : ref_latents.shape[2]] = ref_latents
            inputs_shared["latents"] = latents
        if self.uses_proprioception:
            if proprio is None:
                raise ValueError("use_proprioception=True requires `proprio` during generation.")
            # 状态需要与骨干在同一设备/精度；形状由后续子类按 [D_s]/[B,D_s] 归一。
            inputs_shared["proprio"] = proprio.to(device=device, dtype=dtype)

        # torch.randn 从高斯噪声初始化动作 [1,T_a,D_a]；独立 Generator 固定本次采样种子。
        action_latents = torch.randn(
            1,
            action_num_frames - 1,
            self.action_dim,
            device=device,
            dtype=dtype,
            generator=torch.Generator(device=device).manual_seed(seed),
        )

        # 统一动作空间可能比当前机器人动作宽；多出的零填充维未必参与训练损失，
        # 因此其速度预测不可靠，推理时让这些维度始终遵循 x_sigma=sigma*eps。
        inactive_action_noise = None
        # 统一动作空间中本机器人不使用的维度，其速度预测未受训练约束。
        inactive_action_dims = self._resolve_inactive_action_dims(active_action_mask, device)

        # schedule 中的离散时刻除以训练时刻总数，转为噪声比例 sigma。
        num_train_ts_v = float(self.video_scheduler.num_train_timesteps)
        num_train_ts_a = float(self.action_scheduler.num_train_timesteps)

        t_loop = time.time()

        # 每对相邻时刻构成一步；两路可采用不同 sigma 或暂时只更新其中一路。
        for i in tqdm(range(len(schedule) - 1), desc="Joint denoising"):
            # t_v/t_a 是当前视频/动作时刻；next 为下一步的目标时刻。
            t_v, t_a = schedule[i]
            t_v_next, t_a_next = schedule[i + 1]

            # sigma 越大越接近噪声，去噪一般从大 sigma 向小 sigma 前进。
            sigma_v = t_v / num_train_ts_v
            sigma_a = t_a / num_train_ts_a
            sigma_v_next = t_v_next / num_train_ts_v
            sigma_a_next = t_a_next / num_train_ts_a

            # 某路相邻时刻相同表示本步冻结该路，不更新其潜变量。
            video_stepping = sigma_v != sigma_v_next
            action_stepping = sigma_a != sigma_a_next

            if not video_stepping and not action_stepping:
                continue

            # 冻结一路只意味着“不更新”，仍要把该路送进联合前向作为另一分支的上下文；
            # 若删掉 token，另一分支的注意力条件会改变。
            # torch.tensor([t]) 为单样本建立形状 [1] 的时刻条件。
            v_timestep = torch.tensor([t_v], dtype=dtype, device=device)
            a_timestep = torch.tensor([t_a], dtype=dtype, device=device)

            if (
                dit_cache is not None
                and video_stepping
                and not dit_cache.should_recompute(sigma_v, require_action=action_stepping)
            ):
                # 缓存有效时复用整次联合前向的预测；若动作本步更新，也必须有动作预测缓存。
                # 缓存需同时包含本步会更新的分支预测，才能跳过联合 DiT 前向。
                noise_pred = dit_cache.get_cached()
                action_noise_pred = dit_cache.get_cached_action() if action_stepping else None
            else:
                if cfg_scale_f > 1.0:
                    # CFG 用条件和无条件的速度场组合，作用于视频和动作两路。
                    # _forward_with_cfg 内部负责给每次条件/无条件前向标记 CUDA Graph 步。
                    noise_pred, action_noise_pred = self._forward_with_cfg(
                        action_latents=action_latents,
                        a_timestep=a_timestep,
                        inputs_shared=inputs_shared,
                        v_timestep=v_timestep,
                        cfg_scale=cfg_scale_f,
                        cfg_merge=bool(cfg_merge),
                    )
                else:
                    # 告知 PyTorch CUDA Graph 新一步开始，随后取得与当前潜变量同形的速度预测。
                    torch.compiler.cudagraph_mark_step_begin()
                    noise_pred, action_noise_pred = self.forward(
                        action_latents,
                        a_timestep,
                        **inputs_shared,
                        timestep=v_timestep,
                    )
                if dit_cache is not None and video_stepping:
                    # 保存本步预测供后续时刻复用，避免重复运行大规模 DiT。
                    dit_cache.update(noise_pred, sigma_v, action_noise_pred)

            if video_stepping:
                # 显式 Euler 积分：x_next=x+v*(sigma_next-sigma)，形状仍为 [1,C,T_v,H,W]。
                new_latents = inputs_shared["latents"] + noise_pred * (sigma_v_next - sigma_v)
                ref_latents = inputs_shared.get("first_frame_latents")
                if ref_latents is not None:
                    # 每步更新后重新覆盖条件首帧，否则 Euler 步会逐渐污染观测帧。
                    new_latents = new_latents.clone()
                    new_latents[:, :, : ref_latents.shape[2]] = ref_latents
                inputs_shared["latents"] = new_latents

            if action_stepping and action_noise_pred is not None:
                if inactive_action_dims is not None and inactive_action_noise is None:
                    # 在首次动作更新前，由 x_sigma=sigma*eps 反推出未启用维的固定 eps。
                    # 这些维度对应零填充动作，因此其干净目标 x0=0。
                    sigma_a_f = float(sigma_a)
                    if sigma_a_f <= 0.0:
                        raise ValueError("Cannot initialize inactive action noise from a non-positive sigma.")
                    inactive_action_noise = action_latents[..., inactive_action_dims].detach().clone() / sigma_a_f
                # 动作调度器执行同一 Euler 更新，输出仍为 [1,T_a,D_a]。
                action_latents = self.action_scheduler.flow_step(
                    action_noise_pred, sigma_a, sigma_a_next, action_latents
                )
                if inactive_action_dims is not None:
                    # 覆盖未启用列为 sigma_next*eps，避免使用未经监督的预测速度。
                    action_latents[..., inactive_action_dims] = inactive_action_noise * float(sigma_a_next)

        if profile:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            logger.info("[WAM_PROFILE] denoising_loop: %.3fs", time.time() - t_loop)

        # 请求视频输出时先检查编码器是否可逆；不可逆编码器无法还原像素帧。
        if decode_video:
            # 外部编码器可能不可逆；只有支持像素解码才可还原视频帧。
            _assert_decode_video_supported(vb)
            video_frames = vb.decode_video(inputs_shared["latents"], tiled=tiled)
        else:
            video_frames = None

        # squeeze(0) 去掉单样本批轴，再转 float32/CPU/NumPy 供机器人侧读取。
        actions = action_latents.squeeze(0).float().cpu().numpy()
        normalizer = getattr(self, "normalizer", None)
        if normalizer is not None:
            # 把训练时的归一化动作恢复到真实机器人动作量纲。
            actions = normalizer.unnormalize(actions)

        return {"video": video_frames, "actions": actions}

    # --- §15: Classifier-Free Guidance helpers (inference-time) ---

    def _forward_with_cfg(
        self,
        *,
        action_latents: Optional[Tensor],
        a_timestep: Optional[Tensor],
        inputs_shared: dict,
        v_timestep: Tensor,
        cfg_scale: float,
        cfg_merge: bool,
    ) -> tuple:
        """执行无分类器引导（CFG）：v=v_uncond+s*(v_cond-v_uncond)。

        输入为动作潜变量 [B,T_a,D_a]、共享条件和两路时刻；返回视频与可选
        动作速度预测，形状分别为 [B,C,T_v,H,W]、[B,T_a,D_a]。
        ``cfg_merge=False``：顺序运行两次 forward；先用正常 context，再临时替换为
        uncond_context，最后恢复。``cfg_merge=True``：把两组数据沿批轴拼成 2B，
        一次 forward 后拆开；速度快但峰值显存更高。两路共用同一个 cfg_scale。
        """
        # 无条件文本由视频骨干在推理预处理时生成；缺失时不能计算 CFG。
        uncond_context = inputs_shared.get("uncond_context")
        if not isinstance(uncond_context, Tensor):
            raise RuntimeError(
                "CFG combine requested but `inputs_shared['uncond_context']` is missing "
                "or not a tensor. `preprocess_input_for_inference` should populate it when "
                "cfg_scale > 1.0."
            )

        if cfg_merge:
            # 对所有带批轴的张量做 [无条件, 有条件] 堆叠：B → 2B。
            expanded, exp_al, exp_vt, exp_at = _expand_inputs_for_cfg(
                inputs_shared,
                action_latents=action_latents,
                v_timestep=v_timestep,
                a_timestep=a_timestep,
            )
            torch.compiler.cudagraph_mark_step_begin()
            # 合并前向输出的第 0 轴长 2B，前一半无条件，后一半有条件。
            merged_noise, merged_action = self.forward(exp_al, exp_at, **expanded, timestep=exp_vt)
            # chunk(2, dim=0) 将合并后的 2B 批次拆回无条件/有条件两个 B 批次。
            uncond_noise, cond_noise = merged_noise.chunk(2, dim=0)
            noise_pred = _combine_cfg(uncond_noise, cond_noise, cfg_scale)
            if isinstance(merged_action, Tensor):
                # 如果架构还输出动作速度，对动作分支执行同一引导公式。
                uncond_action, cond_action = merged_action.chunk(2, dim=0)
                action_noise_pred = _combine_cfg(uncond_action, cond_action, cfg_scale)
            else:
                action_noise_pred = None
            return noise_pred, action_noise_pred

        # 顺序路径：先保留原 context 运行有条件前向，再临时替换为无条件 context。
        # 两次 mark_step_begin 让 CUDA Graph 区分两次前向调用。
        torch.compiler.cudagraph_mark_step_begin()
        cond_noise, cond_action = self.forward(action_latents, a_timestep, **inputs_shared, timestep=v_timestep)
        saved_context = inputs_shared["context"]
        inputs_shared["context"] = uncond_context
        try:
            torch.compiler.cudagraph_mark_step_begin()
            uncond_noise, uncond_action = self.forward(action_latents, a_timestep, **inputs_shared, timestep=v_timestep)
        finally:
            # 即使无条件前向抛错，也恢复调用方字典的原始文本条件。
            inputs_shared["context"] = saved_context

        # 最终速度 = 无条件速度 + 引导强度 × (有条件 - 无条件)。
        noise_pred = _combine_cfg(uncond_noise, cond_noise, cfg_scale)
        if isinstance(cond_action, Tensor) and isinstance(uncond_action, Tensor):
            action_noise_pred = _combine_cfg(uncond_action, cond_action, cfg_scale)
        else:
            # One side dropped the action stream; keep cond as-is.
            action_noise_pred = cond_action
        return noise_pred, action_noise_pred

    # --- Deploy helpers (combine action module + video backbone) ---

    def apply_compile_optimizations(self, compile_cfg) -> None:
        """Apply architecture-specific deploy-time compile optimizations."""
        _ = compile_enabled(compile_cfg, default=False, strict=True)
        vb_compile = getattr(getattr(self, "video_backbone", None), "apply_compile_optimizations", None)
        if callable(vb_compile):
            vb_compile(compile_cfg)

    @abstractmethod
    def forward(
        self,
        noisy_actions: Optional[Tensor],
        action_timestep: Optional[Tensor],
        *,
        proprio: Optional[Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **pipeline_inputs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """由子类实现视频与动作联合前向。

        ``noisy_actions`` 为 [B,T_a,D_a] 或 None，``action_timestep`` 为 [B]；
        ``pipeline_inputs`` 含 [B,C,T_v,H,W] 视频潜变量和条件。
        返回 (视频速度 [B,C,T_v,H,W], 可选动作速度 [B,T_a,D_a])。
        动作输入为 None 时，子类可只返回视频预测，动作预测为 None。
        """
        ...


# ----------------------------------------------------------------------
# §15 — Classifier-Free Guidance helpers (module-level so they stay
# stateless / testable without a full architecture instance).
# ----------------------------------------------------------------------


def _combine_cfg(uncond: Tensor, cond: Tensor, scale: float) -> Tensor:
    """组合两个同形速度张量，返回 ``uncond + scale*(cond-uncond)``。

    ``scale=1`` 得到纯条件预测；大于 1 会放大条件与无条件的差值。
    """
    return uncond + float(scale) * (cond - uncond)


# 这些键的张量首轴都是批轴；CFG 合并路径必须与 context 一样从 B 复制到 2B。
_CFG_BATCH_AXIS_KEYS: tuple = (
    "latents",
    "input_latents",
    "proprio",
    "first_frame_latents",
    "seq_lens",
    "context_mask",
    # cosmos_predict25 TI2V emits ``condition_mask`` of shape (B, 1, T_lat, H_lat, W_lat)
    # in ``_finalize_ti2v_inputs`` and the wrapper cats it to ``x_in`` along
    # dim=1; cfg_merge=True must double B here or that cat shape-mismatches.
    "condition_mask",
)


def _expand_inputs_for_cfg(
    inputs_shared: dict,
    *,
    action_latents: Optional[Tensor],
    v_timestep: Tensor,
    a_timestep: Optional[Tensor],
) -> Tuple[dict, Optional[Tensor], Tensor, Optional[Tensor]]:
    """将无条件/有条件输入沿批轴合并，以一次前向计算 CFG。

    ``torch.cat(..., dim=0)`` 把各输入的 B 批次扩为 2B；返回扩展后的
    输入字典、动作潜变量及视频/动作时刻，未带批轴的标量保持原样。
    返回元组依次为扩展后的字典、动作潜变量、视频时刻、动作时刻。
    ``uncond_context`` 在扩展字典中清空，其他带批轴的张量按键表复制。
    """
    uncond_context = inputs_shared["uncond_context"]
    cond_context = inputs_shared["context"]
    # 浅拷贝字典，仅替换本函数需要扩展的张量，不改调用方的键映射。
    expanded = dict(inputs_shared)
    # 无条件在前、有条件在后，与 _forward_with_cfg 的 chunk 顺序严格对应。
    expanded["context"] = torch.cat([uncond_context, cond_context], dim=0)
    expanded["uncond_context"] = None

    # CFG 合并发生在子类 forward 的状态形状归一化之前，故此处先统一为 [B,D_s]。
    proprio = expanded.get("proprio")
    if isinstance(proprio, Tensor):
        if proprio.ndim == 1:
            # [D_s] → [1,D_s]，否则沿 dim=0 cat 会误变成 [2*D_s]。
            expanded["proprio"] = proprio.unsqueeze(0)
        elif proprio.ndim == 3 and proprio.shape[1] == 1:
            # [B,1,D_s] → [B,D_s]，统一状态布局后再扩批。
            expanded["proprio"] = proprio[:, 0, :]

    for key in _CFG_BATCH_AXIS_KEYS:
        v = expanded.get(key)
        if isinstance(v, Tensor):
            # torch.cat([v,v],dim=0) 复制同一个非文本条件到两组 CFG 样本。
            expanded[key] = torch.cat([v, v], dim=0)
    # 动作潜变量和两路时刻也必须扩批，否则会与 2B 个视频/文本条件不匹配。
    al = torch.cat([action_latents, action_latents], dim=0) if isinstance(action_latents, Tensor) else None
    vt = torch.cat([v_timestep, v_timestep], dim=0) if isinstance(v_timestep, Tensor) else v_timestep
    at = torch.cat([a_timestep, a_timestep], dim=0) if isinstance(a_timestep, Tensor) else None
    return expanded, al, vt, at
