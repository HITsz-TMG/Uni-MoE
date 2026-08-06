# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright 2025 The UniMoE Team and HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""
Inference-only UniMoE-2.5 (text + vision + audio MoE) for vLLM v0.22.1.

Differences from upstream Qwen3-VL-MoE (the parent class):
  1. SparseMoE block has 4 ungated shared experts (HF stores them as
     ``mlp.shared_mlp``); we plug a plain MLP into FusedMoE's
     ``shared_experts`` slot so the FusedMoE dispatch optimisation is
     preserved while keeping HF-faithful "no sigmoid gate" semantics.
  2. Attention has element-wise output gating: ``q_proj`` emits
     ``2 * q_size`` channels, half query / half gate (sigmoid).
  3. Sliding-window / full-attention layers are interleaved via
     ``config.layer_types`` and use a separate ``swa_rope_theta``.
  4. Fused-expert weights ship as 2D ``[E*H, 2I]`` / ``[E*I, H]``;
     a one-pass reshape filter on the weight stream lets the parent
     loader handle them with no further changes.
  5. Audio branch (AudioFlamingo3 encoder + DeepStack projectors) is
     added on top; visual and audio deepstack features ride the same
     buffer machinery the parent already provides.
  6. DeepStack inputs keep a fixed ``IntermediateTensors`` structure for
     vLLM 0.22 compiled execution; text/profile calls use zero-valued views
     so the first Dynamo graph cannot erase the multimodal additions.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import BatchFeature

from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.config.multimodal import AudioDummyOptions, BaseDummyOptions
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.inputs import MultiModalDataDict
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import PromptReplacement, PromptUpdate
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
)
# AudioFlamingo3Encoder is public. The two underscore-prefixed helpers are
# internal APIs, so compatibility is intentionally pinned to vLLM 0.22.1.
from vllm.model_executor.models.audioflamingo3 import (
    AudioFlamingo3Encoder,
    _build_audio_encoder_attention_mask,
    _get_audio_post_pool_output_lengths,
)
from vllm.model_executor.models.qwen2_vl import Qwen2VLMultiModalDataParser
from vllm.model_executor.models.qwen3_moe import (
    Qwen3MoeDecoderLayer,
    Qwen3MoeMLP,
    Qwen3MoeSparseMoeBlock,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
)
from vllm.model_executor.models.qwen3_vl_moe import (
    Qwen3MoeLLMForCausalLM,
    Qwen3MoeLLMModel,
    Qwen3VLMoeForConditionalGeneration,
    Qwen3VLMoeMixtureOfExperts,
    Qwen3VLMoeProcessingInfo,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    extract_layer_index,
    maybe_prefix,
)

logger = init_logger(__name__)


# === SECTION_PROCESSING ===
# -----------------------------------------------------------------------------
# Multimodal processor scaffolding
# -----------------------------------------------------------------------------
# These three classes wire UniMoE-2.5's audio branch into vLLM's multimodal
# pipeline. Image/video paths are inherited unchanged from Qwen3-VL-MoE.
# -----------------------------------------------------------------------------


class UniMoE2d5ProcessingInfo(Qwen3VLMoeProcessingInfo):
    """Adds audio-side bookkeeping on top of the inherited Qwen3-VL-MoE info."""

    def get_hf_config(self):
        # Parent hardcodes Qwen3VLMoeConfig; our config is a sibling class.
        return self.ctx.model_config.hf_config

    # ---- audio token / feature-extractor accessors ----

    def get_audio_token_id(self) -> Optional[int]:
        return getattr(self.get_hf_config(), "audio_token_id", None)

    def get_audio_token_str(self) -> str:
        audio_id = self.get_audio_token_id()
        if audio_id is None:
            return ""
        tok = self.get_tokenizer().convert_ids_to_tokens(audio_id)
        return tok[0] if isinstance(tok, list) else (tok or "")

    def get_feature_extractor(self, **kwargs):
        cached = getattr(self, "_unimoe_feature_extractor", None)
        if cached is not None:
            return cached
        from transformers import AutoFeatureExtractor
        fe = AutoFeatureExtractor.from_pretrained(
            self.ctx.model_config.model,
            trust_remote_code=self.ctx.model_config.trust_remote_code,
        )
        object.__setattr__(self, "_unimoe_feature_extractor", fe)
        return fe

    def get_hf_processor(self, **kwargs):
        # Attach feature_extractor + audio_token onto the real Qwen3-VL
        # processor returned by the parent (idempotent). We can't return a
        # standalone shim: the inherited image-processor chain reads
        # ``image_processor`` / ``tokenizer`` off of whatever this returns.
        proc = super().get_hf_processor(**kwargs)
        if not getattr(proc, "audio_token", None):
            proc.audio_token = self.get_audio_token_str()
        if getattr(proc, "feature_extractor", None) is None:
            proc.feature_extractor = self.get_feature_extractor()
        return proc

    # ---- limits / budget ----

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        limits = dict(super().get_supported_mm_limits())
        if getattr(self.get_hf_config(), "enable_audio", False):
            limits["audio"] = None
        return limits

    def get_data_parser(self) -> MultiModalDataParser:
        # vLLM's ``info.data_parser`` cached_property calls this hook
        # (NOT ``_get_data_parser`` on the Processor). The inherited
        # Qwen3-VL parser ships AudioResampler with ``target_sr=None``
        # because Qwen3-VL has no audio path; downstream resample() then
        # raises "target_sr is not provided". We return a fresh parser
        # with both the audio target_sr and Qwen3-VL's
        # ``video_needs_metadata=True`` set.
        return Qwen2VLMultiModalDataParser(
            self.get_hf_config().vision_config.spatial_merge_size,
            target_sr=int(self.get_feature_extractor().sampling_rate),
            video_needs_metadata=True,
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        result = dict(super().get_mm_max_tokens_per_item(seq_len, mm_counts))
        if not getattr(self.get_hf_config(), "enable_audio", False):
            return result
        # Worst case per audio item: max_chunks × 750 tokens-per-30s-chunk.
        # 30s × 16kHz / 160 hop = 3000 mel frames → conv stride-2 → 1500
        # → avg-pool stride-2 → 750 audio tokens.
        result["audio"] = self.get_max_audio_chunks() * 750
        return result

    # ---- audio shape arithmetic (single source of truth, used in 3 places) ----

    def get_max_audio_chunks(self) -> int:
        return int(getattr(self.get_hf_config(), "max_audio_chunks", 1) or 1)

    def get_audio_chunk_num_frames(self) -> int:
        return int(getattr(self.get_feature_extractor(), "nb_max_frames", 3000) or 3000)

    def get_audio_num_mel_bins(self) -> int:
        return int(getattr(self.get_feature_extractor(), "feature_size", 128) or 128)

    @staticmethod
    def compute_audio_output_lengths(mask: torch.Tensor) -> torch.Tensor:
        """
        Mel-mask → per-item pooled token count.

        Delegates to vllm's ``_get_audio_post_pool_output_lengths`` so the
        token-count formula has exactly one home in the codebase.

        Input  mask : [N, num_chunks, n_frames]   (1 = valid mel frame)
        Output      : [N]                         (token count per sample)
        """
        valid = mask.sum(dim=-1).to(torch.int64)               # [N, C]
        # Apply the conv-then-avg-pool formula chunk-wise, then zero out
        # any chunk that had no valid frames (helper would give 1 for them).
        per_chunk = _get_audio_post_pool_output_lengths(valid.reshape(-1))
        per_chunk = per_chunk.reshape(valid.shape)
        per_chunk = torch.where(valid > 0, per_chunk, torch.zeros_like(per_chunk))
        return per_chunk.sum(dim=-1)


class UniMoE2d5DummyInputsBuilder(Qwen3VLDummyInputsBuilder):
    """Adds a single 30s zero-waveform per audio item to the profile run."""

    info: UniMoE2d5ProcessingInfo  # type: ignore[assignment]

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        text = super().get_dummy_text(mm_counts)
        n_audios = int(mm_counts.get("audio", 0) or 0)
        if n_audios > 0:
            text += self.info.get_audio_token_str() * n_audios
        return text

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        # vLLM 0.22 requires ``mm_options`` for image/video profile inputs.
        mm_data: dict = dict(
            super().get_dummy_mm_data(seq_len, mm_counts, mm_options)
        )
        n_audios = int(mm_counts.get("audio", 0) or 0)
        if n_audios > 0:
            fe = self.info.get_feature_extractor()
            length = int(fe.sampling_rate) * int(fe.chunk_length) \
                * self.info.get_max_audio_chunks()
            audio_overrides = mm_options.get("audio")
            if audio_overrides is not None:
                assert isinstance(audio_overrides, AudioDummyOptions)
            mm_data["audio"] = self._get_dummy_audios(
                length=length,
                num_audios=n_audios,
                overrides=audio_overrides,
            )
        return mm_data


class UniMoE2d5MultiModalProcessor(Qwen3VLMultiModalProcessor):
    """
    Adds audio handling on top of the inherited Qwen3-VL processor.

    Hook overrides:
      _call_hf_processor      run WhisperFeatureExtractor on waveforms, merge
                              ``audio_features`` / ``audio_features_mask``
                              into the parent's BatchFeature
      _get_mm_fields_config   declare those two fields as audio-batched
      _get_prompt_updates     expand <|audio_pad|> to N copies, where N is
                              the post-pooling token count

    (The audio target_sr hook lives on UniMoE2d5ProcessingInfo as
    ``get_data_parser`` — vLLM reads it from ``info.data_parser``, not
    from any method on the Processor.)
    """

    info: UniMoE2d5ProcessingInfo  # type: ignore[assignment]

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        mm_data = dict(mm_data)
        # vLLM uses plural ("audios"), HF uses singular ("audio"); pop both so
        # neither leaks to the Qwen3-VL processor (which would warn-and-drop
        # and then trip _merge_mm_kwargs downstream with a missing field).
        audios = mm_data.pop("audios", None) or mm_data.pop("audio", None)

        processed = super()._call_hf_processor(
            prompt=prompt, mm_data=mm_data, mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )
        if not audios:
            return processed

        # Whisper extractor → fixed [N, max_chunks, n_mel, n_frames] tensors.
        fe = self.info.get_feature_extractor()
        sr = int(getattr(fe, "sampling_rate", 16000))
        chunk_sec = int(getattr(fe, "chunk_length", 30))
        max_chunks = self.info.get_max_audio_chunks()
        n_mel = self.info.get_audio_num_mel_bins()
        n_frames = self.info.get_audio_chunk_num_frames()
        chunk_samples = sr * chunk_sec

        per_feats: list[torch.Tensor] = []
        per_masks: list[torch.Tensor] = []

        audios_list = ([audios] if isinstance(audios, np.ndarray)
                       and audios.ndim == 1 else list(audios))

        # Whisper conv stride-2 + avg-pool stride-2 needs at least five mel
        # frames (~50 ms at 16 kHz, hop=160) to emit one pooled token. Pad
        # damaged or nearly empty inputs to 160 ms so placeholder and encoder
        # token counts stay aligned.
        min_samples = int(sr * 0.16)
        for i, w in enumerate(audios_list):
            arr = np.asarray(w, dtype=np.float32).reshape(-1)
            if arr.shape[0] < min_samples:
                arr = np.pad(arr, (0, min_samples - arr.shape[0]))
                audios_list[i] = arr
        # ===============================================================

        for waveform in audios_list:
            wave = np.asarray(waveform, dtype=np.float32).reshape(-1)
            total = wave.shape[0]
            n_chunks = max(1, min(max_chunks,
                                  (total + chunk_samples - 1) // chunk_samples))

            feats = torch.zeros(max_chunks, n_mel, n_frames, dtype=torch.float32)
            masks = torch.zeros(max_chunks, n_frames, dtype=torch.long)

            for ci in range(n_chunks):
                lo, hi = ci * chunk_samples, min((ci + 1) * chunk_samples, total)
                slc = wave[lo:hi]
                if slc.size == 0:
                    continue
                fe_out = fe([slc], sampling_rate=sr,
                            return_attention_mask=True, return_tensors="pt")
                feats_ci = fe_out["input_features"][0]
                # WhisperFeatureExtractor with return_attention_mask=True
                # already down-samples the sample-level mask to mel-frame
                # resolution (``mask[:, ::hop_length]`` inside HF), so the
                # returned attention_mask is already the per-mel-frame
                # validity mask we want. Trust it.
                mask_key = ("attention_mask" if "attention_mask" in fe_out
                            else "feature_attention_mask")
                mask_ci = (fe_out[mask_key][0] if mask_key in fe_out
                           else torch.ones(n_frames, dtype=torch.long))

                # Force exact rectangle shape on the features.
                if feats_ci.shape != (n_mel, n_frames):
                    feats_ci = F.pad(
                        feats_ci,
                        (0, max(0, n_frames - feats_ci.shape[1]),
                         0, max(0, n_mel - feats_ci.shape[0]))
                    )[:n_mel, :n_frames]
                feats[ci] = feats_ci.to(torch.float32)

                # Defensive shape fix only -- fe should already return
                # shape [n_frames], but pad/truncate just in case the
                # extractor was configured with a non-default chunk_length.
                if mask_ci.shape[0] != n_frames:
                    m = torch.zeros(n_frames, dtype=torch.long)
                    cap = min(n_frames, mask_ci.shape[0])
                    m[:cap] = mask_ci[:cap].to(torch.long)
                    mask_ci = m
                masks[ci] = mask_ci.to(torch.long)

            per_feats.append(feats)
            per_masks.append(masks)

        target = (processed.data if hasattr(processed, "data") else processed)
        target["audio_features"] = torch.stack(per_feats, dim=0)
        target["audio_features_mask"] = torch.stack(per_masks, dim=0)
        return processed

    def _get_mm_fields_config(self, hf_inputs: BatchFeature,
                              hf_processor_mm_kwargs: Mapping[str, object],
                              ) -> Mapping[str, MultiModalFieldConfig]:
        fields = dict(super()._get_mm_fields_config(hf_inputs,
                                                    hf_processor_mm_kwargs))
        fields["audio_features"] = MultiModalFieldConfig.batched("audio")
        fields["audio_features_mask"] = MultiModalFieldConfig.batched("audio")
        return fields

    def _get_prompt_updates(self, mm_items: MultiModalDataItems,
                            hf_processor_mm_kwargs: Mapping[str, object],
                            out_mm_kwargs: MultiModalKwargsItems,
                            ) -> Sequence[PromptUpdate]:
        updates = list(super()._get_prompt_updates(
            mm_items, hf_processor_mm_kwargs, out_mm_kwargs))

        audio_token_id = self.info.get_audio_token_id()
        audio_token_str = self.info.get_audio_token_str()
        if audio_token_id is None or not audio_token_str:
            return updates

        def _replacement(item_idx: int) -> list[int]:
            # MultiModalKwargsItems stores one mapping per modality item.
            # ``MultiModalKwargsItems`` is a Sequence-of-items per modality, and
            # each item is a Mapping from field name to MultiModalFieldElem.
            item = out_mm_kwargs["audio"][item_idx]
            mask = item["audio_features_mask"].data
            if not isinstance(mask, torch.Tensor):
                mask = torch.as_tensor(mask)
            # Add batch dim if needed: helper expects [N, num_chunks, n_frames].
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            length = int(self.info.compute_audio_output_lengths(mask).sum().item())
            return [audio_token_id] * max(1, length)

        updates.append(PromptReplacement(
            modality="audio",
            target=audio_token_str,
            replacement=_replacement,
        ))
        return updates


# === SECTION_MOE ===
# -----------------------------------------------------------------------------
# Sparse MoE block with ungated shared experts
# -----------------------------------------------------------------------------


class UniMoE2d5SparseMoeBlock(Qwen3MoeSparseMoeBlock):
    """
    Standard top-k routed experts via FusedMoE, plus 4 shared experts
    (HF stores them as ``mlp.shared_mlp``).

    The shared MLP is plugged into ``FusedMoE.shared_experts`` so the
    routed/shared compute overlap that vLLM does for Qwen3-style MoE
    blocks still applies; we just don't pass an ``expert_gate``, which
    matches the HF reference's "no sigmoid gate" semantics.

    We inherit (rather than subclass nn.Module) so the parent's
    ``forward`` is reused unchanged AND ``isinstance(layer.mlp,
    Qwen3MoeSparseMoeBlock)`` checks in ``set_moe_parameters`` /
    ``update_physical_experts_metadata`` see us as a Qwen3 MoE block.
    """

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        # Skip parent __init__ -- it builds a shared_expert with sigmoid
        # gating, and assumes ``shared_expert_intermediate_size`` from
        # config. UniMoE has neither convention; we set up the same
        # attrs the parent sets, just without the gate.
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config

        self.tp_size = get_tensor_model_parallel_world_size()
        if self.tp_size > config.num_experts:
            raise ValueError(
                f"TP size {self.tp_size} > num_experts {config.num_experts}")

        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.num_experts
        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        eplb_config = get_current_vllm_config().parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb
        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size
        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        self.gate = ReplicatedLinear(
            config.hidden_size, config.num_experts,
            bias=False, quant_config=quant_config,
            prefix=f"{prefix}.gate",
        )

        # Shared MLP: HF name is `shared_mlp`, ungated, no reduce.
        # Width = n_shared_experts × moe_intermediate_size.
        self.shared_mlp = Qwen3MoeMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.n_shared_experts * config.moe_intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            reduce_results=False,
            expert_gate=None,
            prefix=f"{prefix}.shared_mlp",
        )
        # Parent expects ``shared_expert_gate``; we have no gate.
        # Do NOT alias ``shared_mlp`` to ``shared_expert`` -- nn.Module
        # would register the same module twice in ``named_parameters``,
        # tripping AutoWeightsLoader.
        self.shared_expert_gate = None

        self.experts = FusedMoE(
            shared_experts=self.shared_mlp,
            gate=self.gate,
            num_experts=self.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
        )

    # forward is inherited from Qwen3MoeSparseMoeBlock (identical to ours).


# === SECTION_ATTENTION ===
# -----------------------------------------------------------------------------
# GQA + element-wise output gate + per-layer sliding window
# -----------------------------------------------------------------------------


class UniMoE2d5Attention(nn.Module):
    """
    GQA with two UniMoE-specific tweaks vs. Qwen3MoeAttention:

      (a) Element-wise output gate: q_proj emits ``2 * q_size`` channels
          (Q | gate per head), and attention output is multiplied by
          ``sigmoid(gate)`` before o_proj. We split per query head (not
          per KV head) so GQA grouping doesn't mis-pair channels.
      (b) Per-layer RoPE base: sliding-window layers use ``swa_rope_theta``
          (typically 10k) while full-attention layers use ``rope_theta``
          (typically 5M).
    """

    def __init__(
        self,
        *,
        config,
        layer_idx: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp = get_tensor_model_parallel_world_size()

        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_heads % tp == 0
        if self.total_num_kv_heads >= tp:
            assert self.total_num_kv_heads % tp == 0
        else:
            assert tp % self.total_num_kv_heads == 0
        self.num_heads = self.total_num_heads // tp
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp)
        self.head_dim = getattr(config, "head_dim",
                                config.hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        self.attn_output_gate = bool(getattr(
            config, "elementwise_attn_output_gate", False))
        if getattr(config, "headwise_attn_output_gate", False):
            raise NotImplementedError(
                "UniMoE2.5 vLLM port does not implement headwise_attn_output_gate"
                " yet; the released checkpoint uses elementwise only.")

        # q_proj outputs 2x channels when gating is on; pack as
        # ``total_num_heads * 2`` so QKVParallelLinear treats the doubled
        # output as the q section (k, v sections stay normal).
        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads * (2 if self.attn_output_gate else 1),
            total_num_kv_heads=self.total_num_kv_heads,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Per-layer RoPE base.
        layer_types = getattr(config, "layer_types", None)
        self.is_sliding = (layer_types is not None
                           and layer_idx < len(layer_types)
                           and layer_types[layer_idx] == "sliding_attention")
        # CRITICAL: Qwen3VLDivMoeConfig nests rope settings under text_config.
        # Reading from the top-level config returns None for everything and
        # silently degrades RoPE (rope_theta=None, no MRoPE section), causing
        # positional collapse. Source from text_config; only override
        # rope_theta for sliding layers via swa_rope_theta.
        text_config = getattr(config, "text_config", config)
        rope_params = dict(
            getattr(text_config, "rope_parameters", None)
            or getattr(text_config, "rope_scaling", None)
            or {})
        # Override rope_theta for sliding-attention layers.
        if self.is_sliding:
            swa_theta = getattr(text_config, "swa_rope_theta", None)
            if swa_theta is not None:
                rope_params["rope_theta"] = swa_theta
        # Keep rope_type as-is (Qwen3-VL ckpts write rope_type='default' with
        # mrope_section present; vllm-ascend's get_rope inspects mrope_section
        # directly and returns AscendMRotaryEmbedding when present). Do NOT
        # rewrite to 'mrope' because the Ascend implementation detects
        # mrope_section directly.
        rope_params.setdefault("rope_type", "default")
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=getattr(text_config, "max_position_embeddings",
                                 getattr(config, "max_position_embeddings", 8192)),
            rope_parameters=rope_params,
        )

        # Sliding window per layer. vLLM's scheduler treats the model as
        # full-attention (so prefix caching works) once we move SWA into
        # `interleaved_sliding_window`; here we just read back the per-layer
        # value built by UniMoE2d5ForConditionalGeneration.__init__. See
        # plamo3.py / openpangu.py / jais2.py for the same pattern.
        isw = getattr(config, "interleaved_sliding_window", None)
        if isinstance(isw, list) and layer_idx < len(isw):
            sliding_window = isw[layer_idx]
        elif isinstance(isw, int):
            sliding_window = isw
        else:
            sliding_window = None

        self.attn = Attention(
            self.num_heads, self.head_dim, self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, positions: torch.Tensor,
                hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)

        if self.attn_output_gate:
            # Layout (last dim): [Q | gate | K | V], Q and gate interleaved
            # per query head. View by num_heads (NOT num_kv_heads) so chunk(2)
            # separates Q and gate within the same head -- mirrors the HF
            # reference (which views as (-1, head_dim * 2)).
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
            q_gate = q_gate.view(*q_gate.shape[:-1], self.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*hidden_states.shape[:-1], -1)
            gate = gate.reshape(*hidden_states.shape[:-1], -1)
        else:
            q, k, v = qkv.split(
                [self.q_size, self.kv_size, self.kv_size], dim=-1)
            gate = None

        # Per-head QK-norm.
        q = self.q_norm(q.view(*q.shape[:-1], -1, self.head_dim)).view(q.shape)
        k = self.k_norm(k.view(*k.shape[:-1], -1, self.head_dim)).view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        attn_out = self.attn(q, k, v)

        if gate is not None:
            attn_out = attn_out * torch.sigmoid(gate)
        out, _ = self.o_proj(attn_out)
        return out


# === SECTION_LLM ===
# -----------------------------------------------------------------------------
# Decoder stack: layer / model / causal-LM wrapper
# -----------------------------------------------------------------------------
# We inherit aggressively from Qwen3-VL-MoE upstream:
#   - decoder layer only customises __init__ (different attn + MoE block)
#   - LLM model only customises __init__ (different layer type) + a
#     ``[E*H, 2I] → [E, H, 2I]`` reshape filter that runs before the
#     parent's fused-expert loader (which handles 3D layout natively).
#   - causal-LM wrapper only overrides num_shared_experts.
# -----------------------------------------------------------------------------


class UniMoE2d5DecoderLayer(Qwen3MoeDecoderLayer):
    """One UniMoE-2.5 decoder block; same pre-norm shape, custom attn + MoE."""

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        # Skip parent's __init__ -- it builds Qwen3MoeAttention (no gate)
        # and Qwen3MoeSparseMoeBlock (gated shared expert).
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        layer_idx = extract_layer_index(prefix)

        self.hidden_size = config.hidden_size
        self.self_attn = UniMoE2d5Attention(
            config=config,
            layer_idx=layer_idx,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.self_attn",
        )

        # Honour mlp_only_layers if set (empty for the released checkpoint).
        mlp_only = getattr(config, "mlp_only_layers", []) or []
        decoder_sparse_step = getattr(config, "decoder_sparse_step", 1)
        use_moe = (
            layer_idx not in mlp_only
            and config.num_experts > 0
            and (layer_idx + 1) % decoder_sparse_step == 0
        )
        if use_moe:
            self.mlp = UniMoE2d5SparseMoeBlock(
                vllm_config=vllm_config, prefix=f"{prefix}.mlp")
        else:
            self.mlp = Qwen3MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)


class UniMoE2d5LLMModel(Qwen3MoeLLMModel):
    """
    Text decoder. Inherits forward (incl. deepstack injection) + load_weights
    from Qwen3MoeLLMModel; only changes:

      1. ``decoder_layer_type=UniMoE2d5DecoderLayer`` so each layer is built
         with our gated attention + shared-experts MoE block.
      2. Pre-reshape 2D fused-expert tensors to 3D before they reach the
         parent loader. UniMoE-2.5 checkpoints store
             experts.gate_up_proj : [E*H, 2I]
             experts.down_proj    : [E*I,    H]
         while the parent's ``load_fused_expert_weights`` assumes 3D
         ``[E, ...]``. The reshape is the entire delta.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        # Call grandparent (Qwen3MoeModel) with our layer class.
        from vllm.model_executor.models.qwen3_moe import Qwen3MoeModel
        Qwen3MoeModel.__init__(
            self,
            vllm_config=vllm_config,
            prefix=prefix,
            decoder_layer_type=UniMoE2d5DecoderLayer,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        num_experts = self.config.num_experts

        def _split_fused_experts(stream):
            """
            HF ships expert weights doubly-fused (across experts AND, for the
            gate/up pair, across the two projections). vLLM's FusedMoE wants
            per-expert per-shard weights with standard names so its built-in
            ``expert_params_mapping`` can dispatch to the right (w1/w3/w2)
            slot. We split and transpose here:

              HF: experts.gate_up_proj  (E*H, 2I), per-expert layout (H, 2I)
                  -> per-expert split:  (H, I) gate + (H, I) up
                  -> transpose each:    (I, H)  -- vLLM out-in layout
                  -> emit ``...experts.{e}.gate_proj.weight`` and
                          ``...experts.{e}.up_proj.weight``

              HF: experts.down_proj     (E*I, H), per-expert layout (I, H)
                  -> transpose:         (H, I)  -- vLLM out-in layout
                  -> emit ``...experts.{e}.down_proj.weight``
            """
            import re
            # Accept both the checkpoint's suffix-less fused names and
            # standard safetensors names ending in ``.weight``.
            pat = re.compile(
                r"^(.*\.experts)\.(gate_up_proj|down_proj)(?:\.weight)?$"
            )
            for name, w in stream:
                m = pat.match(name)
                if m is None or w.dim() != 2:
                    yield name, w
                    continue
                prefix, kind = m.group(1), m.group(2)
                if kind == "gate_up_proj":
                    two_I = w.shape[-1]
                    I = two_I // 2
                    # (E*H, 2I) -> view (E, H, 2I)
                    v = w.view(num_experts, -1, two_I)
                    for e in range(num_experts):
                        # per-expert (H, 2I) -> split -> two (H, I) halves
                        # -> transpose each -> (I, H) = (out, in)
                        gate = v[e, :, :I].t().contiguous()
                        up = v[e, :, I:].t().contiguous()
                        yield f"{prefix}.{e}.gate_proj.weight", gate
                        yield f"{prefix}.{e}.up_proj.weight", up
                else:  # down_proj
                    H = w.shape[-1]
                    # (E*I, H) -> view (E, I, H)
                    v = w.view(num_experts, -1, H)
                    for e in range(num_experts):
                        # per-expert (I, H) -> transpose -> (H, I) = (out, in)
                        down = v[e].t().contiguous()
                        yield f"{prefix}.{e}.down_proj.weight", down

        return super().load_weights(_split_fused_experts(weights))


class UniMoE2d5LLMForCausalLM(Qwen3MoeLLMForCausalLM):
    """Causal-LM wrapper around UniMoE2d5LLMModel. Same shape as the parent."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        # Skip parent (which builds Qwen3MoeLLMModel) and grandparent.
        nn.Module.__init__(self)
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = UniMoE2d5LLMModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(
            self.config.vocab_size, self.config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)


class UniMoE2d5MixtureOfExperts(Qwen3VLMoeMixtureOfExperts):
    """
    Same machinery as Qwen3VLMoeMixtureOfExperts, but reports the correct
    shared-expert count. The parent hardcodes ``num_shared_experts = 0``.
    """

    def set_moe_parameters(self):
        super().set_moe_parameters()
        # Parent overwrote to 0 above; restore from config.
        self.num_shared_experts = int(
            getattr(self.language_model.config, "n_shared_experts", 0))


# === SECTION_AUDIO ===
# -----------------------------------------------------------------------------
# Audio encoder and projectors adapted for the vLLM 0.22.1 runtime.
# -----------------------------------------------------------------------------


class _DeepStackProjector(nn.Module):
    """Linear → GELU → Linear, matching the HF DeepStackProjector layout."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _normalize_audio_config_aliases(config: Any) -> Any:
    """Translate Qwen-style field names to Whisper-style ones AND fill in
    safe defaults for fields ``Qwen2AudioEncoder`` reads but the UniMoE
    audio config may omit.

    UniMoE-2.5 checkpoints sometimes ship audio configs with
    ``hidden_size`` / ``num_hidden_layers`` etc. (AutoConfig style);
    ``Qwen2AudioEncoder`` (which ``AudioFlamingo3Encoder`` inherits)
    reads ``d_model`` / ``encoder_layers`` instead. We add the missing
    aliases in-place; if both already exist, that part is a no-op.

    Some defaults (``encoder_layerdrop=0.0`` etc.) come from Whisper's
    base config; setting them to 0 is the standard "no regularization"
    inference setup.

    Accepts dict, SimpleNamespace, or PretrainedConfig.
    """
    from transformers import PretrainedConfig

    # ---- 0. coerce raw dict / SimpleNamespace to PretrainedConfig ----
    # setattr/hasattr below assume an object with attributes, so we
    # convert dict-like input upfront.
    if isinstance(config, dict):
        cfg = PretrainedConfig()
        for k, v in config.items():
            setattr(cfg, k, v)
        config = cfg
    elif not isinstance(config, PretrainedConfig):
        # SimpleNamespace etc.
        cfg = PretrainedConfig()
        for k, v in vars(config).items():
            if not k.startswith("_"):
                setattr(cfg, k, v)
        config = cfg

    # ---- 1. translate Qwen-style names -> Whisper-style names ---------
    for old, new in (("hidden_size", "d_model"),
                     ("num_hidden_layers", "encoder_layers"),
                     ("num_attention_heads", "encoder_attention_heads"),
                     ("intermediate_size", "encoder_ffn_dim"),
                     ("layerdrop", "encoder_layerdrop")):
        if hasattr(config, old) and not hasattr(config, new):
            setattr(config, new, getattr(config, old))

    # ---- 2. fill safe defaults for fields Qwen2AudioEncoder reads ----
    # If the ckpt's audio_config omits these, set inference-time safe
        # defaults consistent with the checkpoint tensor shapes:
    #   conv1.weight: (1280, 128, 3)         -> num_mel_bins=128, d_model=1280
    #   embed_positions.weight: (1500, 1280) -> max_source_positions=1500
    #   layers.0..31                         -> encoder_layers=32
    #   fc1: (5120, 1280)                    -> encoder_ffn_dim=5120
    #   self_attn.q_proj: (1280, 1280)       -> encoder_attention_heads=20
    audio_defaults = {
        "d_model":               1280,
        "encoder_layers":          32,
        "encoder_attention_heads": 20,
        "encoder_ffn_dim":       5120,
        "num_mel_bins":           128,
        "max_source_positions":  1500,
        "encoder_layerdrop":      0.0,
        "dropout":                0.0,
        "activation_dropout":     0.0,
        "attention_dropout":      0.0,
        "activation_function":  "gelu",
        "init_std":              0.02,
        "scale_embedding":      False,
    }
    for k, v in audio_defaults.items():
        if not hasattr(config, k):
            setattr(config, k, v)

    return config


class UniMoE2d5AudioModule(nn.Module):
    """
    Audio encoder + (base + per-llm-layer DeepStack) projectors.

    Inputs:
      audio_features        [B, num_chunks, n_mel, n_frames]
      audio_features_mask   [B, num_chunks,        n_frames]

    Outputs:
      flat_features      [sum(per_sample_len), text_hidden]   (base)
      per_sample_lens    list[int]
      deepstack          dict[llm_layer_idx -> [sum_lens, text_hidden]] | None
    """

    def __init__(
        self,
        audio_config: Any,
        text_hidden_size: int,
        enable_deepstack: bool = False,
        deepstack_llm_layer_indexes: Optional[list[int]] = None,
        deepstack_encoder_layer_indexes: Optional[list[int]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        audio_config = _normalize_audio_config_aliases(audio_config)
        self.audio_encoder = AudioFlamingo3Encoder(audio_config)
        if dtype is not None:
            self.audio_encoder = self.audio_encoder.to(dtype=dtype)

        self.audio_pooler = self.audio_encoder.avg_pooler
        self.encoder_dim = int(
            getattr(self.audio_encoder.config, "hidden_size", None)
            or getattr(self.audio_encoder.config, "d_model", None))

        self.audio_projector = _DeepStackProjector(
            in_dim=self.encoder_dim, out_dim=text_hidden_size)

        # DeepStack projectors + forward hooks to capture encoder hidden states
        # (the encoder.forward doesn't expose output_hidden_states).
        self.enable_deepstack = bool(enable_deepstack)
        self.deepstack_injection_map: list[tuple[int, int]] = []
        self.audio_deepstack_projectors = nn.ModuleDict()
        self._captured_hs: dict = {}
        self._hook_handles: list[Any] = []

        if self.enable_deepstack:
            llm_idx = deepstack_llm_layer_indexes or [0, 1, 2]
            enc_idx = deepstack_encoder_layer_indexes or [8, 16, 24]
            if len(llm_idx) != len(enc_idx):
                raise ValueError("deepstack llm/encoder index lengths differ")
            self.deepstack_injection_map = list(zip(llm_idx, enc_idx))
            for ll, _ in self.deepstack_injection_map:
                self.audio_deepstack_projectors[str(ll)] = _DeepStackProjector(
                    self.encoder_dim, text_hidden_size)
            for _, ee in self.deepstack_injection_map:
                if not 0 <= ee < len(self.audio_encoder.layers):
                    raise ValueError(f"encoder layer index {ee} out of range")
                self._hook_handles.append(
                    self.audio_encoder.layers[ee].register_forward_hook(
                        self._make_hook(ee)))

    def _make_hook(self, enc_idx: int):
        def hook(module, inputs, output):
            self._captured_hs[enc_idx] = (
                output[0] if isinstance(output, tuple) else output)
        return hook

    @staticmethod
    def _chunk_token_lengths(mask: torch.Tensor) -> torch.Tensor:
        """Post-pool length per chunk (zero for empty chunks)."""
        valid = mask.sum(dim=-1).to(torch.int64)               # [B, C]
        per_chunk = _get_audio_post_pool_output_lengths(valid.reshape(-1))
        per_chunk = per_chunk.reshape(valid.shape)
        return torch.where(valid > 0, per_chunk, torch.zeros_like(per_chunk))

    def _pool(self, hs: torch.Tensor) -> torch.Tensor:
        return self.audio_pooler(hs.permute(0, 2, 1)).permute(0, 2, 1)

    @staticmethod
    def _collect_valid(chunked: torch.Tensor, lengths: torch.Tensor):
        """Concatenate valid-prefix slices per sample."""
        bsz, max_chunks, seq, dim = chunked.shape
        feats, sizes = [], []
        for i in range(bsz):
            parts = [chunked[i, j, :min(int(lengths[i, j].item()), seq), :]
                     for j in range(max_chunks)
                     if int(lengths[i, j].item()) > 0]
            sample = torch.cat(parts, dim=0) if parts else chunked.new_zeros((0, dim))
            feats.append(sample)
            sizes.append(sample.shape[0])
        return feats, sizes

    def forward(self, audio_features: torch.Tensor,
                audio_features_mask: torch.Tensor,
                return_deepstack: bool = False):
        if audio_features.dim() != 4:
            raise ValueError(f"audio_features must be 4D, got {tuple(audio_features.shape)}")

        bsz, max_chunks = audio_features.shape[:2]
        enc_dtype = next(self.audio_encoder.parameters()).dtype
        flat = audio_features.reshape(
            bsz * max_chunks, *audio_features.shape[2:]).to(dtype=enc_dtype)
        flat_mask = audio_features_mask.reshape(
            bsz * max_chunks, audio_features_mask.shape[-1])

        # 4D additive attention mask: delegated to vllm's helper. Drops
        # 13 lines of hand-rolled arithmetic; numerically identical.
        attn_mask = _build_audio_encoder_attention_mask(
            flat_mask,
            dtype=self.audio_encoder.conv1.weight.dtype,
            device=self.audio_encoder.conv1.weight.device,
        )

        need_hs = self.enable_deepstack and return_deepstack
        if need_hs:
            self._captured_hs.clear()

        last_hidden = self.audio_encoder(flat, attn_mask)
        hidden = last_hidden.reshape(bsz, max_chunks, last_hidden.shape[1], -1)
        lens = self._chunk_token_lengths(audio_features_mask)
        per_sample, sizes = self._collect_valid(hidden, lens)
        feats_flat = (torch.cat(per_sample, dim=0) if per_sample
                      else hidden.new_zeros((0, hidden.shape[-1])))
        projected = self.audio_projector(
            feats_flat.to(next(self.audio_projector.parameters()).dtype))

        deepstack: Optional[dict] = None
        if need_hs:
            deepstack = {}
            for ll, ee in self.deepstack_injection_map:
                if ee not in self._captured_hs:
                    raise RuntimeError(f"DeepStack hook for enc layer {ee} missed")
                # Layer output is pre-pool; apply same avg-pool the encoder
                # tail applies so temporal resolution lines up with `hidden`.
                pooled = self._pool(self._captured_hs[ee])
                enc_h = pooled.reshape(bsz, max_chunks, pooled.shape[1], -1)
                deep_feats, deep_sizes = self._collect_valid(enc_h, lens)
                if deep_sizes != sizes:
                    raise ValueError(
                        f"DeepStack length mismatch at LLM layer {ll}: "
                        f"base={sizes} deep={deep_sizes}")
                flat_d = (torch.cat(deep_feats, dim=0) if deep_feats
                          else enc_h.new_zeros((0, enc_h.shape[-1])))
                proj = self.audio_deepstack_projectors[str(ll)]
                deepstack[ll] = proj(flat_d.to(next(proj.parameters()).dtype))

        return projected, sizes, deepstack


# === SECTION_TOPLEVEL ===
# -----------------------------------------------------------------------------
# Top-level multimodal class (registered with vLLM's registry)
# -----------------------------------------------------------------------------


@MULTIMODAL_REGISTRY.register_processor(
    UniMoE2d5MultiModalProcessor,
    info=UniMoE2d5ProcessingInfo,
    dummy_inputs=UniMoE2d5DummyInputsBuilder,
)
class UniMoE2d5ForConditionalGeneration(
    Qwen3VLMoeForConditionalGeneration,
    UniMoE2d5MixtureOfExperts,
    SupportsMultiModal,
):
    """
    UniMoE-2.5 = Qwen3-VL-MoE + AudioFlamingo3 audio branch + ungated
    shared experts + element-wise attention output gate.

    Most behaviour rides the parent class unchanged: vision tower,
    deepstack-visual buffer machinery (incl. cross-request zeroing),
    EPLB hooks, forward, load_weights via AutoWeightsLoader. We only
    extend for audio (parse/embed/MRoPE) and for the LLM differences
    (handled inside UniMoE2d5LLMForCausalLM and below).
    """

    # qkv_proj / gate_up_proj are packed at runtime; the third entry
    # ("qkv") is for the vision tower's already-packed QKV. We inherit
    # the language-model side via packed_modules_mapping union below.
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "qkv": ["qkv"],
    }

    # HF checkpoint attribute layout (saved by the HF model) differs from
    # our vLLM tree:
    #   HF                              vLLM
    #   model.visual.*                  visual.*
    #   model.audio.*                   audio.*
    #   model.language_model.*          language_model.model.*
    #   lm_head.*                       language_model.lm_head.*
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.visual.": "visual.",
            "model.audio.": "audio.",
            "model.language_model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
        },
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("audio"):
            return "<|audio_start|><|audio_pad|><|audio_end|>"
        return Qwen3VLForConditionalGeneration.get_placeholder_str(modality, i)

    # ---- construction --------------------------------------------------

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        # We cannot just call super().__init__(): the parent hardcodes
        # ``Qwen3MoeLLMForCausalLM`` as the language-model class and we
        # need our subclass instead. Replicate the parent's body, swap
        # in our LLM class, then add the audio module.
        nn.Module.__init__(self)

        from vllm.tokenizers.registry import cached_tokenizer_from_config
        from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        mm_cfg = vllm_config.model_config.multimodal_config

        self.config = config
        self.model_config = vllm_config.model_config
        self._tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
        self.multimodal_config = mm_cfg
        self.use_data_parallel = mm_cfg.mm_encoder_tp_mode == "data"
        self.video_pruning_rate = mm_cfg.video_pruning_rate
        self.is_multimodal_pruning_enabled = mm_cfg.is_multimodal_pruning_enabled()

        # === SWA config normalization (per vLLM official model dev guide) ===
        # vLLM scheduler treats `sliding_window` as "all layers SWA" by default,
        # which evicts KV cache prematurely on full-attention layers and breaks
        # prefix caching. For interleaved layouts (30 sliding + 6 full in
        # UniMoE-2.5), we rename `sliding_window` -> `interleaved_sliding_window`
        # (a per-layer list) so the scheduler treats us as full-attention while
        # each attention module still receives its own per-layer window via
        # `per_layer_sliding_window`, following the in-tree interleaved-SWA
        # implementations.
        text_cfg = config.text_config
        if (getattr(text_cfg, "sliding_window", None) is not None
                and hasattr(text_cfg, "layer_types")
                and not hasattr(text_cfg, "interleaved_sliding_window")):
            sw = text_cfg.sliding_window
            text_cfg.interleaved_sliding_window = [
                sw if lt == "sliding_attention" else None
                for lt in text_cfg.layer_types
            ]
            text_cfg.sliding_window = None

        # DeepStack-visual bookkeeping (audio rides the same buffer).
        self.use_deepstack = hasattr(config.vision_config, "deepstack_visual_indexes")
        self.deepstack_num_level = (
            len(config.vision_config.deepstack_visual_indexes)
            if self.use_deepstack else 0)
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

        # Vision tower + deepstack buffer.
        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )
            if self.use_deepstack:
                self.deepstack_input_embeds = [
                    torch.zeros(
                        vllm_config.scheduler_config.max_num_batched_tokens,
                        config.text_config.hidden_size,
                    ) for _ in range(self.deepstack_num_level)
                ]
                self.deepstack_input_embeds_num_tokens = 0

        # Language model: our subclass, not parent's.
        with self._mark_language_model(vllm_config):
            self.language_model = UniMoE2d5LLMForCausalLM(
                vllm_config=vllm_config.with_hf_config(config.text_config),
                prefix=maybe_prefix(prefix, "language_model"),
            )

        if not get_pp_group().is_first_rank and self.use_deepstack:
            assert self.language_model.model.start_layer >= self.deepstack_num_level

        # Audio module (only built when both config and limit allow it).
        self.enable_audio = bool(getattr(config, "enable_audio", False))
        self.audio = None
        self.audio_token_id = None
        if self.enable_audio and mm_cfg.get_limit_per_prompt("audio"):
            with self._mark_tower_model(vllm_config, "audio"):
                try:
            # Compatible checkpoints ship audio_deepstack_projectors weights,
            # so the default is True. Explicit `False` in config still wins.
                    self.audio = UniMoE2d5AudioModule(
                        audio_config=config.audio_config,
                        text_hidden_size=config.text_config.hidden_size,
                        enable_deepstack=bool(getattr(
                            config, "enable_audio_deepstack", True)),
                        deepstack_llm_layer_indexes=getattr(
                            config, "audio_deepstack_llm_layer_indexes", None),
                        deepstack_encoder_layer_indexes=getattr(
                            config, "audio_deepstack_encoder_layer_indexes", None),
                        dtype=vllm_config.model_config.dtype,
                    )
                    self.audio_token_id = getattr(config, "audio_token_id", None)
                except ImportError as exc:
                    logger.warning(
                        "UniMoE-2.5 audio init failed (%s); falling back to "
                        "text/vision-only inference.", exc)
                    self.enable_audio = False

        # --------------------------------------------------------------
        # Force-materialize parent class's ``deepstack_input_embeds`` buffer
        # onto the target device.
        #
        # The buffer is allocated above inside ``_mark_tower_model`` ctx (line
        # ~1454), which sets the default device to ``meta`` (vLLM's
        # ``init_empty_weights`` mechanism). For vision-only flows that's fine
        # — ``_set_deepstack_input_embeds`` materializes the buffer in
        # ``copy_()``. But for audio-only requests (image=0, video=0), no
        # vision flow ever materializes them, and the LLM forward's per-layer
        # ``hidden_states + deepstack_input_embeds[i]`` crashes with
        #     RuntimeError: Tensor on device meta is not on the expected
        #     device npu:0!
        # We catch that here, after audio init, by reallocating the buffer
        # on the right device with the right dtype. Cost: ~126 MB at
        # max_num_batched_tokens=8192, hidden=2560, K=3, bf16.
        # --------------------------------------------------------------
        if (self.use_deepstack
                and getattr(self, "deepstack_input_embeds", None)):
            buf0 = self.deepstack_input_embeds[0]
            if buf0.device.type == "meta":
                target_device = vllm_config.device_config.device
                target_dtype = vllm_config.model_config.dtype
                seq_len, dim = buf0.shape
                self.deepstack_input_embeds = [
                    torch.zeros(seq_len, dim,
                                device=target_device,
                                dtype=target_dtype)
                    for _ in range(self.deepstack_num_level)
                ]

        self.packed_modules_mapping = (
            self.packed_modules_mapping
            | self.language_model.packed_modules_mapping)
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)
        self.set_moe_parameters()

    def _get_deepstack_input_embeds(
        self,
        num_tokens: int,
    ) -> IntermediateTensors | None:
        """Return a stable DeepStack input structure for compiled execution.

        vLLM 0.22 skips Dynamo guards in ``VLLM_COMPILE``. Its profiling call
        has no staged multimodal features, while later image/audio requests
        do. Returning ``None`` during profiling would let Dynamo remove the
        DeepStack additions and incorrectly reuse that graph for multimodal
        requests. Zero-valued views keep the keys and tensor structure fixed;
        real requests still use the values staged by the parent class.
        """
        deepstack = super()._get_deepstack_input_embeds(num_tokens)
        if deepstack is not None:
            return deepstack

        buffers = getattr(self, "deepstack_input_embeds", None)
        if not buffers:
            return None

        capacity = buffers[0].size(0)
        if num_tokens > capacity:
            raise RuntimeError(
                "DeepStack placeholder exceeds the persistent buffer: "
                f"num_tokens={num_tokens}, capacity={capacity}"
            )

        if num_tokens:
            for buffer in buffers:
                buffer[:num_tokens].zero_()

        return IntermediateTensors(
            {
                f"deepstack_input_embeds_{idx}": buffer[:num_tokens]
                for idx, buffer in enumerate(buffers)
            }
        )

    # ---- multimodal pipeline (image/video inherited; audio added) ------

    def _parse_and_validate_audio_input(self, **kwargs) -> Optional[dict]:
        af = kwargs.pop("audio_features", None)
        afm = kwargs.pop("audio_features_mask", None)
        if af is None and afm is None:
            return None
        if af is None or afm is None:
            raise ValueError(
                "audio_features and audio_features_mask must be paired")

        def _coerce(t):
            if isinstance(t, list):
                if len(t) == 0:
                    return None
                return torch.stack([
                    x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
                    for x in t
                ])
            return t

        af, afm = _coerce(af), _coerce(afm)
        if af is None or afm is None:
            return None

        # Restore [N, C, M, T] / [N, C, T] if flattened by some batch path.
        if af.dim() == 3:
            af = af.view(afm.shape[0], -1, af.shape[1], af.shape[2])
        if afm.dim() == 2:
            afm = afm.view(af.shape[0], -1, afm.shape[1])
        if af.dim() != 4 or afm.dim() != 3:
            raise ValueError(
                f"audio_features must be 4D, audio_features_mask 3D, got "
                f"{tuple(af.shape)} / {tuple(afm.shape)}")
        return {"audio_features": af, "audio_features_mask": afm}

    def _parse_and_validate_multimodal_inputs(self, **kwargs) -> dict:
        # Take a peek for audio first (so super's pop on kwargs doesn't
        # strip our keys). Audio is parsed via its own consumer below
        # so we can safely pass remaining kwargs to super.
        audio = self._parse_and_validate_audio_input(**kwargs) if (
            "audio_features" in kwargs or "audio_features_mask" in kwargs
        ) else None
        mm = super()._parse_and_validate_multimodal_inputs(**kwargs)
        if audio is not None:
            mm["audio"] = audio
        return mm

    def embed_multimodal(self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        # Parent handles image/video; we append audio after. The parent's
        # ``_compute_deepstack_embeds`` is overridden below so it can tell
        # audio apart from visual by last-dim and route appropriately.
        vision = super().embed_multimodal(**kwargs)

        audio_input = self._parse_and_validate_audio_input(**kwargs) \
            if (("audio_features" in kwargs)
                or ("audio_features_mask" in kwargs)) else None
        if audio_input is None or self.audio is None:
            # Reset stale state from previous requests.
            self._pending_audio_deepstack = None
            return vision

        # Audio: encode base + K deepstack levels (matches HF's
        # ``enable_audio_deepstack: true`` path). Base embed scatters at
        # audio_pad positions in ``embed_input_ids``; deepstack levels are
        # written into ``self.deepstack_input_embeds[ll]`` buffer there too,
        # so the LLM forward's per-layer ``+ deepstack_input_embeds[i]``
        # picks them up at LLM layers 0/1/2 (per config:
        # ``audio_deepstack_llm_layer_indexes: [0,1,2]``).
        flat, sizes, deepstack = self.audio(
            audio_input["audio_features"],
            audio_input["audio_features_mask"],
            return_deepstack=True,
        )
        # ``flat``: (sum_lens, visual_dim).
        # ``deepstack``: {llm_layer_idx: (sum_lens, visual_dim)} or None.

        # ===== FIX: pack main + K deepstack levels + 1 marker col into a
        # single tensor, so vLLM crops main and deepstack synchronously =====
        #
        # vLLM stores the WHOLE tensor returned here in encoder_cache (see
        # gpu_model_runner.py::_execute_mm_encoder) and on subsequent chunks
        # crops it by ``encoder_output[start_idx:end_idx]`` along dim-0 only
        # (see _gather_mm_embeddings). last_dim is never touched.
        #
        # By stuffing deepstack into the same tensor along last_dim we get
        # synchronized cropping for free — and crucially the deepstack data
        # is then stored inside encoder_cache, so when a subsequent prefix-
        # cached chunk hits the cache (and ``embed_multimodal`` is NOT re-
        # invoked), the cached tensor still carries the deepstack info.
        # Storing deepstack on ``self._pending_audio_deepstack`` instead
        # silently drops it on cache hits — which is the root of the
        # ``Audio deepstack L0 size 181 != audio embeds 57`` crash under
        # ``enable_prefix_caching=True`` + ``enable_chunked_prefill=True``.
        #
        # Audio combined layout (per token):
        #   [0, visual_dim)                       -> main embed
        #   [visual_dim, visual_dim*(1+K))        -> K deepstack levels
        #   [visual_dim*(1+K), visual_dim*(1+K)+1)-> marker col (=1.0)
        #
        # The +1 marker keeps audio's last_dim distinct from visual's
        # ``visual_dim*(1+K)``, preserving the existing last_dim-based
        # routing logic in ``_compute_deepstack_embeds`` and
        # ``embed_input_ids`` without churning the rest of the file.
        n_tok = flat.shape[0]
        visual_dim = self.visual_dim
        K = self.deepstack_num_level
        expected_audio_last = visual_dim * (1 + K) + 1

        pack_deepstack = (
            self.use_deepstack
            and K > 0
            and deepstack is not None
            and len(deepstack) > 0
        )

        if pack_deepstack:
            llm_layer_indexes = (
                getattr(self.config,
                        "audio_deepstack_llm_layer_indexes", None)
                or [0, 1, 2])
            if len(llm_layer_indexes) < K:
                raise RuntimeError(
                    f"audio_deepstack_llm_layer_indexes length "
                    f"{len(llm_layer_indexes)} < deepstack_num_level {K}; "
                    f"check model config.")

            deep_parts: list[torch.Tensor] = []
            for level_idx in range(K):
                ll = llm_layer_indexes[level_idx]
                d = deepstack.get(ll)
                if d is None:
                    # Fail loud rather than silently substituting zeros — a
                    # missing level means audio_deepstack_llm_layer_indexes
                    # (config) and the audio module's deepstack_injection_map
                    # have drifted. Silent zero-fill would let evaluation
                    # scores degrade undetectably.
                    raise RuntimeError(
                        f"Audio deepstack missing LLM layer {ll}; "
                        f"audio module returned keys="
                        f"{sorted(deepstack.keys())}. Check "
                        f"audio_deepstack_llm_layer_indexes config and "
                        f"the audio module's deepstack_injection_map.")
                d = d.to(flat.device, flat.dtype)
                if d.shape != (n_tok, visual_dim):
                    raise RuntimeError(
                        f"Audio deepstack L{ll} unexpected shape "
                        f"{tuple(d.shape)}, expected {(n_tok, visual_dim)}")
                deep_parts.append(d)
            marker = torch.ones(n_tok, 1, device=flat.device, dtype=flat.dtype)
            combined = torch.cat([flat, *deep_parts, marker], dim=-1)
            if combined.shape[-1] != expected_audio_last:
                raise RuntimeError(
                    f"Audio combined last_dim {combined.shape[-1]} != "
                    f"expected {expected_audio_last}")
        else:
            # No deepstack path — keep audio as (sum_lens, visual_dim). This
            # branch is the legacy shape for builds without audio deepstack.
            combined = flat

        audio_per_item = list(torch.split(combined, sizes, dim=0))

        # Deepstack now travels inside the multimodal_embeddings tuple and is
        # cropped by vLLM in lockstep with main embed. The legacy stash is
        # cleared so any stale data from earlier code paths cannot leak in.
        self._pending_audio_deepstack = None

        if vision is None:
            return tuple(audio_per_item)
        return tuple(list(vision) + audio_per_item)

    # ---- DeepStack + multimodal embed scatter: route audio vs visual ----

    def _compute_deepstack_embeds(self, inputs_embeds, multimodal_embeddings,
                                  is_multimodal):
        """Override: filter out audio embeds before delegating to parent.

        Parent (Qwen3VL) assumes EVERY mm embed has last-dim
        ``visual_dim * (1+K)`` and splits ``[visual_dim, visual_dim*K]``.

        After the deepstack-packing fix in ``embed_multimodal``, audio embeds
        come in one of two shapes:
          - last_dim = ``visual_dim``                  (legacy: no deepstack)
          - last_dim = ``visual_dim*(1+K) + 1``        (packed: main+K+marker)

        Visual embeds keep their canonical last_dim = ``visual_dim*(1+K)``.
        We filter on visual_full_dim equality so both audio shapes are
        excluded; audio scatter (incl. its deepstack) happens later in
        ``embed_input_ids``.
        """
        if not multimodal_embeddings:
            return super()._compute_deepstack_embeds(
                inputs_embeds, multimodal_embeddings, is_multimodal)

        visual_full_dim = self.visual_dim + self.multiscale_dim
        visual_only = tuple(e for e in multimodal_embeddings
                            if e.shape[-1] == visual_full_dim)

        if not visual_only:
            # Pure audio (no visual): nothing for parent to do. Return zero
            # deepstack buffer + empty mm tuple so parent's downstream
            # ``_merge_multimodal_embeddings`` is a no-op (its is_multimodal
            # mask doesn't cover audio_pad anyway).
            deepstack_input_embeds = inputs_embeds.new_zeros(
                self.deepstack_num_level,
                inputs_embeds.size(0),
                inputs_embeds.size(1),
            )
            return deepstack_input_embeds, ()

        # Visual present: delegate to parent with vision-only subset.
        return super()._compute_deepstack_embeds(
            inputs_embeds, visual_only, is_multimodal)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
        *,
        is_multimodal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Override: handle audio embed scatter (parent only handles visual).

        Mirrors HF ``modeling_unimoe2d5_gated_v2_fused_moe.py`` audio branch:
        find ``input_ids == audio_token_id`` positions and ``masked_scatter``
        the audio embeds there. Parent handles visual via the normal
        ``_compute_deepstack_embeds`` -> ``_merge_multimodal_embeddings``
        path (with our override above filtering audio embeds out).

        Audio packed-embed layout (from ``embed_multimodal``):
            [:, 0:visual_dim]                          -> main embed
            [:, visual_dim:visual_dim*(1+K)]           -> K deepstack levels
            [:, visual_dim*(1+K):visual_dim*(1+K)+1]   -> marker col (discarded)

        Both main and deepstack are cropped synchronously by vLLM along dim-0
        (see gpu_model_runner.py::_gather_mm_embeddings) because they share
        the same tensor — eliminating the (181 vs 57) mismatch under prefix
        caching + chunked prefill.
        """
        visual_dim = self.visual_dim
        K = self.deepstack_num_level
        visual_full_dim = visual_dim + self.multiscale_dim   # visual_dim*(1+K)
        audio_packed_dim = visual_full_dim + 1                # +1 marker

        # Split audio off so we can pass visual-only to the parent.
        audio_packed_embs: list[torch.Tensor] = []
        audio_plain_embs: list[torch.Tensor] = []   # legacy: no deepstack
        visual_embs: list[torch.Tensor] = []
        if multimodal_embeddings:
            for emb in multimodal_embeddings:
                last = emb.shape[-1]
                if last == audio_packed_dim:
                    audio_packed_embs.append(emb)
                elif last == visual_dim:
                    audio_plain_embs.append(emb)
                elif last == visual_full_dim:
                    visual_embs.append(emb)
                else:
                    raise RuntimeError(
                        f"Unexpected mm embed last-dim {last}; expected "
                        f"{visual_dim} (audio no-deepstack), "
                        f"{audio_packed_dim} (audio packed), or "
                        f"{visual_full_dim} (visual)")

        # Defensive: packed and plain audio shouldn't ever coexist.
        if audio_packed_embs and audio_plain_embs:
            raise RuntimeError(
                "Mixed packed and plain audio embeddings in one batch is "
                "not supported; check audio encoder configuration.")

        # Resolve audio_token_id up-front: we need it BEFORE the super()
        # call to mask audio positions out of is_multimodal.
        audio_token_id = getattr(self.config, "audio_token_id", None)

        # === FIX: build a visual-only is_multimodal mask for the parent. ===
        # vLLM's gpu_model_runner._gather_mm_embeddings sets is_mm_embed=True
        # for ALL mm placeholder ranges, image AND audio alike. The parent's
        # _compute_deepstack_embeds + _merge_multimodal_embeddings broadcasts
        # visual embeds across every True position in this mask. With a mixed
        # image+audio request, that means (e.g.) 220 visual embeds vs 345
        # mask positions -> shape-mismatch crash:
        #   "Attempted to assign 220 = 220 multimodal tokens to 345
        #    placeholders"
        # We strip audio_pad positions from the mask so the parent only sees
        # visual placeholders. Our own audio scatter (below) covers audio
        # positions separately via input_ids == audio_token_id.
        visual_is_multimodal = is_multimodal
        if (is_multimodal is not None
                and audio_token_id is not None
                and (audio_packed_embs or audio_plain_embs or visual_embs)):
            audio_pos_mask_full = (input_ids == audio_token_id)
            visual_is_multimodal = torch.logical_and(
                is_multimodal.to(input_ids.device),
                torch.logical_not(audio_pos_mask_full),
            )

        # Parent path: text embedding + (if any) visual embed merge.
        inputs_embeds = super().embed_input_ids(
            input_ids,
            multimodal_embeddings=tuple(visual_embs) if visual_embs else None,
            is_multimodal=visual_is_multimodal,
        )

        has_audio = bool(audio_packed_embs or audio_plain_embs)
        if not has_audio:
            # Stale-state hygiene even when no audio this turn.
            self._pending_audio_deepstack = None
            return inputs_embeds

        if audio_token_id is None:
            raise ValueError("`audio_token_id` is not configured.")

        # Extract audio main + (optional) deepstack levels.
        if audio_packed_embs:
            audio_cat = torch.cat(audio_packed_embs, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype)
            # Drop marker col, split body into main + K levels.
            audio_body = audio_cat[:, :visual_full_dim]
            splits = [visual_dim] * (1 + K)
            pieces = torch.split(audio_body, splits, dim=-1)
            audio_main = pieces[0]                # (n_aud, visual_dim)
            audio_levels: Optional[list[torch.Tensor]] = list(pieces[1:])
            if len(audio_levels) != K:
                raise RuntimeError(
                    f"Audio packed split produced {len(audio_levels)} "
                    f"levels, expected {K}")
        else:
            audio_main = torch.cat(audio_plain_embs, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype)
            audio_levels = None

        # Audio scatter: main embed -> inputs_embeds at audio_pad positions.
        audio_pos_mask = (input_ids == audio_token_id)
        n_pad = int(audio_pos_mask.sum().item())
        n_emb = int(audio_main.shape[0])
        if n_pad != n_emb:
            raise ValueError(
                f"Audio features and audio_pad tokens mismatch: "
                f"audio_pad count={n_pad}, audio embed count={n_emb}")
        audio_mask = audio_pos_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_main)

        # Audio DeepStack: write per-LLM-layer audio embeds into the shared
        # ``self.deepstack_input_embeds[ll]`` buffer at audio_pad positions
        # so LLM layer ll's forward picks them up. Config:
        # ``audio_deepstack_llm_layer_indexes = [0, 1, 2]`` matches K=3
        # visual deepstack slots, so we reuse the same buffer.
        #
        # Buffer-state hygiene (the subtle bit):
        # - Mixed visual+audio: parent already called
        #   _set_deepstack_input_embeds with the visual deepstack tensor.
        #   We must read the buffer to preserve visual contributions, then
        #   scatter audio on top.
        # - Pure audio: parent's embed_input_ids returns early when
        #   multimodal_embeddings is empty (we filtered all audio out via
        #   visual_embs=None) and DOES NOT touch the deepstack buffer. The
        #   buffer may contain stale data from the previous request. We
        #   must zero-init in this case rather than read stale state.
        if (audio_levels is not None
                and self.use_deepstack
                and getattr(self, "deepstack_input_embeds", None)):
            num_tokens = inputs_embeds.size(0)
            hidden_size = self.deepstack_input_embeds[0].size(1)

            if visual_embs:
                # Mixed: preserve visual deepstack already in buffer.
                combined = torch.stack(
                    [self.deepstack_input_embeds[k][:num_tokens].clone()
                     for k in range(K)],
                    dim=0,
                )  # (K, num_tokens, hidden)
            else:
                # Pure audio: do NOT trust stale buffer; start from zeros.
                combined = inputs_embeds.new_zeros(
                    K, num_tokens, hidden_size)

            level_mask = audio_pos_mask.unsqueeze(-1).expand(
                num_tokens, hidden_size)
            for level_idx, audio_level in enumerate(audio_levels):
                if audio_level.shape != (n_emb, hidden_size):
                    # Should never happen — main/levels come from the same
                    # vLLM-cropped tensor — but guard explicitly anyway
                    # (cheap, and shields against -O removing asserts).
                    raise RuntimeError(
                        f"Audio packed deepstack level {level_idx} shape "
                        f"{tuple(audio_level.shape)} != expected "
                        f"{(n_emb, hidden_size)}")
                combined[level_idx].masked_scatter_(level_mask, audio_level)

            # Write back via the parent's API; .copy_() handles the
            # full-buffer overwrite cleanly.
            self._set_deepstack_input_embeds(combined)

        elif (audio_levels is None
                and not visual_embs
                and self.use_deepstack
                and getattr(self, "deepstack_input_embeds", None)):
            # Plain audio (no deepstack) + pure-audio request +
            # deepstack-capable buffer: nothing has written to the buffer
            # this turn (parent's embed_input_ids returns early without
            # touching it when multimodal_embeddings is empty), but stale
            # data from previous requests may linger. Zero it explicitly so
            # the LLM forward's per-layer ``+ deepstack_input_embeds[k]``
            # doesn't pick up residue. Mixed visual+audio is intentionally
            # excluded — the parent already wrote visual deepstack here.
            num_tokens = inputs_embeds.size(0)
            hidden_size = self.deepstack_input_embeds[0].size(1)
            zero = inputs_embeds.new_zeros(K, num_tokens, hidden_size)
            self._set_deepstack_input_embeds(zero)

        # Hygiene: legacy stash must remain cleared on the packed path.
        self._pending_audio_deepstack = None

        return inputs_embeds

    # ---- MRoPE: handle audio modality ----------------------------------

    @staticmethod
    def _iter_mm_grid_hw(input_tokens, mm_features, video_token_id,
                         vision_start_token_id, vision_end_token_id,
                         spatial_merge_size):
        """
        Same semantics as parent, but adds an audio branch.

        HF reference (``UniMoE2d5Model.get_rope_index``) has no audio path at
        all: audio_pad tokens fall through the image/video loop's text-segment
        bookkeeping and end up with text-like MRoPE positions (all three axes
        advance one step per token). We reproduce that by yielding
        ``(offset, llm_grid_h=1, llm_grid_w=1, actual=n)`` so the parent's
        ``actual > expected`` lumped-placeholder branch kicks in: it loops n
        times, each iteration appending a ``np.indices((1,1,1))`` (== 0)
        column and bumping ``st_idx`` by 1 -- net effect: three-axis positions
        (k, k+1, ..., k+n-1) for the audio span, matching HF's text-like
        treatment exactly.
        """
        for f in sorted(mm_features, key=lambda x: x.mm_position.offset):
            if f.modality == "audio":
                n = f.mm_position.length
                yield f.mm_position.offset, 1, 1, n
            else:
                # Delegate one feature at a time so parent's offset/video
                # bookkeeping (which uses vision_start_token_id indexing)
                # stays correct.
                yield from Qwen3VLForConditionalGeneration._iter_mm_grid_hw(
                    input_tokens, [f],
                    video_token_id=video_token_id,
                    vision_start_token_id=vision_start_token_id,
                    vision_end_token_id=vision_end_token_id,
                    spatial_merge_size=spatial_merge_size,
                )

    @staticmethod
    def _get_mrope_input_positions(input_tokens, mm_features, config):
        """
        Copy of parent's static helper, with the iter call pointing at
        our class so the audio branch above is picked up. The parent's
        version hardcodes the class name, so subclass override of the
        iter alone is not enough.
        """
        llm_pos_ids_list: list = []
        st = 0
        for (offset, llm_grid_h, llm_grid_w, actual_num_tokens) in \
                UniMoE2d5ForConditionalGeneration._iter_mm_grid_hw(
                    input_tokens, mm_features,
                    video_token_id=config.video_token_id,
                    vision_start_token_id=config.vision_start_token_id,
                    vision_end_token_id=config.vision_end_token_id,
                    spatial_merge_size=config.vision_config.spatial_merge_size,
                ):
            if actual_num_tokens == 0:
                continue

            text_len = offset - st
            st_idx = (llm_pos_ids_list[-1].max() + 1
                      if len(llm_pos_ids_list) > 0 else 0)
            llm_pos_ids_list.append(
                np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx)

            expected = llm_grid_h * llm_grid_w
            if actual_num_tokens > expected:
                num_logical_frames = actual_num_tokens // expected
                remainder = actual_num_tokens % expected
                for _ in range(num_logical_frames):
                    grid = np.indices((1, llm_grid_h, llm_grid_w)).reshape(3, -1)
                    llm_pos_ids_list.append(grid + text_len + st_idx)
                    st_idx = llm_pos_ids_list[-1].max() + 1
                    text_len = 0
                if remainder > 0:
                    full = np.indices((1, llm_grid_h, llm_grid_w)).reshape(3, -1)
                    llm_pos_ids_list.append(full[:, :remainder] + text_len + st_idx)
            else:
                grid = np.indices((1, llm_grid_h, llm_grid_w)).reshape(3, -1)
                llm_pos_ids_list.append(grid + text_len + st_idx)

            st = offset + actual_num_tokens

        if st < len(input_tokens):
            st_idx = (llm_pos_ids_list[-1].max() + 1
                      if len(llm_pos_ids_list) > 0 else 0)
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx)

        llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
        delta = (llm_positions.max() + 1 - len(input_tokens)).item()
        return torch.from_numpy(llm_positions), delta

    # ---- weight loading ------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Strip optional sub-trees that the engine didn't build.
        if self.audio is None:
            weights = ((n, w) for n, w in weights
                       if not n.startswith("model.audio."))
        if self.visual is None:
            weights = ((n, w) for n, w in weights
                       if not n.startswith("model.visual."))
        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        return loaded
