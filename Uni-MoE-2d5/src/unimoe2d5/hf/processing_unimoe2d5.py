# coding=utf-8
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack, VideosKwargs
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import logging
from transformers.video_utils import VideoInput

logger = logging.get_logger(__name__)


class UniMoE2d5VideosKwargs(VideosKwargs, total=False):
    use_audio_in_video: bool
    position_id_per_seconds: int | float


class UniMoE2d5ProcessorKwargs(ProcessingKwargs, total=False):
    videos_kwargs: UniMoE2d5VideosKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_token_type_ids": False,
            "return_mm_token_type_ids": False,
        },
        "videos_kwargs": {
            "return_metadata": True,
            "use_audio_in_video": False,
            "position_id_per_seconds": 25.0,
        },
        "audio_kwargs": {
            "sampling_rate": 16000,
        },
    }


class UniMoE2d5Processor(ProcessorMixin):
    r"""
    Constructs a Qwen3VL processor which wraps a Qwen3VL image processor and a Qwen2 tokenizer into a single processor.
    [`UniMoE2d5Processor`] offers all the functionalities of [`Qwen2VLImageProcessor`] and [`Qwen2TokenizerFast`]. See the
    [`~UniMoE2d5Processor.__call__`] and [`~UniMoE2d5Processor.decode`] for more information.
    Args:
        image_processor ([`Qwen2VLImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The tokenizer is a required input.
        video_processor ([`Qwen3VLVideoProcessor`], *optional*):
            The video processor is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer", "video_processor", "feature_extractor"]
    image_processor_class = "AutoImageProcessor"
    video_processor_class = "AutoVideoProcessor"
    feature_extractor_class = "AutoFeatureExtractor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        feature_extractor=None,
        chat_template=None,
        **kwargs,
    ):
        self.image_token = "<|image_pad|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.video_token = "<|video_pad|>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token
        self.audio_token = "<|audio_pad|>" if not hasattr(tokenizer, "audio_token") else tokenizer.audio_token
        self.audio_start_token = (
            "<|audio_start|>" if not hasattr(tokenizer, "audio_start_token") else tokenizer.audio_start_token
        )
        self.audio_end_token = (
            "<|audio_end|>" if not hasattr(tokenizer, "audio_end_token") else tokenizer.audio_end_token
        )
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if getattr(tokenizer, "video_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )
        self.audio_token_id = (
            tokenizer.audio_token_id
            if getattr(tokenizer, "audio_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.audio_token)
        )
        self.audio_start_token_id = (
            tokenizer.audio_start_token_id
            if getattr(tokenizer, "audio_start_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.audio_start_token)
        )
        self.audio_end_token_id = (
            tokenizer.audio_end_token_id
            if getattr(tokenizer, "audio_end_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.audio_end_token)
        )
        if feature_extractor is None:
            feature_extractor = kwargs.pop("audio_feature_extractor", None)
        else:
            kwargs.pop("audio_feature_extractor", None)
        self.audio_processor_name_or_path = kwargs.pop("audio_processor_name_or_path", None)
        super().__init__(
            image_processor,
            tokenizer,
            video_processor,
            feature_extractor,
            chat_template=chat_template,
        )
        self.vision_start_token = (
            "<|vision_start|>" if not hasattr(tokenizer, "vision_start_token") else tokenizer.vision_start_token
        )
        self.vision_end_token = (
            "<|vision_end|>" if not hasattr(tokenizer, "vision_end_token") else tokenizer.vision_end_token
        )
        self.vision_start_token_id = (
            tokenizer.vision_start_token_id
            if getattr(tokenizer, "vision_start_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.vision_start_token)
        )
        self.vision_end_token_id = (
            tokenizer.vision_end_token_id
            if getattr(tokenizer, "vision_end_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.vision_end_token)
        )

    @property
    def audio_feature_extractor(self):
        return self.feature_extractor

    @audio_feature_extractor.setter
    def audio_feature_extractor(self, value):
        self.feature_extractor = value

    def set_audio_feature_extractor(self, audio_processor_or_path: Any):
        if audio_processor_or_path is None:
            self.feature_extractor = None
            self.audio_processor_name_or_path = None
            return

        if isinstance(audio_processor_or_path, str):
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(audio_processor_or_path, trust_remote_code=True)
            self.feature_extractor = processor.feature_extractor
            self.audio_processor_name_or_path = audio_processor_or_path
            return

        if hasattr(audio_processor_or_path, "feature_extractor"):
            self.feature_extractor = audio_processor_or_path.feature_extractor
            return

        self.feature_extractor = audio_processor_or_path

    @staticmethod
    def _compute_audio_token_lengths(attention_mask: torch.Tensor) -> list[int]:
        valid_mel_frames = attention_mask.sum(dim=1).to(torch.int64)
        l_conv = torch.div(valid_mel_frames - 1, 2, rounding_mode="floor") + 1
        l_pool = torch.div(l_conv - 2, 2, rounding_mode="floor") + 1
        l_pool = torch.where(valid_mel_frames > 0, torch.clamp(l_pool, min=1), torch.zeros_like(l_pool))
        return l_pool.tolist()

    @staticmethod
    def _to_audio_chunk(chunk: Any) -> np.ndarray:
        if isinstance(chunk, np.ndarray):
            waveform = chunk.astype(np.float32, copy=False)
        elif torch.is_tensor(chunk):
            waveform = chunk.detach().cpu().float().numpy()
        else:
            raise TypeError(f"Unsupported audio chunk type: {type(chunk)}")

        if waveform.ndim == 2:
            waveform = waveform.mean(axis=0)
        elif waveform.ndim != 1:
            waveform = waveform.reshape(-1)
        return np.ascontiguousarray(waveform, dtype=np.float32)

    def _process_audio(
        self,
        audios: Sequence[Any],
        sampling_rate: int = 16000,
        **kwargs,
    ) -> dict[str, Any]:
        if self.feature_extractor is None:
            raise ValueError(
                "Audio feature extractor is not set on UniMoE2d5Processor. "
                "Call `processor.set_audio_feature_extractor(path_or_processor)` first."
            )

        all_features = []
        all_masks = []
        audio_token_lengths = []
        for audio in audios:
            if not isinstance(audio, (list, tuple)):
                raise TypeError(
                    "Each audio input should be a list/tuple of preprocessed chunks. "
                    "Use `process_omni_info` to complete audio preprocessing before processor."
                )
            chunks = [self._to_audio_chunk(chunk) for chunk in audio]

            chunks = [chunk for chunk in chunks if chunk.shape[0] > 0]
            if len(chunks) == 0:
                chunks = [np.zeros((max(1, int(sampling_rate)),), dtype=np.float32)]

            audio_inputs = self.feature_extractor(
                chunks,
                sampling_rate=sampling_rate,
                return_tensors="pt",
                return_attention_mask=True,
            )
            attention_mask = audio_inputs.attention_mask.long()
            all_features.append(audio_inputs.input_features)
            all_masks.append(attention_mask)
            audio_token_lengths.append(sum(self._compute_audio_token_lengths(attention_mask)))

        audio_features = torch.cat(all_features, dim=0) if all_features else torch.empty(0)
        audio_features_mask = torch.cat(all_masks, dim=0) if all_masks else torch.empty(0, dtype=torch.long)
        return {
            "audio_features": audio_features,
            "audio_features_mask": audio_features_mask,
            "audio_token_lengths": audio_token_lengths,
        }

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        videos: VideoInput = None,
        audios: Optional[Union[Sequence[Any], Any]] = None,
        **kwargs: Unpack[UniMoE2d5ProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the vision inputs, this method forwards the `vision_infos` and `kwrags` arguments to
        Qwen2VLImageProcessor's [`~Qwen2VLImageProcessor.__call__`] if `vision_infos` is not `None`.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `list[str]`, `list[list[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            videos (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The image or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch
                tensor, or a nested list of 3D frames. Both channels-first and channels-last formats are supported.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **pixel_values_videos** -- Pixel values of videos to be fed to a model. Returned when `videos` is not `None`.
            - **image_grid_thw** -- List of image 3D grid in LLM. Returned when `images` is not `None`.
            - **video_grid_thw** -- List of video 3D grid in LLM. Returned when `videos` is not `None`.
        """
        explicit_audio_kwargs = kwargs.pop("audio_kwargs", None)
        output_kwargs = self._merge_kwargs(
            UniMoE2d5ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        videos_kwargs = output_kwargs["videos_kwargs"].copy()
        use_audio_in_video = bool(videos_kwargs.pop("use_audio_in_video", False))
        position_id_per_seconds = float(videos_kwargs.pop("position_id_per_seconds", 25.0))
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None

        if videos is not None:
            videos_inputs = self.video_processor(videos=videos, **videos_kwargs)
            video_grid_thw = videos_inputs["video_grid_thw"]
            # If user has not requested video metadata, pop it
            if not kwargs.get("return_metadata"):
                video_metadata = videos_inputs.pop("video_metadata")
            else:
                video_metadata = videos_inputs["video_metadata"]
        else:
            videos_inputs = {}
            video_grid_thw = None

        audio_proc_kwargs = output_kwargs.get("audio_kwargs", None)
        if audio_proc_kwargs is None:
            audio_proc_kwargs = explicit_audio_kwargs or UniMoE2d5ProcessorKwargs._defaults.get("audio_kwargs", {})

        if audios is not None:
            if not isinstance(audios, (list, tuple)):
                raise TypeError(
                    "`audios` should be a list of preprocessed audio chunks produced by `process_omni_info`."
                )
            audio_inputs = self._process_audio(audios=audios, **audio_proc_kwargs)
            audio_token_lengths = audio_inputs.pop("audio_token_lengths")
        else:
            audio_inputs = {}
            audio_token_lengths = None

        if not isinstance(text, list):
            text = [text]

        text = text.copy()  # below lines change text in-place
        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    if index >= len(image_grid_thw):
                        raise ValueError(
                            "Image placeholder count exceeds available image inputs. "
                            f"text_index={i}, consumed_image_placeholders={index}, "
                            f"available_image_inputs={len(image_grid_thw)}, "
                            f"remaining_text_image_placeholders={text[i].count(self.image_token)}"
                        )
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(self.image_token, "<|image_placeholder|>" * num_image_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|image_placeholder|>", self.image_token)
            if index != len(image_grid_thw):
                raise ValueError(
                    "Provided image inputs are not fully consumed by image placeholders in text. "
                    f"consumed_image_inputs={index}, available_image_inputs={len(image_grid_thw)}"
                )

        if video_grid_thw is not None:
            merge_length = self.video_processor.merge_size**2
            index = 0
            audio_index_for_video = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    if video_metadata is None:
                        raise ValueError(
                            "Video metadata is missing while video placeholders are present in text. "
                            f"text_index={i}, consumed_video_placeholders={index}, "
                            f"available_video_inputs={len(video_grid_thw)}"
                        )
                    if index >= len(video_metadata):
                        raise ValueError(
                            "Video placeholder count exceeds available video metadata. "
                            f"text_index={i}, consumed_video_placeholders={index}, "
                            f"available_video_metadata={len(video_metadata)}, "
                            f"available_video_inputs={len(video_grid_thw)}, "
                            f"remaining_text_video_placeholders={text[i].count(self.video_token)}"
                        )
                    if index >= len(video_grid_thw):
                        raise ValueError(
                            "Video placeholder count exceeds available video inputs. "
                            f"text_index={i}, consumed_video_placeholders={index}, "
                            f"available_video_inputs={len(video_grid_thw)}, "
                            f"available_video_metadata={len(video_metadata)}, "
                            f"remaining_text_video_placeholders={text[i].count(self.video_token)}"
                        )
                    metadata = video_metadata[index]
                    if metadata.fps is None:
                        logger.warning_once(
                            "Qwen3VL requires frame timestamps to construct prompts, but the `fps` of the input video could not be inferred. "
                            "Probably `video_metadata` was missing from inputs and you passed pre-sampled frames. "
                            "Defaulting to `fps=24`. Please provide `video_metadata` for more accurate results."
                        )
                        metadata.fps = 24 if metadata.fps is None else metadata.fps

                    # if timestamps are not provided, calculate them
                    curr_timestamp = self._calculate_timestamps(
                        metadata.frames_indices,
                        metadata.fps,
                        self.video_processor.merge_size,
                    )
                    interleave_audio = (
                        use_audio_in_video
                        and audio_token_lengths is not None
                        and audio_index_for_video < len(audio_token_lengths)
                    )
                    if use_audio_in_video and not interleave_audio:
                        logger.warning_once(
                            "`use_audio_in_video=True`, but no matching audio inputs were found for a video placeholder. "
                            "Falling back to video-only encoding for this sample."
                        )
                    if interleave_audio:
                        num_audio_tokens = int(audio_token_lengths[audio_index_for_video])
                        audio_index_for_video += 1
                        audio_tokens_per_frame = self._split_audio_tokens_by_video_timestamps(
                            num_audio_tokens=num_audio_tokens,
                            timestamps=curr_timestamp,
                            position_id_per_seconds=position_id_per_seconds,
                        )
                    video_placeholder = ""
                    frame_seqlen = video_grid_thw[index][1:].prod() // merge_length
                    for frame_idx in range(video_grid_thw[index][0]):
                        curr_time = curr_timestamp[frame_idx]
                        video_placeholder += f"<{curr_time:.1f} seconds>"
                        video_placeholder += (
                            self.vision_start_token + "<|video_placeholder|>" * frame_seqlen + self.vision_end_token
                        )
                        if interleave_audio:
                            frame_audio_tokens = audio_tokens_per_frame[frame_idx]
                            video_placeholder += (
                                self.audio_start_token
                                + "<|audio_placeholder|>" * frame_audio_tokens
                                + self.audio_end_token
                            )
                    if f"{self.vision_start_token}{self.video_token}{self.vision_end_token}" in text[i]:
                        text[i] = text[i].replace(
                            f"{self.vision_start_token}{self.video_token}{self.vision_end_token}", video_placeholder, 1
                        )
                    else:
                        # vllm may input video token directly
                        text[i] = text[i].replace(self.video_token, video_placeholder, 1)
                    if interleave_audio and self.audio_token in text[i]:
                        # Drop one explicit audio placeholder if this video already consumed its paired audio input.
                        video_pos = text[i].find(video_placeholder)
                        search_start = (video_pos + len(video_placeholder)) if video_pos >= 0 else 0
                        audio_pos = text[i].find(self.audio_token, search_start)
                        if audio_pos < 0:
                            audio_pos = text[i].find(self.audio_token)
                        if audio_pos >= 0:
                            text[i] = text[i][:audio_pos] + text[i][audio_pos + len(self.audio_token) :]
                    index += 1

                text[i] = text[i].replace("<|video_placeholder|>", self.video_token)
            if video_metadata is not None and index != len(video_metadata):
                raise ValueError(
                    "Provided video metadata is not fully consumed by video placeholders in text. "
                    f"consumed_video_metadata={index}, available_video_metadata={len(video_metadata)}"
                )
            if index != len(video_grid_thw):
                raise ValueError(
                    "Provided video inputs are not fully consumed by video placeholders in text. "
                    f"consumed_video_inputs={index}, available_video_inputs={len(video_grid_thw)}"
                )

        if audio_token_lengths is not None:
            index = audio_index_for_video if video_grid_thw is not None else 0
            for i in range(len(text)):
                while self.audio_token in text[i]:
                    if index >= len(audio_token_lengths):
                        raise ValueError("Audio placeholder count exceeds the provided audio inputs.")
                    num_audio_tokens = int(audio_token_lengths[index])
                    audio_placeholder = (
                        self.audio_start_token + "<|audio_placeholder|>" * num_audio_tokens + self.audio_end_token
                    )
                    text[i] = text[i].replace(self.audio_token, audio_placeholder, 1)
                    index += 1
                text[i] = text[i].replace("<|audio_placeholder|>", self.audio_token)
            if index != len(audio_token_lengths):
                raise ValueError("Provided audio inputs are not fully consumed by audio placeholders in text.")

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            mm_token_type_ids[array_ids == self.video_token_id] = 2
            mm_token_type_ids[array_ids == self.audio_token_id] = 3
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs, **audio_inputs}, tensor_type=return_tensors)

    def _get_num_multimodal_tokens(self, image_sizes=None, video_sizes=None, audio_lengths=None, **kwargs):
        """
        Computes the number of placeholder tokens needed for multimodal inputs with the given sizes.
        Args:
            image_sizes (`list[list[int]]`, *optional*):
                The input sizes formatted as (height, width) per each image.
            video_sizes (`list[list[int]]`, *optional*):
                The input sizes formatted as (num_frames, height, width) per each video.
        Returns:
            `MultiModalData`: A `MultiModalData` object holding number of tokens per each of the provided
            input modalities, along with other useful data.
        """

        vision_data = {}
        if image_sizes is not None:
            images_kwargs = UniMoE2d5ProcessorKwargs._defaults.get("images_kwargs", {})
            images_kwargs.update(kwargs)
            merge_size = images_kwargs.get("merge_size", None) or self.image_processor.merge_size

            num_image_patches = [
                self.image_processor.get_number_of_image_patches(*image_size, images_kwargs)
                for image_size in image_sizes
            ]
            num_image_tokens = [(num_patches // merge_size**2) for num_patches in num_image_patches]
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        if video_sizes is not None:
            videos_kwargs = UniMoE2d5ProcessorKwargs._defaults.get("videos_kwargs", {})
            videos_kwargs.update(kwargs)
            num_video_patches = [
                self.video_processor.get_number_of_video_patches(*video_size, videos_kwargs)
                for video_size in video_sizes
            ]
            num_video_tokens = [(num_patches // merge_size**2) for num_patches in num_video_patches]
            vision_data["num_video_tokens"] = num_video_tokens
        if audio_lengths is not None:
            vision_data["num_audio_tokens"] = list(audio_lengths)

        return MultiModalData(**vision_data)

    def post_process_image_text_to_text(
        self, generated_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False, **kwargs
    ):
        """
        Post-process the output of the model to decode the text.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                or `(sequence_length,)`.
            skip_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether or not to remove special tokens in the output. Argument passed to the tokenizer's `batch_decode` method.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
                Whether or not to clean up the tokenization spaces. Argument passed to the tokenizer's `batch_decode` method.
            **kwargs:
                Additional arguments to be passed to the tokenizer's `batch_decode method`.

        Returns:
            `list[str]`: The decoded text.
        """
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    def _calculate_timestamps(self, indices: Union[list[int], np.ndarray], video_fps: float, merge_size: int = 2):
        if not isinstance(indices, list):
            indices = indices.tolist()
        if len(indices) % merge_size != 0:
            indices.extend(indices[-1] for _ in range(merge_size - len(indices) % merge_size))
        timestamps = [idx / video_fps for idx in indices]

        # Use the first frame timestamp of each merged temporal patch for video-audio alignment.
        # timestamps = [timestamps[i] for i in range(0, len(timestamps), merge_size)]
        # Round to whole seconds (half-up). This keeps per-second audio alignment stable:
        # with `position_id_per_seconds=25`, each full second maps to ~25 audio tokens.
        # timestamps = [float(np.floor(ts + 0.5)) for ts in timestamps]

        # timestamps = [
            # round((timestamps[i] + timestamps[i + merge_size - 1]) / 2) for i in range(0, len(timestamps), merge_size)
        # ]
        timestamps = [
            round(timestamps[i]) for i in range(0, len(timestamps), merge_size)
        ]
        # timestamps = [
        #     (timestamps[i] + timestamps[i + merge_size - 1]) / 2 for i in range(0, len(timestamps), merge_size)
        # ]
        return timestamps

    @staticmethod
    def _split_audio_tokens_by_video_timestamps(
        num_audio_tokens: int,
        timestamps: list[float],
        position_id_per_seconds: float,
    ) -> list[int]:
        if len(timestamps) == 0:
            return []
        if num_audio_tokens <= 0:
            return [0] * len(timestamps)
        if position_id_per_seconds <= 0:
            raise ValueError(
                f"`position_id_per_seconds` should be positive when using audio-video interleaving, got {position_id_per_seconds}."
            )

        base_timestamp = float(timestamps[0])
        token_starts = [
            max(0, int(np.floor((float(ts) - base_timestamp) * position_id_per_seconds + 0.5)))
            for ts in timestamps
        ]
        token_starts = np.minimum(np.maximum.accumulate(np.array(token_starts, dtype=np.int64)), num_audio_tokens).tolist()

        tokens_per_frame = []
        for idx, token_start in enumerate(token_starts):
            token_end = token_starts[idx + 1] if idx + 1 < len(token_starts) else num_audio_tokens
            token_end = min(max(token_end, token_start), num_audio_tokens)
            tokens_per_frame.append(token_end - token_start)

        remaining_tokens = num_audio_tokens - sum(tokens_per_frame)
        if remaining_tokens > 0:
            tokens_per_frame[-1] += remaining_tokens

        return tokens_per_frame


__all__ = ["UniMoE2d5Processor"]


UniMoE2d5Processor.register_for_auto_class("AutoProcessor")
