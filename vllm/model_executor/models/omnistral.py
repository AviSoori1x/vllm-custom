# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property
from math import ceil
from typing import Optional, Union, cast

import numpy as np
import regex as re
import torch
import torch.nn as nn
from mistral_common.protocol.instruct.messages import (AudioChunk, ImageChunk, 
                                                       RawAudio, TextChunk, 
                                                       UserMessage)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.transcription.request import TranscriptionRequest
from mistral_common.tokens.tokenizers.audio import Audio, AudioEncoder
from mistral_common.tokens.tokenizers.multimodal import ImageEncoder
from PIL import Image
from transformers import PixtralVisionConfig, TensorType
from transformers.image_utils import ImageInput
from transformers.tokenization_utils_base import TextInput

from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.inputs.data import PromptType
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models import SupportsPP
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs, NestedTensors)
from vllm.multimodal.parse import (AudioProcessorItems, ImageProcessorItems,
                                   ImageSize, MultiModalDataItems,
                                   MultiModalDataParser)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, MultiModalHashes,
                                        PromptReplacement, PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.tokenizer import (MistralTokenizer,
                                               cached_tokenizer_from_config)

from .interfaces import (MultiModalEmbeddings, SupportsMultiModal,
                         SupportsTranscription)
from .utils import (flatten_bn, init_vllm_registered_model, maybe_prefix,
                    merge_multimodal_embeddings)

# Import components from Voxtral and Pixtral
from .voxtral import (VoxtralEncoderModel, AudioLanguageAdapter, 
                      VoxtralProcessorAdapter)
from .pixtral import (VisionTransformer, VisionLanguageAdapter, PatchMerger,
                      VisionEncoderArgs, PATCH_MERGE)

logger = init_logger(__name__)


class OmnistralProcessorAdapter:
    """
    Unified processor adapter that handles both audio and image inputs,
    combining capabilities from VoxtralProcessorAdapter and PixtralProcessorAdapter.
    """

    def __init__(self, tokenizer: MistralTokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer

    @cached_property
    def _audio_processor(self) -> AudioEncoder:
        audio_encoder = self.tokenizer.instruct.audio_encoder
        
        # Accept compatible encoder implementations (e.g. Tekken's) that
        # expose the expected interface but don't inherit from AudioEncoder.
        if not isinstance(audio_encoder, AudioEncoder):
            required_attrs = [
                "special_ids", "audio_config",
                "next_multiple_of_chunk_frames", "pad"
            ]
            if not all(hasattr(audio_encoder, attr) for attr in required_attrs):
                raise TypeError(
                    "The tokenizer's audio_encoder is not compatible with "
                    "vLLM. Expected attributes "
                    f"{required_attrs}, got {type(audio_encoder)}")
        return audio_encoder

    @cached_property
    def _image_processor(self) -> ImageEncoder:
        image_encoder = self.tokenizer.instruct.mm_encoder
        assert isinstance(image_encoder, ImageEncoder)
        return image_encoder

    # Audio properties
    @cached_property
    def audio_token_id(self) -> int:
        return self._audio_processor.special_ids.audio

    @cached_property
    def begin_audio_token_id(self) -> int:
        return self._audio_processor.special_ids.begin_audio

    @cached_property
    def sampling_rate(self) -> int:
        return self._audio_processor.audio_config.sampling_rate

    @cached_property
    def frame_rate(self) -> float:
        return self._audio_processor.audio_config.frame_rate

    # Image properties
    @cached_property
    def image_break_id(self) -> int:
        return self._image_processor.special_ids.img_break

    @cached_property
    def image_token_id(self) -> int:
        return self._image_processor.special_ids.img

    @cached_property
    def image_end_id(self) -> int:
        return self._image_processor.special_ids.img_end

    @cached_property
    def image_size(self) -> int:
        return self._image_processor.mm_config.max_image_size

    @cached_property
    def patch_size(self) -> int:
        return self._image_processor.mm_config.image_patch_size

    def get_num_audio_tokens(self, audio_length: int) -> int:
        pad_audio_length = self._audio_processor.next_multiple_of_chunk_frames(
            audio_length, self.sampling_rate)
        return ceil(pad_audio_length / (self.sampling_rate // self.frame_rate))

    # First, fix the OmnistralProcessorAdapter.__call__ method:
    def __call__(
        self,
        text: Optional[Union[TextInput, list[TextInput]]] = None,
        audios: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
        images: Optional[Union[ImageInput, list[ImageInput]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> Mapping[str, NestedTensors]:
        if text is None:
            text = []
        if not isinstance(text, list):
            text = [text]
        if audios is None:
            audios = []
        if not isinstance(audios, list):
            audios = [audios]
        if images is None:
            images = []
        if not isinstance(images, list):
            images = [images]

        # Handle text-only case
        if not audios and not images:
            input_ids = self.tokenizer(text).input_ids
            return {"input_ids": torch.tensor(input_ids)}

        # Validate that we're using pre-tokenized inputs for multimodal
        if any(len(t) > 0 for t in text):
            raise ValueError(
                "You've passed text inputs instead of token inputs. "
                "Make sure to process your input via `mistral_common`'s "
                "tokenizer or pass a chat completion request.")

        # Process audio inputs
        audios_processed = []
        audio_tokens_list = []
        for audio in audios:
            assert isinstance(audio, np.ndarray)
            assert audio.ndim == 1

            # Pad if necessary
            audio = self._audio_processor.pad(audio, self.sampling_rate)
            audios_processed.append(torch.tensor(audio))
            
            # Generate audio tokens
            n_tokens = self.get_num_audio_tokens(len(audio))
            audio_tokens = [self.begin_audio_token_id] + [self.audio_token_id] * n_tokens
            audio_tokens_list.extend(audio_tokens)

        # Process image inputs - THIS IS THE KEY FIX
        images_processed = []
        image_tokens_list = []
        for image in images:
            # Process the image to get its encoding
            image_encoding = self._image_processor(ImageChunk(image=image))
            image_processed = torch.tensor(image_encoding.image)
            images_processed.append(image_processed)
            
            # Use the actual tokens from the image encoding
            # This ensures the token count matches the number of patches
            image_tokens_list.extend(image_encoding.tokens)

        # Combine all tokens
        all_tokens = audio_tokens_list + image_tokens_list
        
        if all_tokens:
            input_ids = torch.tensor(all_tokens)[None].expand(len(text) if text else 1, -1)
        else:
            input_ids = torch.tensor([[]])

        result = {"input_ids": input_ids}
        
        if audios_processed:
            result["audio_arrays"] = audios_processed
        if images_processed:
            result["images"] = images_processed
            
        return result

    # Second, fix get_multimodal_embeddings to return embeddings correctly:
    def get_multimodal_embeddings(self, **kwargs) -> MultiModalEmbeddings:
        """Process both audio and image inputs and return combined embeddings."""
        # Process each modality separately
        audio_embeddings = self._process_audio_inputs(**kwargs)
        image_embeddings = self._process_image_inputs(**kwargs)
        
        # Return a flat list of all embeddings
        # Each item in the list corresponds to one multimodal input (audio or image)
        all_embeddings = []
        if audio_embeddings:
            all_embeddings.extend(audio_embeddings)
        if image_embeddings:
            all_embeddings.extend(image_embeddings)
        
        return all_embeddings

    # Third, simplify get_input_embeddings:
    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        """Merge multimodal embeddings with text embeddings."""
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        
        if multimodal_embeddings is None or not multimodal_embeddings:
            return inputs_embeds
        
        # Get token IDs
        audio_tok_id = self.tokenizer.instruct.audio_encoder.audio_token
        image_tok_id = self.tokenizer.instruct.mm_encoder.special_ids.img
        
        # For single modality inputs, merge directly
        has_audio = (input_ids == audio_tok_id).any()
        has_image = (input_ids == image_tok_id).any()
        
        if has_audio and not has_image:
            # All embeddings are audio
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings, audio_tok_id
            )
        elif has_image and not has_audio:
            # All embeddings are image
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings, image_tok_id
            )
        elif has_audio and has_image:
            # Mixed modality - more complex
            # Count expected embeddings for each type
            num_audio_tokens = (input_ids == audio_tok_id).sum().item()
            
            # Split embeddings based on what we know about sizes
            # Audio embeddings typically have size matching token count
            audio_embeds = []
            image_embeds = []
            
            for emb in multimodal_embeddings:
                # Simple heuristic: if embedding size matches audio token count,
                # it's probably audio. Otherwise, it's image.
                if len(audio_embeds) == 0 and emb.shape[0] == num_audio_tokens:
                    audio_embeds.append(emb)
                else:
                    image_embeds.append(emb)
            
            # Merge each type
            if audio_embeds:
                inputs_embeds = merge_multimodal_embeddings(
                    input_ids, inputs_embeds, audio_embeds, audio_tok_id
                )
            if image_embeds:
                inputs_embeds = merge_multimodal_embeddings(
                    input_ids, inputs_embeds, image_embeds, image_tok_id
                )
        
        return inputs_embeds

class OmnistralProcessingInfo(BaseProcessingInfo):

    def get_tokenizer(self) -> MistralTokenizer:
        tokenizer = cached_tokenizer_from_config(self.ctx.model_config)
        if not isinstance(tokenizer, MistralTokenizer):
            raise ValueError("This model requires `--tokenizer-mode mistral`")
        return tokenizer

    def get_hf_processor(self) -> OmnistralProcessorAdapter:
        return OmnistralProcessorAdapter(self.get_tokenizer())

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {
            "audio": 5,    # Performance tends to degrade after 5
            "image": None  # No specific limit for images
        }

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        return {
            "audio": self.get_max_audio_tokens(),
            "image": self.get_max_image_tokens()
        }

    def get_max_audio_tokens(self) -> int:
        return self.ctx.model_config.max_model_len // 4  # Reserve 1/4 for audio

    def get_max_image_tokens(self) -> int:
        return self.ctx.model_config.max_model_len // 2  # Reserve 1/2 for images

    def get_max_audio_array_len(self) -> int:
        processor = self.get_hf_processor()
        return self.get_max_audio_tokens() * int(
            processor.sampling_rate // processor.frame_rate)

    def get_vision_config(self) -> PixtralVisionConfig:
        processor = self.get_hf_processor()
        return PixtralVisionConfig(
            image_size=processor.image_size,
            patch_size=processor.patch_size,
        )

    def get_num_image_tokens(self, *, image_width: int, image_height: int) -> int:
        processor = self.get_hf_processor()
        ncols, nrows = processor._image_processor._image_to_num_tokens(
            Image.new("RGB", (image_width, image_height)))
        return ncols * nrows

    def get_image_size_with_most_features(self) -> ImageSize:
        processor = self.get_hf_processor()
        max_image_size = processor.image_size
        return ImageSize(width=max_image_size, height=max_image_size)


class OmnistralDummyInputsBuilder(BaseDummyInputsBuilder[OmnistralProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        mm_data = {}
        
        # Add dummy audio data
        num_audios = mm_counts.get("audio", 0)
        if num_audios > 0:
            target_audio_length = self.info.get_max_audio_array_len()
            mm_data["audio"] = self._get_dummy_audios(
                length=target_audio_length, num_audios=num_audios)

        # Add dummy image data
        num_images = mm_counts.get("image", 0)
        if num_images > 0:
            target_width, target_height = \
                self.info.get_image_size_with_most_features()
            mm_data["image"] = self._get_dummy_images(
                width=target_width, height=target_height, num_images=num_images)

        return mm_data

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        tokenizer = self.info.get_tokenizer()

        dummy_text = self.get_dummy_text(mm_counts)
        dummy_mm_data = self.get_dummy_mm_data(seq_len, mm_counts)
        dummy_audios = dummy_mm_data.get("audio", [])
        dummy_images = dummy_mm_data.get("image", [])

        # Create chunks for both modalities
        content_chunks = [TextChunk(text=dummy_text)]
        
        # Add audio chunks
        for audio in dummy_audios:
            audio_item = Audio(
                audio_array=audio,
                sampling_rate=self.info.get_hf_processor().sampling_rate,
                format="wav",
            )
            chunk = AudioChunk(input_audio=RawAudio.from_audio(audio_item))
            content_chunks.append(chunk)

        # Add image chunks
        for image in dummy_images:
            chunk = ImageChunk(image=image)
            content_chunks.append(chunk)

        request = ChatCompletionRequest(messages=[
            UserMessage(content=content_chunks),
        ])
        res = tokenizer.mistral.encode_chat_completion(request)
        dummy_tokens = res.tokens

        # Update dummy data with processed audio/images
        if hasattr(res, 'audios') and res.audios:
            dummy_mm_data["audio"] = [a.audio_array for a in res.audios]

        return ProcessorInputs(prompt=dummy_tokens, mm_data=dummy_mm_data)


class OmnistralMultiModalProcessor(BaseMultiModalProcessor[OmnistralProcessingInfo]):

    def _get_mm_fields_config(
        self,
        hf_inputs: Mapping[str, NestedTensors],
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {
            "audio_arrays": MultiModalFieldConfig.batched("audio"),
            "images": MultiModalFieldConfig.batched("image")
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        # Audio token replacement
        audio_id = processor.audio_token_id

        def get_audio_replacement(item_idx: int):
            audios = mm_items.get_items("audio", AudioProcessorItems)
            audio_len = audios.get_audio_length(item_idx)
            nb_audio_tokens = processor.get_num_audio_tokens(audio_len)
            return [audio_id] * nb_audio_tokens

        # Image token replacement
        image_break_id = processor.image_break_id
        image_token_id = processor.image_token_id
        image_end_id = processor.image_end_id

        def get_image_replacement(item_idx: int):
            images = mm_items.get_items("image", ImageProcessorItems)
            image_size = images.get_image_size(item_idx)

            ncols, nrows = processor._image_processor._image_to_num_tokens(
                Image.new("RGB", (image_size.width, image_size.height)))

            tokens = ([image_token_id] * ncols + [image_break_id]) * nrows
            tokens[-1] = image_end_id

            from vllm.multimodal.processing import PromptUpdateDetails
            return PromptUpdateDetails.select_token_id(tokens, image_token_id)

        return [
            PromptReplacement(
                modality="audio",
                target="",  # Never match the prompt
                replacement=get_audio_replacement,
            ),
            PromptReplacement(
                modality="image", 
                target="",  # Never match the prompt
                replacement=get_image_replacement,
            ),
        ]

    def _cached_apply_hf_processor(
        self,
        prompt: Union[str, list[int]],
        mm_data_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
        *,
        return_mm_hashes: bool,
    ) -> tuple[list[int], MultiModalKwargs, Optional[MultiModalHashes], bool]:
        prompt_ids, mm_kwargs, mm_hashes, _ = super(
        )._cached_apply_hf_processor(
            prompt=prompt,
            mm_data_items=mm_data_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            tokenization_kwargs=tokenization_kwargs,
            return_mm_hashes=return_mm_hashes,
        )

        # NOTE: The tokens are already inserted by the chat template
        return prompt_ids, mm_kwargs, mm_hashes, True

    def _get_data_parser(self) -> MultiModalDataParser:
        sampling_rate = self.info.get_hf_processor().sampling_rate
        return MultiModalDataParser(target_sr=sampling_rate)


@MULTIMODAL_REGISTRY.register_processor(OmnistralMultiModalProcessor,
                                        info=OmnistralProcessingInfo,
                                        dummy_inputs=OmnistralDummyInputsBuilder)
class OmnistralForConditionalGeneration(nn.Module, SupportsMultiModal,
                                        SupportsPP, SupportsTranscription):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.tokenizer = cached_tokenizer_from_config(vllm_config.model_config)

        config = vllm_config.model_config.hf_config
        
        # Defensive validation
        if not hasattr(config, 'audio_config'):
            raise ValueError("Omnistral requires audio_config in model configuration")
        if not hasattr(config, 'vision_config'):
            raise ValueError("Omnistral requires vision_config in model configuration")
        
        # Audio configuration
        self.config = config
        self.audio_downsample_factor = self.config.audio_config.downsample_factor

        # Vision configuration
        from dataclasses import dataclass, fields
        dataclass_fields = {field.name for field in fields(VisionEncoderArgs)}
        vision_args = {
            key: value
            for key, value in self.config.vision_config.to_dict().items()
            if key in dataclass_fields
        }
        self.vision_args = VisionEncoderArgs(**vision_args)
        
        # Override adapter_bias to False for compatibility with checkpoints
        # that don't include bias weights
        self.vision_args.adapter_bias = False

        # Initialize language model
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

        # Initialize audio components (from Voxtral)
        self.whisper_encoder = VoxtralEncoderModel(
            vllm_config.with_hf_config(config.audio_config),
            prefix=maybe_prefix(prefix, "whisper_encoder"),
        )
        self.audio_language_adapter = AudioLanguageAdapter(
            hidden_size=config.audio_config.d_model * self.audio_downsample_factor,
            dim=config.text_config.hidden_size,
        )

        # Initialize vision components (from Pixtral)
        self.vision_encoder = VisionTransformer(self.vision_args)

        if self.vision_args.add_pre_mm_projector_layer_norm:
            from vllm.model_executor.layers.layernorm import RMSNorm
            self.pre_mm_projector_norm = RMSNorm(self.vision_args.hidden_size,
                                                 eps=1e-5)

        if self.vision_args.mm_projector_id == PATCH_MERGE:
            self.patch_merger = PatchMerger(
                vision_encoder_dim=self.vision_args.hidden_size,
                spatial_merge_size=self.vision_args.spatial_merge_size,
                use_mlp_bias=False,
            )

        self.vision_language_adapter = VisionLanguageAdapter(
            self.vision_args, dim=config.text_config.hidden_size)

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        print(f"DEBUG forward: intermediate_tensors is None = {intermediate_tensors is None}")
        print(f"DEBUG forward: inputs_embeds is None = {inputs_embeds is None}")
        print(f"DEBUG forward: kwargs keys = {kwargs.keys()}")
        
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            print("DEBUG: Calling get_multimodal_embeddings")
            multimodal_embeddings = self.get_multimodal_embeddings(**kwargs)
            print(f"DEBUG: Got embeddings, calling get_input_embeddings")
            inputs_embeds = self.get_input_embeddings(input_ids, multimodal_embeddings)
            input_ids = None

        hidden_states = self.language_model.model(input_ids,
                                                  positions,
                                                  intermediate_tensors,
                                                  inputs_embeds=inputs_embeds)

        return hidden_states

    def get_multimodal_embeddings(self, **kwargs) -> MultiModalEmbeddings:
        """Process both audio and image inputs and return combined embeddings."""
        # Process each modality separately
        audio_embeddings = self._process_audio_inputs(**kwargs)
        image_embeddings = self._process_image_inputs(**kwargs)
        
        # Return a flat list of all embeddings
        all_embeddings = []
        if audio_embeddings:
            all_embeddings.extend(audio_embeddings)
        if image_embeddings:
            all_embeddings.extend(image_embeddings)
        
        return all_embeddings
    
    def _merge_multimodal_embeddings_safe(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        multimodal_embeddings: list[torch.Tensor],
        placeholder_token_id: int,
    ) -> torch.Tensor:
        """
        Safely merge multimodal embeddings, handling count mismatches.
        This is a workaround for tokenizer not properly expanding tokens.
        """
        # Find positions of placeholder tokens
        positions = (input_ids == placeholder_token_id).nonzero(as_tuple=True)[0]
        
        if len(positions) == 0:
            return inputs_embeds
        
        # Flatten all embeddings
        all_embeds = torch.cat(multimodal_embeddings, dim=0)
        
        # Check if we have a count mismatch
        if all_embeds.shape[0] == len(positions):
            # Perfect match - use standard merge
            return merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings, placeholder_token_id
            )
        elif all_embeds.shape[0] < len(positions):
            # Fewer embeddings than positions - pad with zeros or repeat
            logger.warning(
                f"Fewer embeddings ({all_embeds.shape[0]}) than positions ({len(positions)}). "
                "Padding with zeros."
            )
            padded_embeds = torch.zeros(
                (len(positions), inputs_embeds.shape[1]),
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device
            )
            padded_embeds[:all_embeds.shape[0]] = all_embeds
            inputs_embeds[positions] = padded_embeds
        else:
            # More embeddings than positions - sample or truncate
            logger.warning(
                f"More embeddings ({all_embeds.shape[0]}) than positions ({len(positions)}). "
                f"Using strided sampling to fit."
            )
            # Use strided sampling to select which embeddings to keep
            stride = all_embeds.shape[0] / len(positions)
            indices = [int(i * stride) for i in range(len(positions))]
            sampled_embeds = all_embeds[indices]
            inputs_embeds[positions] = sampled_embeds
        
        return inputs_embeds

    def _process_audio_inputs(self, **kwargs) -> list[torch.Tensor]:
        """Process audio inputs using Voxtral components"""
        audio_inputs = self._parse_and_validate_audio_arrays(**kwargs)
        if audio_inputs is None:
            return []

        audio_embeddings = self.whisper_encoder(audio_inputs)

        # Apply downsampling and reshaping (from Voxtral)
        for i, audio_embedding in enumerate(audio_embeddings):
            seq_len, dim = audio_embedding.shape
            # Pad such that seq_len is divisible by downsample_factor
            target_seq_len = self.audio_downsample_factor * math.ceil(
                seq_len / self.audio_downsample_factor)
            audio_embedding = torch.nn.functional.pad(
                audio_embedding,
                (0, 0, 0, target_seq_len - seq_len),
            )
            audio_embeddings[i] = audio_embedding.reshape(
                target_seq_len // self.audio_downsample_factor,
                dim * self.audio_downsample_factor)

        # Project through audio adapter
        audio_embeddings_packed = torch.cat(audio_embeddings, dim=0)
        audio_embeddings_packed = self.audio_language_adapter(
            audio_embeddings_packed)
        audio_embeddings = torch.split(audio_embeddings_packed,
                                       [a.shape[0] for a in audio_embeddings],
                                       dim=0)

        return list(audio_embeddings)

    def _process_image_inputs(self, **kwargs) -> list[torch.Tensor]:
        """Process image inputs using Pixtral components"""
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []

        images = image_input["images"]
        image_features = self.vision_encoder(images)
        feature_sizes = [
            image_feature.shape[0] for image_feature in image_features
        ]
        image_features = torch.cat(image_features)

        # Apply optional pre-projection normalization
        if (self.vision_args.add_pre_mm_projector_layer_norm and 
            hasattr(self, 'pre_mm_projector_norm')):
            image_features = self.pre_mm_projector_norm(image_features)

        # Apply optional patch merging
        if (self.vision_args.mm_projector_id == PATCH_MERGE and 
            hasattr(self, 'patch_merger')):
            patch_size = self.vision_args.patch_size
            spatial_merge_size_square = self.vision_args.spatial_merge_size**2
            img_patch_dims = [(img.shape[1] // patch_size,
                               img.shape[2] // patch_size) for img in images]
            feature_sizes = [
                feature_size // spatial_merge_size_square
                for feature_size in feature_sizes
            ]
            image_features = self.patch_merger(image_features,
                                               image_sizes=img_patch_dims)

        # Project through vision adapter
        image_embeds = self.vision_language_adapter(image_features)
        image_embeds = torch.split(image_embeds, feature_sizes)

        return list(image_embeds)
    
    def _count_token_groups(self, input_ids: torch.Tensor, token_id: int) -> int:
        """Count the number of contiguous groups of a specific token in input_ids."""
        mask = input_ids == token_id
        groups = 0
        in_group = False
        
        for is_token in mask:
            if is_token and not in_group:
                groups += 1
                in_group = True
            elif not is_token:
                in_group = False
        
        return groups

    def _parse_and_validate_audio_arrays(
            self, **kwargs: object) -> Union[list[torch.Tensor], None]:
        """Parse audio arrays (from Voxtral)"""
        audio_arrays = kwargs.pop("audio_arrays", None)
        if audio_arrays is None:
            return None

        if not isinstance(audio_arrays, (torch.Tensor, list)):
            raise ValueError("Incorrect type of audio_arrays. "
                             f"Got type: {type(audio_arrays)}")

        audio_arrays = flatten_bn(audio_arrays)
        if isinstance(audio_arrays, torch.Tensor):
            audio_arrays = list(audio_arrays.unbind(0))
        return audio_arrays

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[dict]:
        """Parse image inputs (from Pixtral)"""
        images = kwargs.pop("images", None)
        if images is None:
            return None

        if not isinstance(images, (torch.Tensor, list)):
            raise ValueError("Incorrect type of images. "
                             f"Got type: {type(images)}")

        return {
            "type": "pixel_values",
            "images": flatten_bn(images),
        }

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        """Merge multimodal embeddings with text embeddings."""
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        
        if multimodal_embeddings is None or not multimodal_embeddings:
            return inputs_embeds
        
        # Get token IDs
        audio_tok_id = self.tokenizer.instruct.audio_encoder.audio_token
        image_tok_id = self.tokenizer.instruct.mm_encoder.special_ids.img
        
        # Check what modalities are present
        has_audio = (input_ids == audio_tok_id).any()
        has_image = (input_ids == image_tok_id).any()
        
        # Handle different cases
        if has_audio and not has_image:
            # Only audio - use safe merge
            inputs_embeds = self._merge_multimodal_embeddings_safe(
                input_ids, inputs_embeds, multimodal_embeddings, audio_tok_id
            )
        
        elif has_image and not has_audio:
            # Only image - use safe merge
            inputs_embeds = self._merge_multimodal_embeddings_safe(
                input_ids, inputs_embeds, multimodal_embeddings, image_tok_id
            )
        
        elif has_audio and has_image:
            # Both modalities - need to split embeddings
            logger.info("Processing mixed audio-image input")
            
            # Count tokens to estimate split
            num_audio_tokens = (input_ids == audio_tok_id).sum().item()
            
            # Simple heuristic: audio embeddings usually match their token count closely
            # Find where to split based on embedding sizes
            split_idx = 0
            cumulative_size = 0
            
            for i, emb in enumerate(multimodal_embeddings):
                cumulative_size += emb.shape[0]
                # If we've accumulated about the right number for audio, split here
                if cumulative_size >= num_audio_tokens:
                    split_idx = i + 1
                    break
            
            # If we couldn't find a good split, try another approach
            if split_idx == 0 or split_idx == len(multimodal_embeddings):
                # Look for embedding that matches audio token count exactly
                for i, emb in enumerate(multimodal_embeddings):
                    if emb.shape[0] == num_audio_tokens:
                        split_idx = i + 1
                        break
            
            # Split and merge
            if 0 < split_idx < len(multimodal_embeddings):
                audio_embeds = multimodal_embeddings[:split_idx]
                image_embeds = multimodal_embeddings[split_idx:]
                
                logger.info(f"Split embeddings: {len(audio_embeds)} audio, {len(image_embeds)} image")
                
                # Merge each modality
                inputs_embeds = self._merge_multimodal_embeddings_safe(
                    input_ids, inputs_embeds, audio_embeds, audio_tok_id
                )
                inputs_embeds = self._merge_multimodal_embeddings_safe(
                    input_ids, inputs_embeds, image_embeds, image_tok_id
                )
            else:
                # Couldn't split properly - fall back to treating all as one type
                logger.warning("Could not split mixed modality embeddings. Using fallback.")
                # Try image first since that's what usually has the count mismatch
                try:
                    inputs_embeds = self._merge_multimodal_embeddings_safe(
                        input_ids, inputs_embeds, multimodal_embeddings, image_tok_id
                    )
                except Exception as e:
                    logger.error(f"Failed to merge as image: {e}")
                    # Try audio as last resort
                    inputs_embeds = self._merge_multimodal_embeddings_safe(
                        input_ids, inputs_embeds, multimodal_embeddings, audio_tok_id
                    )
        
        return inputs_embeds
    
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states,
                                                  sampling_metadata)

    # Speech-to-text support (from Voxtral)
    @classmethod
    def get_speech_to_text_config(cls, model_config: ModelConfig,
                                  task_type: str) -> SpeechToTextConfig:
        tokenizer = cached_tokenizer_from_config(model_config)
        audio_config = tokenizer.instruct.audio_encoder.audio_config
        max_audio_clip_s = audio_config.chunk_length_s
        sample_rate = audio_config.sampling_rate
        return SpeechToTextConfig(
            max_audio_clip_s=max_audio_clip_s,
            sample_rate=sample_rate,
            min_energy_split_window_size=None,
        )

    @classmethod
    def get_generation_prompt(cls, audio: np.ndarray,
                              model_config: ModelConfig,
                              stt_config: SpeechToTextConfig, language: str,
                              task_type: str,
                              request_prompt: str) -> PromptType:
        tokenizer = cached_tokenizer_from_config(model_config)
        audio = Audio(audio, int(stt_config.sample_rate), format="wav")
        req = TranscriptionRequest(model=model_config.model,
                                   audio=RawAudio.from_audio(audio),
                                   language=language)

        tokenized = tokenizer.instral.encode_transcription(req)
        audio = (tokenized.audios[0].audio_array, stt_config.sample_rate)
        prompts_dict = {"multi_modal_data": {"audio": audio}}
        prompts_dict["prompt_token_ids"] = tokenized.tokens
        return cast(PromptType, prompts_dict)

    @classmethod
    def validate_language(cls, language: str) -> bool:
        # Use Voxtral's validation
        from .voxtral import VoxtralForConditionalGeneration
        return VoxtralForConditionalGeneration.validate_language(language)

    @classmethod
    def get_num_audio_tokens(cls, audio_duration_s: float,
                             stt_config: SpeechToTextConfig,
                             model_config: ModelConfig) -> Optional[int]:
        tokenizer = cached_tokenizer_from_config(model_config)
        adapter = VoxtralProcessorAdapter(tokenizer)
        return adapter.get_num_audio_tokens(
            int(audio_duration_s * stt_config.sample_rate))

    def load_weights(self, weights: Iterable[tuple[str,
                                               torch.Tensor]]) -> set[str]:
        """
        Load weights for all components: language model, audio encoder, 
        vision encoder, and adapters.
        """
        # Weight remapping rules
        remapping_rules = [
            # Audio component remapping (from Voxtral)
            (r"mm_whisper_embeddings\.(.*)", r"\1"),
            (r"audio_language_projection\.(.*)", r"audio_language_adapter.\1"),
            (r"audio_language_adapter\.0\.weight", r"audio_language_adapter.w_in.weight"),
            (r"audio_language_adapter\.2\.weight", r"audio_language_adapter.w_out.weight"),
        ]

        # Get parameter dictionaries for direct loading
        audio_params = dict(
            nn.ModuleDict({
                "audio_language_adapter": self.audio_language_adapter,
            }).named_parameters())
            
        # Vision components - handle both prefixed and non-prefixed names
        vision_encoder_dict = dict(self.vision_encoder.named_parameters())
        vision_lang_adapter_dict = dict(
            self.vision_language_adapter.named_parameters())
            
        # Optional components
        patch_merger_dict = dict(self.patch_merger.named_parameters(
        )) if self.vision_args.mm_projector_id == PATCH_MERGE else dict()
        pre_mm_projector_norm_dict = dict(
            self.pre_mm_projector_norm.named_parameters(
        )) if self.vision_args.add_pre_mm_projector_layer_norm else dict()

        loaded_weights = set()

        def llm_weights_generator():
            nonlocal loaded_weights
            for name, w in weights:
                # Check for audio encoder weights
                is_audio_encoder = (
                    name.startswith("mm_whisper_embeddings") and
                    not name.startswith("mm_whisper_embeddings.tok_embeddings")
                    and not name.startswith(
                        "mm_whisper_embeddings.audio_language_projection"))

                # Check for vision component weights
                is_vision_encoder = name.startswith("mm_embeddings.vision_encoder")
                is_vision_lang_adapter = name.startswith("mm_embeddings.vision_language_projection")
                is_patch_merger = name.startswith("mm_embeddings.patch_merger")
                is_pre_mm_projector_norm = name.startswith("mm_embeddings.pre_mm_projector_norm")

                # Apply remapping rules
                for pattern, repl in remapping_rules:
                    if re.fullmatch(pattern, name):
                        name = re.sub(pattern, repl, name)

                # Load audio encoder weights
                if is_audio_encoder:
                    name = self.whisper_encoder.load_weight((name, w))
                    loaded_weights.add(f"whisper_encoder.{name}")
                    continue

                # Load audio adapter weights
                if name in audio_params:
                    param = audio_params[name]
                    with torch.no_grad():
                        default_weight_loader(param, w)
                    loaded_weights.add(name)
                    continue

                # Load vision encoder weights
                if is_vision_encoder:
                    # Remove "mm_embeddings.vision_encoder." prefix
                    trimmed_name = name.replace("mm_embeddings.vision_encoder.", "")
                    if trimmed_name in vision_encoder_dict:
                        param = vision_encoder_dict[trimmed_name]
                        with torch.no_grad():
                            default_weight_loader(param, w)
                        # Add the actual model parameter name, not the checkpoint name
                        loaded_weights.add(f"vision_encoder.{trimmed_name}")
                    continue

                # Load vision language adapter weights
                if is_vision_lang_adapter:
                    # Handle the numbered weight format (e.g., vision_language_projection.0.weight)
                    trimmed_name = name.replace("mm_embeddings.vision_language_projection.", "")
                    # Map numbered indices to named attributes
                    mapping = {
                        "0.weight": "w_in.weight",
                        "0.bias": "w_in.bias",
                        "2.weight": "w_out.weight",
                        "2.bias": "w_out.bias"
                    }
                    mapped_name = mapping.get(trimmed_name, trimmed_name)
                    if mapped_name in vision_lang_adapter_dict:
                        param = vision_lang_adapter_dict[mapped_name]
                        with torch.no_grad():
                            default_weight_loader(param, w)
                        # Add the actual model parameter name
                        loaded_weights.add(f"vision_language_adapter.{mapped_name}")
                    continue

                # Load patch merger weights
                if is_patch_merger and patch_merger_dict:
                    trimmed_name = name.replace("mm_embeddings.patch_merger.", "")
                    if trimmed_name in patch_merger_dict:
                        param = patch_merger_dict[trimmed_name]
                        with torch.no_grad():
                            default_weight_loader(param, w)
                        # Add the actual model parameter name
                        loaded_weights.add(f"patch_merger.{trimmed_name}")
                    continue

                # Load pre-projection norm weights
                if is_pre_mm_projector_norm and pre_mm_projector_norm_dict:
                    trimmed_name = name.replace("mm_embeddings.pre_mm_projector_norm.", "")
                    if trimmed_name in pre_mm_projector_norm_dict:
                        param = pre_mm_projector_norm_dict[trimmed_name]
                        with torch.no_grad():
                            default_weight_loader(param, w)
                        # Add the actual model parameter name
                        loaded_weights.add(f"pre_mm_projector_norm.{trimmed_name}")
                    continue

                # Skip multimodal text embeddings - they'll be handled separately
                if name == "mm_embeddings.tok_embeddings.weight":
                    # These are the unified text embeddings for the multimodal model
                    # We'll handle them after the language model loads its weights
                    continue

                # Pass remaining weights to language model (base LLM weights)
                yield (name, w)

        # Load language model weights
        for name in self.language_model.load_weights(llm_weights_generator()):
            loaded_weights.add(f"language_model.{name}")

        # Handle missing position embeddings for whisper encoder
        sin_key = "whisper_encoder.whisper_encoder.embed_positions.weight"
        if sin_key not in loaded_weights:
            loaded_weights.add(sin_key)

        return loaded_weights