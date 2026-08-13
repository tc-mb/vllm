# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import torch

from vllm.model_executor.models.minicpmv4_6 import (
    MiniCPMV4_6ForConditionalGeneration,
    MiniCPMV4_6ProcessingInfo,
    _get_audio_config,
    _get_image_embed_token_id,
    _get_image_slices,
    _get_sliced_grid,
    _get_text_config,
    _get_text_model_type,
    _has_audio_input,
    _normalize_text_config,
)


def test_detect_audio_input_from_weight_index(tmp_path):
    index = {
        "weight_map": {
            "apm.conv1.weight": "model-00001-of-00001.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    config = SimpleNamespace(_name_or_path=str(tmp_path))

    assert _has_audio_input(config)


def test_no_audio_input_without_apm_weights(tmp_path):
    index = {
        "weight_map": {
            "model.language_model.embed_tokens.weight": (
                "model-00001-of-00001.safetensors"
            ),
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    config = SimpleNamespace(_name_or_path=str(tmp_path))

    assert not _has_audio_input(config)


def test_default_audio_config_matches_checkpoint():
    config = _get_audio_config(SimpleNamespace())

    assert config.num_mel_bins == 80
    assert config.d_model == 1024
    assert config.encoder_layers == 24
    assert config.encoder_attention_heads == 16
    assert config.encoder_ffn_dim == 4096
    assert config.max_source_positions == 1500


def test_flat_omni_config_is_used_as_text_config():
    config = SimpleNamespace(model_type="qwen3_5_moe_text", hidden_size=2048)

    assert _get_text_config(config) is config


def test_nested_text_config_is_preserved():
    text_config = SimpleNamespace(model_type="qwen3_5_moe_text", hidden_size=2048)
    config = SimpleNamespace(text_config=text_config)

    assert _get_text_config(config) is text_config


def test_flat_omni_config_uses_declared_text_model_type():
    config = SimpleNamespace(
        model_type="minicpmo",
        text_model_type="qwen3_5_moe_text",
    )

    assert _get_text_model_type(config) == "qwen3_5_moe_text"


def test_flat_omni_config_is_normalized_for_text_layers():
    config = SimpleNamespace(
        model_type="minicpmo",
        text_model_type="qwen3_5_moe_text",
    )

    assert _normalize_text_config(config) is config
    assert config.model_type == "qwen3_5_moe_text"


def test_native_image_processor_output_is_split():
    output = {
        "pixel_values": torch.zeros(1, 3, 2, 6),
        "target_sizes": torch.tensor([[1, 2], [1, 1]]),
    }

    slices, target_sizes = _get_image_slices(output, patch_size=2)

    assert [item.shape for item in slices] == [(3, 2, 4), (3, 2, 2)]
    assert torch.equal(target_sizes, output["target_sizes"])


def test_omni_image_processor_output_is_unwrapped():
    output = {
        "pixel_values": [[torch.zeros(3, 2, 4), torch.zeros(3, 2, 2)]],
        "tgt_sizes": [torch.tensor([[1, 2], [1, 1]])],
    }

    slices, target_sizes = _get_image_slices(output, patch_size=2)

    assert [item.shape for item in slices] == [(3, 2, 4), (3, 2, 2)]
    assert torch.equal(target_sizes, output["tgt_sizes"][0])


def test_omni_sliced_grid_does_not_enable_never_split():
    class ImageProcessor:
        def get_sliced_grid(
            self,
            image_size,
            max_slice_nums,
            never_split=False,
        ):
            assert not never_split
            return [4, 2]

    assert _get_sliced_grid(ImageProcessor(), (2353, 3722), 9, 448) == [4, 2]


def test_legacy_image_placeholder_uses_unk_embedding_token():
    class ImageProcessor:
        def get_slice_image_placeholder(self):
            pass

    tokenizer = SimpleNamespace(
        unk_token="<unk>",
        image_token="<|image_pad|>",
        encode=lambda text, add_special_tokens: {
            "<unk>": [248077],
            "<|image_pad|>": [248056],
        }[text],
    )

    assert _get_image_embed_token_id(tokenizer, ImageProcessor()) == 248077


def test_mm_max_tokens_per_item_includes_requested_modalities():
    processing_info = SimpleNamespace(
        get_max_image_tokens=lambda: 648,
        get_max_video_tokens=lambda seq_len, mm_counts: 1024,
        get_max_audio_tokens=lambda: 360,
    )

    result = MiniCPMV4_6ProcessingInfo.get_mm_max_tokens_per_item(
        processing_info,
        seq_len=800,
        mm_counts={"image": 1, "video": 1, "audio": 1},
    )

    assert result == {"image": 648, "video": 800, "audio": 360}


def test_visual_token_size_preserves_width_height_order():
    class ImageProcessor:
        max_slice_nums = 9
        patch_size = 14
        scale_resolution = 448

        def get_sliced_grid(
            self,
            image_size,
            max_slice_nums,
            never_split=False,
        ):
            assert image_size == (896, 448)
            return None

        def find_best_resize(
            self,
            image_size,
            scale_resolution,
            patch_size,
            allow_upscale=False,
        ):
            assert image_size == (896, 448)
            return image_size

    image_processor = ImageProcessor()
    processing_info = SimpleNamespace(
        get_image_processor=lambda: image_processor,
        _get_downsample_mode=lambda mode: mode or "16x",
    )

    grids, source_tokens, patch_tokens = (
        MiniCPMV4_6ProcessingInfo._compute_visual_tokens(
            processing_info,
            SimpleNamespace(width=896, height=448),
        )
    )

    assert grids == [0, 0]
    assert source_tokens == 128
    assert patch_tokens == 0


def test_omni_checkpoint_weight_prefixes_are_mapped():
    mapper = MiniCPMV4_6ForConditionalGeneration.hf_to_vllm_mapper

    assert (
        mapper._map_name("llm.model.embed_tokens.weight")
        == "language_model.model.embed_tokens.weight"
    )
    assert mapper._map_name("llm.lm_head.weight") == "language_model.lm_head.weight"
    assert (
        mapper._map_name("resampler.mlp.0.pre_norm.weight")
        == "merger.mlp.0.pre_norm.weight"
    )
    assert (
        mapper._map_name("resampler.mlp.0.mlp.0.weight")
        == "merger.mlp.0.linear_1.weight"
    )
    assert (
        mapper._map_name("resampler.mlp.0.mlp.2.weight")
        == "merger.mlp.0.linear_2.weight"
    )
    assert mapper._map_name("vit_merger.layer_norm2.weight") is None
    assert mapper._map_name("vit_merger.mlp.fc1.weight") is None
