# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

from vllm.model_executor.models.minicpmv4_6 import (
    _get_audio_config,
    _has_audio_input,
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
