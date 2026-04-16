# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniCPM-V HF5 model configuration (nested text_config + vision_config)."""

from transformers.configuration_utils import PretrainedConfig
from transformers.models.idefics2.configuration_idefics2 import (
    Idefics2VisionConfig,
)

from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5TextConfig


class MiniCPMV4_6SliceConfig(PretrainedConfig):
    model_type = "minicpmv4_6_slice"

    def __init__(
        self,
        patch_size=14,
        max_slice_nums=9,
        scale_resolution=448,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.max_slice_nums = max_slice_nums
        self.scale_resolution = scale_resolution


class MiniCPMV4_6Config(PretrainedConfig):
    model_type = "minicpmv4_6"
    sub_configs = {
        "text_config": Qwen3_5TextConfig,
        "vision_config": Idefics2VisionConfig,
    }

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        slice_config=None,
        insert_layer_id=6,
        query_num=64,
        image_size=448,
        drop_vision_last_layer=False,
        batch_vision_input=True,
        use_image_id=True,
        vision_batch_size=16,
        slice_mode=True,
        image_token_id=None,
        tie_word_embeddings=False,
        downsample_mode="16x",
        **kwargs,
    ):
        self.insert_layer_id = insert_layer_id
        self.query_num = query_num
        self.image_size = image_size
        self.drop_vision_last_layer = drop_vision_last_layer
        self.batch_vision_input = batch_vision_input
        self.use_image_id = use_image_id
        self.vision_batch_size = vision_batch_size
        self.slice_mode = slice_mode
        self.image_token_id = image_token_id
        self.downsample_mode = downsample_mode

        # --- Vision config ---
        if isinstance(vision_config, dict):
            vc = {k: v for k, v in vision_config.items() if k != "model_type"}
            self.vision_config = Idefics2VisionConfig(**vc)
        elif vision_config is None:
            self.vision_config = Idefics2VisionConfig()
        else:
            self.vision_config = vision_config

        # --- Slice config ---
        if isinstance(slice_config, dict):
            self.slice_config = MiniCPMV4_6SliceConfig(**slice_config)
        elif slice_config is None:
            self.slice_config = MiniCPMV4_6SliceConfig(max_slice_nums=1)
        else:
            self.slice_config = slice_config

        # --- Text config ---
        if isinstance(text_config, dict):
            self.text_config = Qwen3_5TextConfig(**text_config)
        elif text_config is None:
            self.text_config = Qwen3_5TextConfig()
        else:
            self.text_config = text_config

        super().__init__(**kwargs)
        self.tie_word_embeddings = tie_word_embeddings


__all__ = ["MiniCPMV4_6Config", "MiniCPMV4_6SliceConfig"]
