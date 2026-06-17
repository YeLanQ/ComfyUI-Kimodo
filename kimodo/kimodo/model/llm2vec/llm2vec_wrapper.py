# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM2Vec encoder wrapper for Kimodo text conditioning."""

import os

import numpy as np
import torch

from .llm2vec import LLM2Vec


class LLM2VecEncoder:
    """LLM2Vec text embeddings."""

    def __init__(
        self,
        base_model_name_or_path: str,
        peft_model_name_or_path: str,
        dtype: str,
        llm_dim: int,
    ) -> None:
        torch_dtype = getattr(torch, dtype)
        self.llm_dim = llm_dim

        cache_dir = os.environ.get("HUGGINGFACE_CACHE_DIR")

        # Priority: TEXT_ENCODER_DIR + ADAPTER_DIR > TEXT_ENCODERS_DIR > original paths
        # TEXT_ENCODER_DIR: specific base model directory (set by node selection)
        # ADAPTER_DIR: specific adapter model folder path (set by node selection)
        # TEXT_ENCODERS_DIR: base text encoders directory
        text_encoder_dir = os.environ.get("TEXT_ENCODER_DIR")
        adapter_dir = os.environ.get("ADAPTER_DIR")
        text_encoders_dir = os.environ.get("TEXT_ENCODERS_DIR")
        
        if text_encoder_dir and os.path.isdir(text_encoder_dir):
            # TEXT_ENCODER_DIR is a directory - use it as base model path directly
            base_model_name_or_path = text_encoder_dir
            # Adapter: use ADAPTER_DIR if set, otherwise try original path
            if adapter_dir and os.path.isdir(adapter_dir):
                peft_model_name_or_path = os.path.join(adapter_dir, os.path.basename(peft_model_name_or_path))
            else:
                peft_model_name = os.path.basename(peft_model_name_or_path)
                peft_model_name_or_path = os.path.join(os.path.dirname(text_encoder_dir), "adapter", peft_model_name)
        elif text_encoders_dir and os.path.isdir(text_encoders_dir):
            # Use base text encoders directory
            base_model_name_or_path = os.path.join(text_encoders_dir, base_model_name_or_path)
            peft_model_name_or_path = os.path.join(text_encoders_dir, peft_model_name_or_path)

        self.model = LLM2Vec.from_pretrained(
            base_model_name_or_path=base_model_name_or_path,
            peft_model_name_or_path=peft_model_name_or_path,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def to(self, device: torch.device):
        self.model = self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self

    def get_device(self):
        return self.model.model.device

    def __call__(self, text: list[str] | str):
        is_string = False
        if isinstance(text, str):
            text = [text]
            is_string = True

        with torch.no_grad():
            encoded_text = self.model.encode(text, batch_size=len(text), show_progress_bar=False)

        assert len(encoded_text.shape)
        assert self.llm_dim == encoded_text.shape[-1]

        encoded_text = encoded_text[:, None]
        lengths = np.ones(len(encoded_text), dtype=int).tolist()

        if is_string:
            encoded_text = encoded_text[0]
            lengths = lengths[0]

        encoded_text = torch.tensor(encoded_text).to(self.get_device())
        return encoded_text, lengths
