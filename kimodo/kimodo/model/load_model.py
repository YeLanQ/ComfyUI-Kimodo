# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load Kimodo diffusion models from local checkpoints or Hugging Face."""

import os
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download
from omegaconf import OmegaConf

from .loading import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    DEFAULT_TEXT_ENCODER_URL,
    MODEL_NAMES,
    TMR_MODELS,
    get_env_var,
    instantiate_from_dict,
)
from .registry import get_model_info, resolve_model_name

DEFAULT_TEXT_ENCODER = "llm2vec"
TEXT_ENCODER_PRESETS = {
    "llm2vec": {
        "target": "kimodo.model.LLM2VecEncoder",
        "kwargs": {
            "base_model_name_or_path": "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            "peft_model_name_or_path": "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
            "dtype": "bfloat16",
            "llm_dim": 4096,
        },
    }
}


def _resolve_hf_model_path(modelname: str) -> Path:
    """Resolve model name to a local path, using CHECKPOINT_DIR or Hugging Face cache."""
    try:
        repo_id = MODEL_NAMES[modelname]
    except KeyError:
        raise ValueError(f"Model '{modelname}' not found. Available models: {MODEL_NAMES.keys()}")

    # 1. First, check CHECKPOINT_DIR for local model
    configured_checkpoint_dir = get_env_var("CHECKPOINT_DIR")
    if configured_checkpoint_dir:
        info = get_model_info(modelname)
        checkpoint_folder_name = info.display_name if info is not None else modelname
        local_path = Path(configured_checkpoint_dir) / checkpoint_folder_name
        if local_path.exists() and (local_path / "config.yaml").exists():
            print(f"Found local model in CHECKPOINT_DIR: {local_path}")
            return local_path
        # Fallback: try short_key
        local_path = Path(configured_checkpoint_dir) / modelname
        if local_path.exists() and (local_path / "config.yaml").exists():
            print(f"Found local model in CHECKPOINT_DIR: {local_path}")
            return local_path

    # 2. Check HuggingFace local cache (no network)
    try:
        snapshot_dir = snapshot_download(repo_id=repo_id, local_files_only=True)
        return Path(snapshot_dir)
    except Exception:
        pass

    # 3. Download from HuggingFace (last resort)
    try:
        snapshot_dir = snapshot_download(repo_id=repo_id)
        return Path(snapshot_dir)
    except Exception:
        raise RuntimeError(
            f"Could not resolve model '{modelname}' from Hugging Face (repo: {repo_id}). "
            f"Please download the model to {configured_checkpoint_dir or 'models/Kimodo'}."
        ) from None


def _build_api_text_encoder_conf(text_encoder_url: str) -> dict:
    return {
        "_target_": "kimodo.model.text_encoder_api.TextEncoderAPI",
        "url": text_encoder_url,
    }


def _build_local_text_encoder_conf(text_encoder_dir: Optional[str] = None) -> dict:
    """Build local text encoder configuration.
    
    Args:
        text_encoder_dir: Optional custom directory for text encoder models.
                        If provided, model paths are resolved relative to this directory.
                        If None, uses TEXT_ENCODERS_DIR env var or default paths.
    """
    text_encoder_name = get_env_var("TEXT_ENCODER", DEFAULT_TEXT_ENCODER)
    if text_encoder_name not in TEXT_ENCODER_PRESETS:
        available = ", ".join(sorted(TEXT_ENCODER_PRESETS))
        raise ValueError(f"Unknown TEXT_ENCODER='{text_encoder_name}'. Available: {available}")

    preset = TEXT_ENCODER_PRESETS[text_encoder_name]
    kwargs = dict(preset["kwargs"])
    
    # Use custom text encoder directory if provided
    if text_encoder_dir:
        base = kwargs["base_model_name_or_path"]
        peft = kwargs["peft_model_name_or_path"]
        # Only prepend if not already absolute
        if not os.path.isabs(base):
            kwargs["base_model_name_or_path"] = os.path.join(text_encoder_dir, base)
        if not os.path.isabs(peft):
            kwargs["peft_model_name_or_path"] = os.path.join(text_encoder_dir, peft)
    
    return {
        "_target_": preset["target"],
        **kwargs,
    }


def _select_text_encoder_conf(text_encoder_url: str, text_encoder_dir: Optional[str] = None) -> dict:
    """Select text encoder configuration based on mode.
    
    Args:
        text_encoder_url: URL for remote text encoder API.
        text_encoder_dir: Optional custom directory for local text encoder models.
    """
    # TEXT_ENCODER_MODE options:
    # - "api": force TextEncoderAPI
    # - "local": force local LLM2VecEncoder
    # - "auto": try API first, fallback to local if unreachable
    mode = get_env_var("TEXT_ENCODER_MODE", "auto").lower()
    if mode == "local":
        return _build_local_text_encoder_conf(text_encoder_dir)
    if mode == "api":
        return _build_api_text_encoder_conf(text_encoder_url)

    api_conf = _build_api_text_encoder_conf(text_encoder_url)
    try:
        text_encoder = instantiate_from_dict(api_conf)
        # Probe availability early so inference doesn't fail later.
        text_encoder(["healthcheck"])
        return api_conf
    except Exception as error:
        print(
            "Text encoder service is unreachable, falling back to local LLM2Vec "
            f"encoder. ({type(error).__name__}: {error})"
        )
        return _build_local_text_encoder_conf(text_encoder_dir)


def load_model(
    modelname=None,
    device=None,
    eval_mode: bool = True,
    default_family: Optional[str] = "Kimodo",
    return_resolved_name: bool = False,
    text_encoder_dir: Optional[str] = None,
):
    """Load a kimodo model by name (e.g. 'g1', 'soma').

    Resolution of partial/full names (e.g. Kimodo-SOMA-RP-v1, SOMA) is done
    inside this function using default_family when the name is not a known
    short key.

    Args:
        modelname: Model identifier; uses DEFAULT_MODEL if None. Can be a short key,
            a full name (e.g. Kimodo-SOMA-RP-v1), or a partial name; unknown names
            are resolved via resolve_model_name using default_family.
        device: Target device for the model (e.g. 'cuda', 'cpu').
        eval_mode: If True, set model to eval mode.
        default_family: Used when modelname is not in AVAILABLE_MODELS to resolve
            partial names ("Kimodo" for demo/generation, "TMR" for embed script).
            Default "Kimodo".
        return_resolved_name: If True, return (model, resolved_short_key). If False,
            return only the model.
        text_encoder_dir: Optional custom directory for text encoder models.
            If provided, text encoder will be loaded from this directory.

    Returns:
        Loaded model in eval mode, or (model, resolved short key) if
        return_resolved_name is True.

    Raises:
        ValueError: If modelname is not in AVAILABLE_MODELS and cannot be resolved.
        FileNotFoundError: If config.yaml is missing in the checkpoint folder.
    """
    if modelname is None:
        modelname = DEFAULT_MODEL

    # First, try to find model locally (before registry validation)
    configured_checkpoint_dir = get_env_var("CHECKPOINT_DIR")
    model_path = None
    resolved_modelname = modelname

    if configured_checkpoint_dir:
        print(f"CHECKPOINT_DIR is set to {configured_checkpoint_dir}, checking the local cache...")
        # Checkpoint folders are named by display name (e.g. Kimodo-SOMA-RP-v1)
        info = get_model_info(modelname)
        checkpoint_folder_name = info.display_name if info is not None else modelname
        model_path = Path(configured_checkpoint_dir) / checkpoint_folder_name
        if not model_path.exists() and modelname != checkpoint_folder_name:
            # Fallback: try short_key for backward compatibility
            model_path = Path(configured_checkpoint_dir) / modelname
        if not model_path.exists():
            model_path = None

    if model_path and model_path.exists():
        # Model found locally - use it directly
        print(f"Found local model at '{model_path}'")
        resolved_modelname = modelname
    else:
        # Not found locally - try registry resolution
        if modelname not in AVAILABLE_MODELS:
            if default_family is not None:
                modelname = resolve_model_name(modelname, default_family)
            else:
                raise ValueError(
                    f"""The model is not recognized.
                Please choose between: {AVAILABLE_MODELS}"""
                )
        resolved_modelname = modelname
        # Try to download from HuggingFace
        model_path = _resolve_hf_model_path(modelname)

    model_config_path = model_path / "config.yaml"
    if not model_config_path.exists():
        raise FileNotFoundError(f"The model checkpoint folder exists but config.yaml is missing: {model_config_path}")

    model_conf = OmegaConf.load(model_config_path)

    if modelname in TMR_MODELS:
        # Same process at the moment for TMR and Kimodo
        pass

    text_encoder_url = get_env_var("TEXT_ENCODER_URL", DEFAULT_TEXT_ENCODER_URL)
    
    # Use custom text_encoder_dir if provided, otherwise check env var
    effective_te_dir = text_encoder_dir or get_env_var("TEXT_ENCODER_DIR")
    
    runtime_conf = OmegaConf.create(
        {
            "checkpoint_dir": str(model_path),
            "text_encoder": _select_text_encoder_conf(text_encoder_url, effective_te_dir),
        }
    )
    model_cfg = OmegaConf.to_container(OmegaConf.merge(model_conf, runtime_conf), resolve=True)
    model_cfg.pop("checkpoint_dir", None)

    model = instantiate_from_dict(model_cfg, overrides={"device": device})
    if eval_mode:
        model = model.eval()
    if return_resolved_name:
        return model, resolved_modelname
    return model
