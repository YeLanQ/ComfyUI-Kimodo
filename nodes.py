import os
import sys
import uuid
import math
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

print("[Kimodo] nodes.py: starting imports...", flush=True)

import torch
import numpy as np

import folder_paths
import comfy.model_management as mm
import comfy.utils

print("[Kimodo] nodes.py: comfy imports OK", flush=True)

# ---------------------------------------------------------------------------
# Path setup: add kimodo project root to sys.path so 'kimodo' package works
# ---------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
KIMODO_PROJECT_DIR = os.path.join(CURRENT_DIR, "kimodo")

if KIMODO_PROJECT_DIR not in sys.path:
    sys.path.insert(0, KIMODO_PROJECT_DIR)
    print(f"[Kimodo] Added to sys.path: {KIMODO_PROJECT_DIR}", flush=True)

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Install kimodo package in-place if not importable yet
try:
    import kimodo as _kimodo_pkg
    print(f"[Kimodo] kimodo package found: {_kimodo_pkg.__file__}", flush=True)
except ImportError:
    print("[Kimodo] kimodo package not installed, running pip install -e ...", flush=True)
    import subprocess
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-e", KIMODO_PROJECT_DIR,
        "--no-build-isolation",
    ], env={**os.environ, "SKIP_MOTION_CORRECTION_IN_SETUP": "1"})
    import kimodo as _kimodo_pkg
    print(f"[Kimodo] kimodo package installed: {_kimodo_pkg.__file__}", flush=True)

from kimodo import load_model, AVAILABLE_MODELS
from kimodo.constraints import load_constraints_lst
from kimodo.model.registry import MODEL_INFOS, KIMODO_MODELS, get_model_info
from kimodo.tools import seed_everything

print("[Kimodo] All imports OK", flush=True)

# ---------------------------------------------------------------------------
# Constants - Use ComfyUI standard model paths, no hardcoding
# ---------------------------------------------------------------------------
KIMODO_MODELS_DIR = os.path.join(folder_paths.models_dir, "Kimodo")
TEXT_ENCODERS_DIR = os.path.join(folder_paths.models_dir, "llm2vec")
os.makedirs(KIMODO_MODELS_DIR, exist_ok=True)
os.makedirs(TEXT_ENCODERS_DIR, exist_ok=True)

# Set TEXT_ENCODERS_DIR for LLM2Vec wrapper to resolve local text encoder paths
os.environ["TEXT_ENCODERS_DIR"] = TEXT_ENCODERS_DIR

# Set HuggingFace cache to Kimodo models dir to avoid network downloads
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = KIMODO_MODELS_DIR

print(f"[Kimodo] Kimodo models dir: {KIMODO_MODELS_DIR}", flush=True)
print(f"[Kimodo] Text encoders dir: {TEXT_ENCODERS_DIR}", flush=True)
print(f"[Kimodo] HF_HOME: {os.environ.get('HF_HOME', 'not set')}", flush=True)


def _scan_local_models() -> Dict[str, str]:
    """Scan Kimodo model directory and return mapping of display_name -> path.
    
    Supports both folder-based models (with config.yaml) and single safetensors files.
    Only scans local directory, no hardcoded registry.
    """
    local_models = {}
    
    if not os.path.exists(KIMODO_MODELS_DIR):
        return local_models
    
    for item in os.listdir(KIMODO_MODELS_DIR):
        item_path = os.path.join(KIMODO_MODELS_DIR, item)
        
        # Check if it's a folder with config.yaml (official format)
        if os.path.isdir(item_path):
            config_path = os.path.join(item_path, "config.yaml")
            if os.path.exists(config_path):
                local_models[item] = item_path
        
        # Check if it's a safetensors file (user-downloaded format)
        elif item.endswith(".safetensors"):
            display_name = item.replace(".safetensors", "")
            local_models[display_name] = item_path
    
    return local_models


def _scan_base_models() -> List[str]:
    """Scan TEXT_ENCODERS_DIR for available base model directories."""
    base_models = []
    if not os.path.exists(TEXT_ENCODERS_DIR):
        return base_models
    for item in os.listdir(TEXT_ENCODERS_DIR):
        item_path = os.path.join(TEXT_ENCODERS_DIR, item)
        # Check if it's a directory that contains model.safetensors (base model format)
        if os.path.isdir(item_path):
            if os.path.exists(os.path.join(item_path, "model.safetensors")):
                base_models.append(item)
    return sorted(base_models)


def _scan_adapter_models() -> List[str]:
    """Scan adapter directory for available adapter model folders."""
    adapter_models = []
    adapter_dir = os.path.join(TEXT_ENCODERS_DIR, "adapter")
    if not os.path.exists(adapter_dir):
        return adapter_models
    for item in os.listdir(adapter_dir):
        item_path = os.path.join(adapter_dir, item)
        # Check if it's a directory that contains adapter_config.json
        if os.path.isdir(item_path) and item != ".git":
            if os.path.exists(os.path.join(item_path, "adapter_config.json")):
                adapter_models.append(item)
    return sorted(adapter_models)


def _build_model_choices() -> List[str]:
    """Build model choices from local Kimodo directory only (no hardcoded registry)."""
    local_models = _scan_local_models()
    return list(local_models.keys())


# Build display name list for dropdown - scan directory only
_MODEL_CHOICES = _build_model_choices()
_BASE_MODEL_CHOICES = _scan_base_models()
_ADAPTER_MODEL_CHOICES = _scan_adapter_models()
print(f"[Kimodo] Available Kimodo models: {_MODEL_CHOICES}", flush=True)
print(f"[Kimodo] Available base models: {_BASE_MODEL_CHOICES}", flush=True)
print(f"[Kimodo] Available adapter models: {_ADAPTER_MODEL_CHOICES}", flush=True)





# ---------------------------------------------------------------------------
# Wrapper data classes
# ---------------------------------------------------------------------------
class KimodoCondData:
    """Wraps text conditioning output for passing between TextEncode and Sampler."""
    def __init__(self, text_feat, text_pad_mask, texts):
        self.text_feat = text_feat            # [B, max_len, dim] tensor on device
        self.text_pad_mask = text_pad_mask    # [B, max_len] bool tensor
        self.texts = texts                    # list of strings


class KimodoMotionData:
    """Wraps Kimodo generation output for passing between nodes."""
    def __init__(self, output_dict, model_name, skeleton_name, fps, texts, num_frames, num_samples,
                 joint_parents=None, joint_names=None, neutral_joints=None,
                 skeleton=None, constraint_lst=None):
        self.output_dict = output_dict        # dict with numpy/torch arrays
        self.model_name = model_name
        self.skeleton_name = skeleton_name
        self.fps = fps
        self.texts = texts
        self.num_frames = num_frames
        self.num_samples = num_samples
        self.batch_size = int(output_dict["posed_joints"].shape[0])
        self.joint_parents = joint_parents    # list of ints (-1 for root)
        self.joint_names = joint_names        # list of strings
        self.neutral_joints = neutral_joints  # [J, 3] T-pose positions (numpy)
        self.skeleton = skeleton              # skeleton object (for post-process)
        self.constraint_lst = constraint_lst  # constraints (for post-process)

    def reorder_joints(self, src_to_tgt_idx: list[int], tgt_joint_names: list[str],
                       tgt_joint_parents: list[int] | None = None) -> "KimodoMotionData":
        """Return a new KimodoMotionData with joints reordered/selected via *src_to_tgt_idx*.

        Each element of *src_to_tgt_idx* is the source joint index that maps to
        position ``i`` in the output.  ``tgt_joint_names`` and (optionally)
        ``tgt_joint_parents`` become the metadata on the returned motion.
        """
        od = self.output_dict
        new_od = dict(od)

        for key in ("posed_joints",):
            if key in od and od[key].ndim >= 3:
                new_od[key] = od[key][..., src_to_tgt_idx, :]

        for key in ("global_rot_mats", "local_rot_mats"):
            if key in od and od[key].shape[-2] >= len(src_to_tgt_idx):
                new_od[key] = od[key][..., src_to_tgt_idx, :, :]

        tgt_parents = tgt_joint_parents if tgt_joint_parents is not None else [-1] * len(tgt_joint_names)

        return KimodoMotionData(
            output_dict=new_od,
            model_name=self.model_name,
            skeleton_name=self.skeleton_name,
            fps=self.fps,
            texts=self.texts,
            num_frames=self.num_frames,
            num_samples=self.num_samples,
            joint_parents=tgt_parents,
            joint_names=tgt_joint_names,
            neutral_joints=None,
            skeleton=self.skeleton,
            constraint_lst=self.constraint_lst,
        )

    def combine_with(self, other: "KimodoMotionData", mode: str = "append",
                     frame_offset: int = 0,
                     overwrite_length: int = 0) -> "KimodoMotionData":
        """Combine this motion with *other* along the time (frame) axis.

        Modes
        -----
        ``append``
            Concatenate *other* after this motion's last frame.
        ``overwrite``
            Replace frames in this motion starting at *frame_offset* with
            *other*'s frames.  If *overwrite_length* > 0, only that many
            frames from *other* are used; otherwise all of *other* is used.
            If *other* overflows past the end of this motion the result is
            padded with zeros.

        Both motions must share the same skeleton structure (same joint
        count).  A warning is printed when joint names differ.
        """
        if self.joint_names and other.joint_names and self.joint_names != other.joint_names:
            print("[Kimodo] Warning: combining motions with different joint names", flush=True)

        _FRAME_KEYS = [
            "posed_joints", "global_rot_mats", "local_rot_mats",
            "root_positions", "smooth_root_pos", "foot_contacts",
            "global_root_heading",
        ]

        import numpy as np

        def _to_np(v):
            return v.cpu().numpy() if isinstance(v, torch.Tensor) else v

        new_od = dict(self.output_dict)

        for key in _FRAME_KEYS:
            s = self.output_dict.get(key)
            o = other.output_dict.get(key)
            if s is None and o is None:
                continue
            if s is None:
                new_od[key] = _to_np(o)
                continue
            if o is None:
                continue

            s_np = _to_np(s)
            o_np = _to_np(o)

            # Align last dimension if they differ (e.g. foot_contacts: 4 vs 6,
            # global_root_heading: 1 vs 2).  Truncate to the smaller size so
            # that np.concatenate along axis=1 does not raise.
            if s_np.ndim >= 3 and o_np.ndim >= 3 and s_np.shape[-1] != o_np.shape[-1]:
                min_last = min(s_np.shape[-1], o_np.shape[-1])
                print(f"[Kimodo] Warning: aligning '{key}' last dim: "
                      f"{s_np.shape[-1]} vs {o_np.shape[-1]} -> {min_last}", flush=True)
                s_np = s_np[..., :min_last]
                o_np = o_np[..., :min_last]

            if mode == "append":
                new_od[key] = np.concatenate([s_np, o_np], axis=1)

            elif mode == "overwrite":
                offset = frame_offset
                s_frames = s_np.shape[1]
                o_frames = o_np.shape[1]
                if overwrite_length > 0:
                    o_np = o_np[:, :min(overwrite_length, o_frames)]
                    o_frames = o_np.shape[1]
                total = max(s_frames, offset + o_frames)

                if total > s_frames:
                    pad_shape = list(s_np.shape)
                    pad_shape[1] = total - s_frames
                    buf = np.concatenate([s_np, np.zeros(pad_shape, dtype=s_np.dtype)], axis=1)
                else:
                    buf = s_np.copy()

                end = min(offset + o_frames, total)
                buf[:, offset:end] = o_np[:, :end - offset]
                new_od[key] = buf

        total_frames = new_od.get("posed_joints",
                                   self.output_dict.get("posed_joints")).shape[1]

        return KimodoMotionData(
            output_dict=new_od,
            model_name=self.model_name,
            skeleton_name=self.skeleton_name,
            fps=self.fps,
            texts=self.texts,
            num_frames=[total_frames],
            num_samples=self.num_samples,
            joint_parents=self.joint_parents,
            joint_names=self.joint_names,
            neutral_joints=None,
            skeleton=self.skeleton,
            constraint_lst=self.constraint_lst,
        )


class KimodoSkeletonData:
    """Wraps skeleton definition data for passing between nodes.

    Can be extracted from a loaded model or from generated motion data.
    """
    def __init__(self, joint_names: list[str], joint_parents: list[int],
                 neutral_joints=None, skeleton_name: str = "",
                 skeleton=None):
        self.joint_names = joint_names        # list of strings
        self.joint_parents = joint_parents    # list of ints (-1 for root)
        self.neutral_joints = neutral_joints  # [J, 3] or None
        self.skeleton_name = skeleton_name    # e.g. "somaskel30"
        self.skeleton = skeleton              # SkeletonBase instance or None
        self.num_joints = len(joint_names)


# ---------------------------------------------------------------------------
# Node: Load Model
# ---------------------------------------------------------------------------
class Kimodo_LoadModel:
    @classmethod
    def INPUT_TYPES(s):
        model_choices = _MODEL_CHOICES if _MODEL_CHOICES else ["No models found"]
        base_model_choices = _BASE_MODEL_CHOICES if _BASE_MODEL_CHOICES else ["model.safetensors"]
        adapter_model_choices = _ADAPTER_MODEL_CHOICES if _ADAPTER_MODEL_CHOICES else ["adapter"]
        
        return {
            "required": {
                "model": (model_choices, {
                    "default": model_choices[0],
                    "tooltip": "Select Kimodo model from models/Kimodo/ directory."
                }),
            },
            "optional": {
                "base_model": (base_model_choices, {
                    "default": base_model_choices[0],
                    "tooltip": "Select base text encoder from models/llm2vec/base_model/ directory."
                }),
                "adapter_model": (adapter_model_choices, {
                    "default": adapter_model_choices[0],
                    "tooltip": "Select adapter model from models/llm2vec/adapter/ directory."
                }),
            },
        }

    RETURN_TYPES = ("KIMODO_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "Kimodo"

    def load(self, model, base_model="base_model", adapter_model="adapter"):
        device = mm.get_torch_device()
        print(f"[Kimodo] Loading model: {model}", flush=True)

        # Set CHECKPOINT_DIR to ComfyUI Kimodo models folder
        os.environ["CHECKPOINT_DIR"] = KIMODO_MODELS_DIR
        
        # Set base model directory path
        base_model_dir = os.path.join(TEXT_ENCODERS_DIR, base_model)
        if os.path.isdir(base_model_dir):
            os.environ["TEXT_ENCODER_DIR"] = base_model_dir
            print(f"[Kimodo] Using base model dir: {base_model_dir}", flush=True)
        else:
            os.environ["TEXT_ENCODER_DIR"] = TEXT_ENCODERS_DIR
            print(f"[Kimodo] Base model dir not found: {base_model_dir}, using default", flush=True)
        
        # Set adapter model path
        adapter_model_path = os.path.join(TEXT_ENCODERS_DIR, "adapter", adapter_model)
        if os.path.isdir(adapter_model_path):
            os.environ["ADAPTER_DIR"] = adapter_model_path
            print(f"[Kimodo] Using adapter model: {adapter_model_path}", flush=True)
        else:
            os.environ["ADAPTER_DIR"] = os.path.join(TEXT_ENCODERS_DIR, "adapter")
            print(f"[Kimodo] Adapter model not found: {adapter_model_path}, using default", flush=True)

        # Load from local directory (models/Kimodo/)
        local_models = _scan_local_models()
        if model in local_models:
            local_path = local_models[model]
            print(f"[Kimodo] Found local model: {model} at {local_path}", flush=True)
            
            if os.path.isdir(local_path):
                return self._load_from_local(model, device)
            elif local_path.endswith(".safetensors"):
                return self._load_from_safetensors(local_path, model, device)

        # Model not found locally
        available = list(local_models.keys()) if local_models else ["No models found"]
        raise FileNotFoundError(
            f"Model '{model}' not found in {KIMODO_MODELS_DIR}. "
            f"Available: {available}"
        )

    def _load_from_safetensors(self, safetensors_path: str, model_name: str, device) -> tuple:
        """Load model directly from safetensors file."""
        print(f"[Kimodo] Loading from safetensors: {safetensors_path}", flush=True)
        
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
        
        from kimodo.model.registry import get_short_key_from_display_name
        
        short_key = get_short_key_from_display_name(model_name)
        if short_key:
            kimodo_model, resolved = load_model(
                short_key, device=str(device), return_resolved_name=True
            )
            try:
                kimodo_model.load_state_dict(state_dict, strict=False)
                print(f"[Kimodo] Loaded safetensors weights into model", flush=True)
            except Exception as e:
                print(f"[Kimodo] Warning: Could not load safetensors weights: {e}", flush=True)
            
            info = get_model_info(resolved)
            display = info.display_name if info else resolved
            print(f"[Kimodo] Model loaded: {display} (skeleton={kimodo_model.skeleton.name}, fps={kimodo_model.fps})", flush=True)
            return (kimodo_model,)
        
        print(f"[Kimodo] No registry match for {model_name}, trying default config", flush=True)
        kimodo_model, resolved = load_model(
            "kimodo-soma-rp", device=str(device), return_resolved_name=True
        )
        
        try:
            kimodo_model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"[Kimodo] Warning: Could not load safetensors weights: {e}", flush=True)
        
        info = get_model_info(resolved)
        display = info.display_name if info else resolved
        print(f"[Kimodo] Model loaded: {display} (skeleton={kimodo_model.skeleton.name}, fps={kimodo_model.fps})", flush=True)
        return (kimodo_model,)

    def _load_from_local(self, model: str, device) -> tuple:
        """Load model from local directory."""
        from kimodo.model.registry import get_short_key_from_display_name
        short_key = get_short_key_from_display_name(model)
        if short_key is None:
            short_key = model

        kimodo_model, resolved = load_model(
            short_key, device=str(device), return_resolved_name=True
        )

        info = get_model_info(resolved)
        display = info.display_name if info else resolved
        print(f"[Kimodo] Model loaded: {display} (skeleton={kimodo_model.skeleton.name}, fps={kimodo_model.fps})", flush=True)

        return (kimodo_model,)



# ---------------------------------------------------------------------------
# Node: Preview Motion (2D skeleton stick figure)
# ---------------------------------------------------------------------------
class Kimodo_Preview:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion": ("KIMODO_MOTION",),
                "sample_index": ("INT", {"default": 0, "min": 0, "max": 15,
                                         "tooltip": "Which sample to preview (0-indexed)"}),
                "frame_index": ("INT", {"default": 0, "min": 0,
                                        "tooltip": "Which frame to render (0-indexed)"}),
                "image_size": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 64}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "preview"
    CATEGORY = "Kimodo"

    def preview(self, motion, sample_index=0, frame_index=0, image_size=512):
        import cv2

        output = motion.output_dict
        joints = output["posed_joints"]  # [B, T, J, 3]
        parents = motion.joint_parents or []

        idx = min(sample_index, joints.shape[0] - 1)
        fidx = min(frame_index, joints.shape[1] - 1)
        frame_joints = joints[idx, fidx]  # [J, 3]

        img = np.ones((image_size, image_size, 3), dtype=np.uint8) * 40

        # Front view: X=horizontal, Y=vertical (inverted)
        x = frame_joints[:, 0]
        y = frame_joints[:, 1]

        # Normalize to image coordinates
        all_coords = np.stack([x, y], axis=-1)
        cmin = all_coords.min(axis=0)
        cmax = all_coords.max(axis=0)
        span = (cmax - cmin).max() + 1e-6
        margin = image_size * 0.1
        effective = image_size - 2 * margin

        px = ((x - cmin[0]) / span * effective + margin).astype(np.int32)
        py = (image_size - ((y - cmin[1]) / span * effective + margin)).astype(np.int32)

        # Draw bones using parent indices
        for j in range(len(px)):
            if j < len(parents) and parents[j] >= 0:
                p = parents[j]
                cv2.line(img, (int(px[j]), int(py[j])), (int(px[p]), int(py[p])), (100, 180, 255), 2)

        # Draw joints on top
        for j in range(len(px)):
            cv2.circle(img, (int(px[j]), int(py[j])), 3, (0, 220, 120), -1)

        cv2.putText(img, f"Frame {fidx}/{joints.shape[1]-1}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(img, f"Sample {idx}/{joints.shape[0]-1}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(img, f"Joints: {joints.shape[2]}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        result = img.astype(np.float32) / 255.0
        return (torch.from_numpy(result).unsqueeze(0),)


# ---------------------------------------------------------------------------
# Node: Save NPZ
# ---------------------------------------------------------------------------
class Kimodo_SaveNPZ:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion": ("KIMODO_MOTION",),
                "filename_prefix": ("STRING", {"default": "kimodo_motion"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save"
    CATEGORY = "Kimodo"
    OUTPUT_NODE = True

    def save(self, motion, filename_prefix="kimodo_motion"):
        output = motion.output_dict
        output_dir = folder_paths.get_output_directory()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        uid = str(uuid.uuid4())[:8]

        n_samples = motion.batch_size
        saved_paths = []

        if n_samples == 1:
            path = os.path.join(output_dir, f"{filename_prefix}_{ts}_{uid}.npz")
            single = {k: (v[0] if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == n_samples else v)
                      for k, v in output.items()}
            np.savez(path, **single)
            saved_paths.append(path)
            print(f"[Kimodo] Saved NPZ: {path}", flush=True)
        else:
            for i in range(n_samples):
                path = os.path.join(output_dir, f"{filename_prefix}_{ts}_{uid}_{i:02d}.npz")
                single = {k: (v[i] if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == n_samples else v)
                          for k, v in output.items()}
                np.savez(path, **single)
                saved_paths.append(path)
            print(f"[Kimodo] Saved {n_samples} NPZ files to {output_dir}", flush=True)

        return (saved_paths[0],)


# ---------------------------------------------------------------------------
# Node: Export BVH
# ---------------------------------------------------------------------------
class Kimodo_ExportBVH:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion": ("KIMODO_MOTION",),
                "filename_prefix": ("STRING", {"default": "kimodo_motion"}),
                "sample_index": ("INT", {"default": 0, "min": 0, "max": 15,
                                         "tooltip": "Which sample to export (0-indexed)"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "export"
    CATEGORY = "Kimodo"
    OUTPUT_NODE = True

    def export(self, motion, filename_prefix="kimodo_motion", sample_index=0):
        output = motion.output_dict
        output_dir = folder_paths.get_output_directory()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        uid = str(uuid.uuid4())[:8]

        from kimodo.exports.bvh import save_motion_bvh
        from kimodo.skeleton import global_rots_to_local_rots, SOMASkeleton30

        # BVH export requires SOMA skeleton
        device = mm.get_torch_device()

        # We need the skeleton - reload briefly for export
        from kimodo import load_model as _load
        # Use a lightweight approach: get skeleton from motion data
        # The skeleton info is embedded in the output

        try:
            model_for_skel = _load(motion.model_name, device=str(device), return_resolved_name=False)
            skeleton = model_for_skel.skeleton
        except Exception:
            print("[Kimodo] Warning: Could not load skeleton for BVH export. Trying SOMA default.", flush=True)
            model_for_skel = _load("kimodo-soma-rp", device=str(device), return_resolved_name=False)
            skeleton = model_for_skel.skeleton

        if "somaskel" not in skeleton.name:
            print("[Kimodo] BVH export is only supported for SOMA skeletons. Skipping.", flush=True)
            return ("",)

        if isinstance(skeleton, SOMASkeleton30):
            skeleton = skeleton.somaskel77.to(device)

        idx = min(sample_index, motion.batch_size - 1)
        joints_pos = torch.from_numpy(output["posed_joints"][idx]).to(device)
        joints_rot = torch.from_numpy(output["global_rot_mats"][idx]).to(device)
        local_rot_mats = global_rots_to_local_rots(joints_rot, skeleton)
        root_positions = joints_pos[:, skeleton.root_idx, :]

        path = os.path.join(output_dir, f"{filename_prefix}_{ts}_{uid}.bvh")
        save_motion_bvh(path, local_rot_mats, root_positions, skeleton=skeleton, fps=motion.fps)

        print(f"[Kimodo] Saved BVH: {path}", flush=True)
        return (path,)


# ---------------------------------------------------------------------------
# Node: Preview 3D (Three.js skeleton animation)
# ---------------------------------------------------------------------------
class Kimodo_Preview3D:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion": ("KIMODO_MOTION",),
                "sample_index": ("INT", {"default": 0, "min": 0, "max": 15,
                                         "tooltip": "Which sample to preview (0-indexed)"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "preview"
    CATEGORY = "Kimodo"
    OUTPUT_NODE = True

    def preview(self, motion, sample_index=0):
        import json

        output = motion.output_dict
        joints = output["posed_joints"]  # [B, T, J, 3]
        idx = min(sample_index, joints.shape[0] - 1)
        sample_joints = joints[idx]  # [T, J, 3]

        # Flatten to list for JSON
        joints_flat = sample_joints.flatten().tolist()

        # Get parent indices
        parents = motion.joint_parents or list(range(-1, sample_joints.shape[1] - 1))
        joint_names = motion.joint_names or [f"joint_{i}" for i in range(sample_joints.shape[1])]

        motion_json = json.dumps({
            "joints": joints_flat,
            "parents": parents,
            "joint_names": joint_names,
            "num_joints": int(sample_joints.shape[1]),
            "num_frames": int(sample_joints.shape[0]),
            "fps": int(motion.fps),
            "text": " ".join(motion.texts) if motion.texts else "",
            "skeleton": motion.skeleton_name,
        })

        print(f"[Kimodo] Preview3D: sample {idx}, {sample_joints.shape[0]} frames, "
              f"{sample_joints.shape[1]} joints", flush=True)

        return {"ui": {"motion_json": [motion_json]}}


# ---------------------------------------------------------------------------
# Node: Export FBX (Mixamo retarget)
# ---------------------------------------------------------------------------
class Kimodo_ExportFBX:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion": ("KIMODO_MOTION",),
                "custom_fbx_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to Mixamo-rigged FBX character. "
                               "Supports: 'input/3d/char.fbx', absolute path, "
                               "or relative to ComfyUI input/ folder.",
                }),
                "filename_prefix": ("STRING", {"default": "kimodo_fbx"}),
                "sample_index": ("INT", {"default": 0, "min": 0, "max": 15,
                                         "tooltip": "Which sample to export (0-indexed)"}),
            },
            "optional": {
                "yaw_offset": ("FLOAT", {
                    "default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0,
                    "tooltip": "Rotate character around Y-axis (degrees).",
                }),
                "scale": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01,
                    "tooltip": "Force scale multiplier (0 = auto height-based scaling).",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "export"
    CATEGORY = "Kimodo"
    OUTPUT_NODE = True

    @staticmethod
    def _resolve_fbx_path(path_str: str) -> str | None:
        """Resolve FBX path: absolute, input/…, output/…, or default to input/."""
        print(f"[Kimodo] _resolve_fbx_path input: '{path_str}'", flush=True)
        if not path_str or not path_str.strip():
            print("[Kimodo] FBX path is empty", flush=True)
            return None
        path = path_str.strip().replace("\\", "/")
        print(f"[Kimodo] Normalized path: '{path}'", flush=True)

        # Absolute
        if os.path.isabs(path):
            print(f"[Kimodo] Checking absolute: '{path}' exists={os.path.exists(path)}", flush=True)
            if os.path.exists(path):
                return path

        # Relative to ComfyUI root
        comfy_root = os.path.dirname(folder_paths.get_output_directory())
        print(f"[Kimodo] ComfyUI root: '{comfy_root}'", flush=True)

        candidates = []
        if path.startswith("input/") or path.startswith("output/"):
            candidates.append(os.path.normpath(os.path.join(comfy_root, path)))
        else:
            candidates.append(os.path.normpath(os.path.join(comfy_root, "input", path)))
            candidates.append(os.path.normpath(os.path.join(comfy_root, path)))

        for c in candidates:
            exists = os.path.exists(c)
            print(f"[Kimodo] Trying: '{c}' exists={exists}", flush=True)
            if exists:
                return c

        print(f"[Kimodo] FBX path not found in any candidate location", flush=True)
        return None

    def export(self, motion, custom_fbx_path, filename_prefix="kimodo_fbx",
               sample_index=0, yaw_offset=0.0, scale=0.0):

        from kimodo_retarget_fbx import export_kimodo_fbx, HAS_FBX_SDK

        if not HAS_FBX_SDK:
            print("[Kimodo] FBX SDK not installed. Install fbxsdkpy first.", flush=True)
            return ("FBX SDK not installed",)

        resolved = self._resolve_fbx_path(custom_fbx_path)
        if resolved is None:
            print(f"[Kimodo] No valid FBX path provided. Input was: '{custom_fbx_path}'", flush=True)
            print("[Kimodo] Please provide a Mixamo FBX file path. Examples:", flush=True)
            print("[Kimodo]   Absolute: I:/ComfyUI/input/3d/character.fbx", flush=True)
            print("[Kimodo]   Relative: 3d/character.fbx (looks in ComfyUI/input/)", flush=True)
            print("[Kimodo]   Prefixed: input/3d/character.fbx", flush=True)
            return ("",)

        output_dir = folder_paths.get_output_directory()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        uid = str(uuid.uuid4())[:8]
        idx = min(sample_index, motion.batch_size - 1)
        out_path = os.path.join(output_dir, f"{filename_prefix}_{ts}_{uid}.fbx")

        try:
            export_kimodo_fbx(
                motion_data=motion,
                target_fbx_path=resolved,
                output_path=out_path,
                sample_index=idx,
                yaw_offset=yaw_offset,
                force_scale=scale,
            )
            print(f"[Kimodo] FBX exported: {out_path}", flush=True)
        except Exception as e:
            print(f"[Kimodo] FBX export error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return (f"Export failed: {e}",)

        return (out_path,)


# ---------------------------------------------------------------------------
# Node: Text Encode (split from Generate)
# ---------------------------------------------------------------------------
class Kimodo_TextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("KIMODO_MODEL",),
                "prompt": ("STRING", {"default": "A person walks forward.",
                                      "multiline": True,
                                      "tooltip": "Text prompt. Use periods to separate multiple motion segments."}),
            },
        }

    RETURN_TYPES = ("KIMODO_COND",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "Kimodo/Conditioning"

    def encode(self, model, prompt):
        device = model.device

        texts = [t.strip() + "." for t in prompt.split(".") if t.strip()]
        if not texts:
            texts = [prompt]

        print(f"[Kimodo] TextEncode: {len(texts)} segment(s)", flush=True)
        for i, t in enumerate(texts):
            print(f"  [{i}] '{t}'", flush=True)

        from kimodo.model.kimodo_model import sanitize_texts
        texts = sanitize_texts(texts)

        text_feat, text_length = model.text_encoder(texts)
        text_feat = text_feat.to(device)

        empty_mask = [len(t.strip()) == 0 for t in texts]
        text_feat[empty_mask] = 0

        batch_size, maxlen = text_feat.shape[:2]
        tl = torch.tensor(text_length, device=device)
        tl[empty_mask] = 0
        text_pad_mask = torch.arange(maxlen, device=device).expand(batch_size, maxlen) < tl[:, None]

        print(f"[Kimodo] TextEncode: text_feat shape={text_feat.shape}", flush=True)

        cond = KimodoCondData(text_feat=text_feat, text_pad_mask=text_pad_mask, texts=texts)
        return (cond,)


# ---------------------------------------------------------------------------
# Node: Sampler (split from Generate — diffusion only, no post-processing)
# ---------------------------------------------------------------------------
class Kimodo_Sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("KIMODO_MODEL",),
                "conditioning": ("KIMODO_COND",),
                "duration": ("FLOAT", {"default": 5.0, "min": 5.0, "max": 30.0, "step": 0.5,
                                       "tooltip": "Duration in seconds per segment"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**32 - 1}),
                "num_samples": ("INT", {"default": 1, "min": 1, "max": 16}),
                "diffusion_steps": ("INT", {"default": 100, "min": 10, "max": 500}),
            },
            "optional": {
                "constraints_json": ("STRING", {"default": "",
                                                "tooltip": "Path to constraints JSON file (optional)"}),
                "existing_motion": ("KIMODO_MOTION", {
                    "tooltip": "Existing motion to extend/overwrite. If set, composition_mode must not be 'new'.",
                }),
                "composition_mode": (["new", "append", "overwrite"], {
                    "default": "new",
                    "tooltip": "'new' = fresh generation; 'append' = concatenate frames after existing; "
                               "'overwrite' = replace frames in existing at overwrite_frame offset",
                }),
                "overwrite_frame": ("INT", {"default": 0, "min": 0, "max": 100000,
                                            "tooltip": "Frame index where new motion overwrites existing (overwrite mode)"}),
                "overwrite_length": ("INT", {"default": 0, "min": 0, "max": 100000,
                                              "tooltip": "Number of frames to overwrite (0 = use all generated frames)"}),
            },
        }

    RETURN_TYPES = ("KIMODO_MOTION",)
    RETURN_NAMES = ("motion",)
    FUNCTION = "sample"
    CATEGORY = "Kimodo"

    def sample(self, model, conditioning, duration=5.0, seed=42, num_samples=1,
               diffusion_steps=100, constraints_json="", existing_motion=None,
               composition_mode="new", overwrite_frame=0, overwrite_length=0):

        seed_everything(seed)
        texts = conditioning.texts
        num_frames = [int(duration * model.fps)] * len(texts)
        multi_prompt = len(texts) > 1

        print(f"[Kimodo] Sampler: {len(texts)} segment(s), {num_frames[0]} frames, "
              f"{num_samples} sample(s), {diffusion_steps} steps, mode={composition_mode}", flush=True)

        constraint_lst = []
        if constraints_json and os.path.isfile(constraints_json):
            constraint_lst = load_constraints_lst(constraints_json, model.skeleton)
            print(f"[Kimodo] Loaded {len(constraint_lst)} constraint(s)", flush=True)

        # Call model WITHOUT post-processing
        output = model(
            texts,
            num_frames,
            num_denoising_steps=diffusion_steps,
            num_samples=num_samples,
            multi_prompt=multi_prompt,
            constraint_lst=constraint_lst,
            post_processing=False,
            return_numpy=True,
        )

        motion = self._wrap_output(model, output, texts, num_frames, num_samples, constraint_lst)

        # ------------------------------------------------------------------
        # Composition: append or overwrite into existing_motion
        # Inspired by Kimodo_Blender_Bridge's _apply_to_existing_source
        # (overwrite the armature action) and the segment system that imports
        # BVH at a specific start_frame offset (append).
        # ------------------------------------------------------------------
        if existing_motion is not None and composition_mode != "new":
            if composition_mode == "append":
                motion = existing_motion.combine_with(motion, mode="append")
                print(f"[Kimodo] Appended new motion after existing ({existing_motion.output_dict['posed_joints'].shape[1]} frames → {motion.output_dict['posed_joints'].shape[1]} frames)", flush=True)
            elif composition_mode == "overwrite":
                motion = existing_motion.combine_with(motion, mode="overwrite",
                                                       frame_offset=overwrite_frame,
                                                       overwrite_length=overwrite_length)
                print(f"[Kimodo] Overwrote existing motion at frame {overwrite_frame}", flush=True)

        return (motion,)

    @staticmethod
    def _wrap_output(model, output, texts, num_frames, num_samples, constraint_lst):
        from kimodo.skeleton.definitions import SOMASkeleton30
        skeleton_name = model.skeleton.name
        num_output_joints = output['posed_joints'].shape[2]
        skel_for_viz = model.skeleton
        if hasattr(model.skeleton, 'somaskel77') and num_output_joints == 77:
            skel_for_viz = model.skeleton.somaskel77
        elif hasattr(model.skeleton, 'somaskel30') and num_output_joints == 30:
            skel_for_viz = model.skeleton.somaskel30 if hasattr(model.skeleton, 'somaskel30') else model.skeleton

        joint_parents = skel_for_viz.joint_parents.cpu().tolist()
        joint_names = list(skel_for_viz.bone_order_names) if hasattr(skel_for_viz, 'bone_order_names') else []
        neutral_joints = None
        if hasattr(skel_for_viz, 'neutral_joints') and skel_for_viz.neutral_joints is not None:
            neutral_joints = skel_for_viz.neutral_joints.cpu().numpy()

        return KimodoMotionData(
            output_dict=output,
            model_name=str(getattr(model, '_resolved_name', 'unknown')),
            skeleton_name=skeleton_name,
            fps=model.fps,
            texts=texts,
            num_frames=num_frames,
            num_samples=num_samples,
            joint_parents=joint_parents,
            joint_names=joint_names,
            neutral_joints=neutral_joints,
            skeleton=model.skeleton,
            constraint_lst=constraint_lst,
        )


# ---------------------------------------------------------------------------
# Node: Post Process (foot-skate cleanup, separate from sampling)
# ---------------------------------------------------------------------------
class Kimodo_PostProcess:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion": ("KIMODO_MOTION",),
            },
            "optional": {
                "root_margin": ("FLOAT", {"default": 0.04, "min": 0.0, "max": 1.0, "step": 0.01,
                                          "tooltip": "Root correction margin in meters"}),
            },
        }

    RETURN_TYPES = ("KIMODO_MOTION",)
    RETURN_NAMES = ("motion",)
    FUNCTION = "process"
    CATEGORY = "Kimodo"

    def process(self, motion, root_margin=0.04):
        skeleton = motion.skeleton
        if skeleton is None:
            print("[Kimodo] PostProcess: No skeleton available, skipping.", flush=True)
            return (motion,)

        if "g1" in skeleton.name.lower():
            print("[Kimodo] PostProcess: G1 skeleton, skipping.", flush=True)
            return (motion,)

        output = motion.output_dict

        # Need torch tensors for post-processing
        def _to_torch(v):
            if isinstance(v, np.ndarray):
                return torch.from_numpy(v)
            return v

        local_rot_mats = _to_torch(output["local_rot_mats"])
        root_positions = _to_torch(output["root_positions"])
        foot_contacts = _to_torch(output["foot_contacts"])

        try:
            from kimodo.postprocess import post_process_motion
            print(f"[Kimodo] PostProcess: applying foot-skate correction (root_margin={root_margin})", flush=True)

            corrected = post_process_motion(
                local_rot_mats,
                root_positions,
                foot_contacts,
                skeleton,
                motion.constraint_lst or [],
                root_margin=root_margin,
            )

            # Update output dict
            new_output = dict(output)
            for k, v in corrected.items():
                new_output[k] = v.cpu().numpy() if isinstance(v, torch.Tensor) else v

            # Rebuild KimodoMotionData with corrected output
            new_motion = KimodoMotionData(
                output_dict=new_output,
                model_name=motion.model_name,
                skeleton_name=motion.skeleton_name,
                fps=motion.fps,
                texts=motion.texts,
                num_frames=motion.num_frames,
                num_samples=motion.num_samples,
                joint_parents=motion.joint_parents,
                joint_names=motion.joint_names,
                neutral_joints=motion.neutral_joints,
                skeleton=motion.skeleton,
                constraint_lst=motion.constraint_lst,
            )
            print("[Kimodo] PostProcess: done.", flush=True)
            return (new_motion,)

        except Exception as e:
            print(f"[Kimodo] PostProcess error: {e}", flush=True)
            print("[Kimodo] Returning uncorrected motion.", flush=True)
            import traceback
            traceback.print_exc()
            return (motion,)


# ---------------------------------------------------------------------------
# Node: Kimodo Configuration
# ---------------------------------------------------------------------------
class Kimodo_Config:
    """Configuration node for Kimodo text encoder settings."""
    
    @classmethod
    def INPUT_TYPES(s):
        base_model_choices = _BASE_MODEL_CHOICES if _BASE_MODEL_CHOICES else ["base_model"]
        adapter_model_choices = _ADAPTER_MODEL_CHOICES if _ADAPTER_MODEL_CHOICES else ["adapter"]
        
        return {
            "required": {
                "text_encoder_mode": (["local", "api", "auto"], {
                    "default": "local",
                    "tooltip": "Text encoder mode: 'local' uses local LLM2Vec (default), "
                               "'api' uses remote API, 'auto' tries API then local"
                }),
            },
            "optional": {
                "base_model": (base_model_choices, {
                    "default": base_model_choices[0],
                    "tooltip": "Select base text encoder from models/llm2vec/ directory."
                }),
                "adapter_model": (adapter_model_choices, {
                    "default": adapter_model_choices[0],
                    "tooltip": "Select adapter model from models/llm2vec/adapter/ directory."
                }),
                "text_encoder_url": ("STRING", {
                    "default": "http://127.0.0.1:9550/",
                    "tooltip": "URL for remote text encoder API (used when mode is 'api' or 'auto')"
                }),
            },
        }

    RETURN_TYPES = ("KIMODO_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "configure"
    CATEGORY = "Kimodo/Configuration"

    def configure(self, text_encoder_mode="local", base_model="base_model", 
                  adapter_model="adapter", text_encoder_url="http://127.0.0.1:9550/"):
        os.environ["TEXT_ENCODER_MODE"] = text_encoder_mode
        os.environ["TEXT_ENCODER_URL"] = text_encoder_url
        
        # Set base model directory path
        base_model_dir = os.path.join(TEXT_ENCODERS_DIR, base_model)
        if os.path.isdir(base_model_dir):
            os.environ["TEXT_ENCODER_DIR"] = base_model_dir
            print(f"[Kimodo] Configuration: base model dir = {base_model_dir}", flush=True)
        else:
            os.environ["TEXT_ENCODER_DIR"] = TEXT_ENCODERS_DIR
            print(f"[Kimodo] Configuration: base model dir not found, using default", flush=True)
        
        # Set adapter model path
        adapter_model_path = os.path.join(TEXT_ENCODERS_DIR, "adapter", adapter_model)
        if os.path.isdir(adapter_model_path):
            os.environ["ADAPTER_DIR"] = adapter_model_path
            print(f"[Kimodo] Configuration: adapter model = {adapter_model_path}", flush=True)
        else:
            os.environ["ADAPTER_DIR"] = os.path.join(TEXT_ENCODERS_DIR, "adapter")
            print(f"[Kimodo] Configuration: adapter model not found, using default", flush=True)
        
        config = {
            "text_encoder_mode": text_encoder_mode,
            "base_model": base_model,
            "adapter_model": adapter_model,
            "text_encoder_url": text_encoder_url,
        }
        
        print(f"[Kimodo] Configuration: text_encoder_mode={text_encoder_mode}", flush=True)
        
        return (config,)


# ---------------------------------------------------------------------------
# Skeleton operation nodes
# ---------------------------------------------------------------------------

class Kimodo_SkeletonInfo:
    """Extract skeleton metadata from motion data.

    Inspired by the bone hierarchy inspection in Kimodo_Blender_Bridge's
    retarget.py and properties.py — exposes joint names, parent indices,
    and rest-pose positions for inspection or downstream use.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion": ("KIMODO_MOTION",),
            },
        }

    RETURN_TYPES = ("KIMODO_SKELETON", "STRING", "STRING")
    RETURN_NAMES = ("skeleton", "joint_names_text", "summary")
    FUNCTION = "extract"
    CATEGORY = "Kimodo/Skeleton"

    def extract(self, motion):
        joint_names = motion.joint_names or []
        joint_parents = motion.joint_parents or []
        neutral_joints = motion.neutral_joints

        # Build formatted name list with hierarchy
        lines = []
        for i, name in enumerate(joint_names):
            p_idx = joint_parents[i] if i < len(joint_parents) else -1
            parent_name = joint_names[p_idx] if p_idx >= 0 and p_idx < len(joint_names) else "ROOT"
            lines.append(f"  [{i:2d}] {name:30s} parent={parent_name}")

        names_text = "\n".join(lines)

        skel_name = motion.skeleton_name or "unknown"
        n_joints = len(joint_names)
        summary = (
            f"Skeleton: {skel_name}\n"
            f"Joints  : {n_joints}\n"
            f"FPS     : {motion.fps}\n"
            f"Samples : {motion.batch_size}\n"
            f"Frames  : {motion.num_frames}"
        )

        skel_data = KimodoSkeletonData(
            joint_names=joint_names,
            joint_parents=joint_parents,
            neutral_joints=neutral_joints,
            skeleton_name=skel_name,
            skeleton=motion.skeleton,
        )

        return (skel_data, names_text, summary)


class Kimodo_Retarget:
    """Retarget motion by remapping/reordering joints via a bone name mapping.

    Inspired by Kimodo_Blender_Bridge's retarget.py — bone-pair mappings
    define how source bones map to output bones.  Supports three modes:

      * **Explicit mapping** — each line in the mapping text is a pair::

            source_bone -> target_bone

        Lines starting with ``#`` are ignored.  Unmapped source bones are
        dropped; target bones without a source are filled with zeros.

      * **Auto name match** — leave the mapping empty or set it to ``auto``
        and the node tries to align bone names case-insensitively.

      * **Identity** — pass ``identity`` to get a copy with the same bone
        order (useful as a passthrough that normalises the data wrapper).
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion": ("KIMODO_MOTION",),
                "bone_mapping": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": (
                        "Bone remapping. One 'src -> tgt' per line, or:\n"
                        "  'auto'     — case-insensitive name match\n"
                        "  'identity' — keep current order\n"
                        "  empty      — same as 'auto'"
                    ),
                }),
            },
        }

    RETURN_TYPES = ("KIMODO_MOTION",)
    RETURN_NAMES = ("motion",)
    FUNCTION = "retarget"
    CATEGORY = "Kimodo/Skeleton"

    def retarget(self, motion, bone_mapping):
        src_names = motion.joint_names or []
        if not src_names:
            print("[Kimodo_Retarget] No joint names on motion — returning as-is.", flush=True)
            return (motion,)

        mapping = (bone_mapping or "").strip().lower()

        if mapping in ("", "auto"):
            return self._auto_retarget(motion, src_names)
        elif mapping == "identity":
            return (motion,)

        return self._explicit_retarget(motion, src_names, bone_mapping)

    def _auto_retarget(self, motion, src_names):
        """Case-insensitive name matching (same as Blender's auto_build_mapping)."""
        src_lower = {n.lower(): n for n in src_names}
        tgt_order = list(src_names)  # default: same order

        # Try to sort so that "Hips" comes first, then spine, etc.
        def _priority(name: str) -> int:
            low = name.lower()
            if "hip" in low:
                return 0
            if "spine" in low or "chest" in low:
                return 1
            if "neck" in low or "head" in low:
                return 2
            if "shoulder" in low:
                return 3
            if "arm" in low or "forearm" in low or "hand" in low:
                return 4
            if "leg" in low or "shin" in low or "foot" in low or "toe" in low:
                return 5
            return 6

        tgt_order.sort(key=_priority)

        # Build index mapping: for each target bone, find the source index
        tgt_final: list[str] = []
        idx_map: list[int] = []
        for tgt in tgt_order:
            tgt_low = tgt.lower()
            found = None
            for i, s in enumerate(src_names):
                if s.lower() == tgt_low:
                    found = i
                    break
            if found is not None:
                tgt_final.append(src_names[found])
                idx_map.append(found)

        if not idx_map:
            print("[Kimodo_Retarget] Auto-map found no matches — returning as-is.", flush=True)
            return (motion,)

        # Rebuild parent indices for the new ordering
        src_to_new = {old: new for new, old in enumerate(idx_map)}
        tgt_parents = []
        for tgt_name in tgt_final:
            orig_idx = src_names.index(tgt_name) if tgt_name in src_names else -1
            orig_parent = motion.joint_parents[orig_idx] if orig_idx >= 0 and motion.joint_parents and orig_idx < len(motion.joint_parents) else -1
            if orig_parent >= 0 and orig_parent in src_to_new:
                tgt_parents.append(src_to_new[orig_parent])
            else:
                tgt_parents.append(-1)

        result = motion.reorder_joints(idx_map, tgt_final, tgt_parents)
        n = len(idx_map)
        print(f"[Kimodo_Retarget] Auto-mapped {n} bones ({len(src_names)} → {n})", flush=True)
        return (result,)

    def _explicit_retarget(self, motion, src_names, bone_mapping):
        """Parse user-provided 'src -> tgt' lines."""
        src_name_set = {n.lower(): n for n in src_names}

        tgt_final: list[str] = []
        idx_map: list[int] = []

        for raw_line in bone_mapping.strip().split("\n"):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            # Accept "->", "→", "=" as separators
            sep = None
            for s in ("->", "→", "="):
                if s in line:
                    sep = s
                    break
            if sep is None:
                continue
            src_part, tgt_part = line.split(sep, 1)
            src_name = src_part.strip()
            tgt_name = tgt_part.strip()
            if not src_name or not tgt_name:
                continue

            # Find source index
            src_low = src_name.lower()
            found = None
            for i, s in enumerate(src_names):
                if s.lower() == src_low:
                    found = i
                    break
            if found is None:
                print(f"[Kimodo_Retarget] Source bone '{src_name}' not found — skipping.", flush=True)
                continue

            tgt_final.append(tgt_name)
            idx_map.append(found)

        if not idx_map:
            print("[Kimodo_Retarget] No valid mappings — returning as-is.", flush=True)
            return (motion,)

        # Build parent indices for output skeleton
        src_to_new = {old: new for new, old in enumerate(idx_map)}
        tgt_parents = []
        for i, tgt_name in enumerate(tgt_final):
            orig_idx = idx_map[i]
            orig_parent = motion.joint_parents[orig_idx] if motion.joint_parents and orig_idx < len(motion.joint_parents) else -1
            if orig_parent >= 0 and orig_parent in src_to_new:
                tgt_parents.append(src_to_new[orig_parent])
            else:
                tgt_parents.append(-1)

        result = motion.reorder_joints(idx_map, tgt_final, tgt_parents)
        n = len(idx_map)
        print(f"[Kimodo_Retarget] Applied {n} bone mappings ({len(src_names)} → {n})", flush=True)
        return (result,)


class Kimodo_SelectBones:
    """Select a subset of bones from motion data by name.

    Provide one or more bone names separated by commas or newlines.
    Only the listed bones are kept in the output motion; all others
    are removed.  Useful for isolating specific body parts.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion": ("KIMODO_MOTION",),
                "bone_names": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Bone names to keep, comma or newline separated. Leave empty to keep all.",
                }),
            },
        }

    RETURN_TYPES = ("KIMODO_MOTION",)
    RETURN_NAMES = ("motion",)
    FUNCTION = "select"
    CATEGORY = "Kimodo/Skeleton"

    def select(self, motion, bone_names):
        src_names = motion.joint_names or []
        if not src_names:
            return (motion,)

        bone_names = (bone_names or "").strip()
        if not bone_names:
            return (motion,)

        # Parse bone name list (comma or newline separated)
        wanted: list[str] = []
        for chunk in bone_names.replace(",", "\n").split("\n"):
            name = chunk.strip()
            if name:
                wanted.append(name)

        if not wanted:
            return (motion,)

        # Build index mapping (case-insensitive)
        src_lower = {n.lower(): (i, n) for i, n in enumerate(src_names)}
        idx_map: list[int] = []
        kept_names: list[str] = []
        for w in wanted:
            w_low = w.lower()
            if w_low in src_lower:
                i, orig = src_lower[w_low]
                idx_map.append(i)
                kept_names.append(orig)

        if not idx_map:
            print("[Kimodo_SelectBones] No matching bones found — returning as-is.", flush=True)
            return (motion,)

        # Rebuild parent indices
        src_to_new = {old: new for new, old in enumerate(idx_map)}
        tgt_parents = []
        for i in idx_map:
            orig_parent = motion.joint_parents[i] if motion.joint_parents and i < len(motion.joint_parents) else -1
            if orig_parent >= 0 and orig_parent in src_to_new:
                tgt_parents.append(src_to_new[orig_parent])
            else:
                tgt_parents.append(-1)

        result = motion.reorder_joints(idx_map, kept_names, tgt_parents)
        n = len(idx_map)
        print(f"[Kimodo_SelectBones] Selected {n}/{len(src_names)} bones", flush=True)
        return (result,)


# ---------------------------------------------------------------------------
# Node: Load Motion (BVH / NPZ)
# ---------------------------------------------------------------------------
class Kimodo_LoadMotion:
    """Load BVH or NPZ motion files and convert to KimodoMotionData.

    Supports:
      - BVH files: parses skeleton hierarchy and motion channels.
      - NPZ files: loads Kimodo-saved NPZ (posed_joints, rot_mats, etc.)
        or raw motion arrays.

    For BVH, the skeleton hierarchy is extracted from the file automatically.
    For NPZ, an optional skeleton input provides joint names/parents;
    otherwise they are auto-detected from joint count.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to BVH or NPZ motion file. "
                               "Supports: 'input/motion.bvh', absolute path, "
                               "or relative to ComfyUI input/ folder.",
                }),
            },
            "optional": {
                "skeleton": ("KIMODO_SKELETON", {
                    "tooltip": "Optional skeleton for joint metadata "
                               "(required for NPZ without joint info).",
                }),
                "fps": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 240.0,
                                  "step": 0.1,
                                  "tooltip": "Override FPS. 0 = auto-detect "
                                             "(BVH reads frame time, NPZ defaults to 30)."}),
            },
        }

    RETURN_TYPES = ("KIMODO_MOTION", "KIMODO_SKELETON", "STRING")
    RETURN_NAMES = ("motion", "skeleton_data", "summary")
    FUNCTION = "load"
    CATEGORY = "Kimodo"

    @staticmethod
    def _resolve_path(path_str: str) -> str | None:
        """Resolve file path: absolute > ComfyUI/input/ > ComfyUI root.
        
        Mirrors the pattern used by TTS-Audio-Suite's _resolve_audio_file_path.
        """
        if not path_str or not path_str.strip():
            return None
        path = path_str.strip().replace("\\", "/")

        # Absolute path
        if os.path.isabs(path):
            if os.path.exists(path):
                return path

        # Relative to ComfyUI input directory (primary — files uploaded via JS land here)
        input_dir = folder_paths.get_input_directory()
        candidates = [
            os.path.normpath(os.path.join(input_dir, path)),
        ]

        # Also check subfolder/motion_name patterns
        comfy_root = os.path.dirname(folder_paths.get_output_directory())
        if path.startswith("input/") or path.startswith("output/"):
            candidates.append(os.path.normpath(os.path.join(comfy_root, path)))
        else:
            candidates.append(os.path.normpath(os.path.join(comfy_root, "input", path)))
            candidates.append(os.path.normpath(os.path.join(comfy_root, path)))

        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    def load(self, file_path, skeleton=None, fps=0.0):
        resolved = self._resolve_path(file_path)
        if resolved is None or not os.path.exists(resolved):
            print(f"[Kimodo_LoadMotion] File not found: {file_path}", flush=True)
            print("[Kimodo_LoadMotion] Supported locations: absolute path, "
                  "ComfyUI/input/, or relative to ComfyUI root.", flush=True)
            raise FileNotFoundError(f"Motion file not found: {file_path}")

        ext = os.path.splitext(resolved)[1].lower()
        print(f"[Kimodo_LoadMotion] Loading: {resolved}", flush=True)

        if ext == ".bvh":
            motion, skel_data, summary = self._load_bvh(resolved, fps)
        elif ext == ".npz":
            motion, skel_data, summary = self._load_npz(resolved, skeleton, fps)
        else:
            raise ValueError(f"Unsupported format: {ext}. Supported: .bvh, .npz")

        print(f"[Kimodo_LoadMotion] {summary}", flush=True)
        return (motion, skel_data, summary)

    def _load_bvh(self, path: str, fps: float):
        from kimodo.skeleton.bvh import SkeletonBvh, load_bvh_animation, Bvh
        from kimodo.skeleton.kinematics import batch_rigid_transform

        # --- Parse skeleton hierarchy from BVH ---
        skel_bvh = SkeletonBvh()
        exclude_bones = {"Root"}
        skel_bvh.load_from_bvh(path, exclude_bones=exclude_bones)

        joint_names = skel_bvh.get_bones_names()
        joint_parents = skel_bvh.get_parent_indices()
        neutral_joints_cm = skel_bvh.get_neutral_joints()  # [J, 3] in cm

        # --- Load animation ---
        root_trans, local_rot_mats = load_bvh_animation(path, skel_bvh)
        # root_trans: [T, 3] in cm, local_rot_mats: [T, J, 3, 3]

        T = local_rot_mats.shape[0]
        n_joints = len(joint_names)

        # Convert cm → m
        root_trans_np = root_trans * 0.01
        neutral_joints_np = neutral_joints_cm * 0.01

        # --- Auto-detect FPS ---
        if fps <= 0:
            try:
                with open(path) as f:
                    mocap = Bvh(f.read())
                fps = round(1.0 / mocap.frame_time)
            except Exception:
                fps = 30.0
        print(f"[Kimodo_LoadMotion] BVH: {T} frames, {n_joints} joints, {fps} FPS", flush=True)

        # --- Convert to torch ---
        device = "cpu"
        local_rot_mats_t = torch.from_numpy(
            local_rot_mats if isinstance(local_rot_mats, np.ndarray)
            else local_rot_mats.numpy()
        ).float().to(device)
        root_pos_t = torch.from_numpy(root_trans_np).float().to(device)

        # --- Compute FK: global_rot_mats + posed_joints ---
        root_idx = 0
        neutral_t = torch.from_numpy(neutral_joints_np).float().to(device)
        pelvis_offset = neutral_t[root_idx:root_idx + 1]
        neutral_centered = neutral_t - pelvis_offset

        joints_b = neutral_centered.unsqueeze(0).expand(T, -1, -1)  # [T, J, 3]
        parents_t = torch.tensor(joint_parents, dtype=torch.long, device=device)

        posed_joints_noroot, global_rot_mats = batch_rigid_transform(
            local_rot_mats_t, joints_b, parents_t, root_idx
        )
        # posed_joints_noroot: [T, J, 3] with root at origin
        # global_rot_mats: [T, J, 3, 3]

        posed_joints = posed_joints_noroot + root_pos_t.unsqueeze(1)  # [T, J, 3]

        # --- Build output_dict ---
        output_dict = {
            "posed_joints": posed_joints.cpu().numpy()[None],       # [1, T, J, 3]
            "global_rot_mats": global_rot_mats.cpu().numpy()[None], # [1, T, J, 3, 3]
            "local_rot_mats": local_rot_mats_t.cpu().numpy()[None], # [1, T, J, 3, 3]
            "root_positions": root_pos_t.cpu().numpy()[None],       # [1, T, 3]
            "smooth_root_pos": root_pos_t.cpu().numpy()[None],
            "foot_contacts": np.zeros((1, T, 4), dtype=np.float32),
            "global_root_heading": np.zeros((1, T, 1), dtype=np.float32),
        }

        motion_data = KimodoMotionData(
            output_dict=output_dict,
            model_name="bvh_loaded",
            skeleton_name="bvh_custom",
            fps=fps,
            texts=["Loaded from BVH"],
            num_frames=[T],
            num_samples=1,
            joint_parents=joint_parents,
            joint_names=joint_names,
            neutral_joints=neutral_joints_np,
            skeleton=None,
            constraint_lst=[],
        )

        skel_data = KimodoSkeletonData(
            joint_names=joint_names,
            joint_parents=joint_parents,
            neutral_joints=neutral_joints_np,
            skeleton_name="bvh_custom",
            skeleton=None,
        )

        summary = f"Loaded BVH: {T} frames, {n_joints} joints, {fps} FPS"
        return motion_data, skel_data, summary

    def _load_npz(self, path: str, skeleton, fps: float):
        data = np.load(path)

        # Detect format by checking keys
        if "posed_joints" in data:
            # Kimodo NPZ format (from SaveNPZ node)
            posed_joints = data["posed_joints"]   # [T, J, 3] or [1, T, J, 3]
            has_batch = posed_joints.ndim == 4
            if has_batch:
                posed_joints = posed_joints[0]

            T, n_joints = posed_joints.shape[:2]

            output_dict = {}
            for key in ["posed_joints", "global_rot_mats", "local_rot_mats",
                         "root_positions", "smooth_root_pos",
                         "foot_contacts", "global_root_heading"]:
                if key in data:
                    arr = data[key]
                    if arr.ndim == 4 and arr.shape[0] == 1:
                        arr = arr[0]
                    output_dict[key] = arr[None]  # re-add batch dim
                else:
                    # Generate placeholder
                    if key == "foot_contacts":
                        output_dict[key] = np.zeros((1, T, 4), dtype=np.float32)
                    elif key == "global_root_heading":
                        output_dict[key] = np.zeros((1, T, 1), dtype=np.float32)
                    elif key == "smooth_root_pos" and "root_positions" in output_dict:
                        output_dict[key] = output_dict["root_positions"]
                    else:
                        output_dict[key] = np.zeros((1, T, n_joints, 3), dtype=np.float32) \
                            if "mats" in key else np.zeros((1, T, 3), dtype=np.float32)

            root_pos = output_dict.get("root_positions", np.zeros((1, T, 3)))
        else:
            # Unknown NPZ — try to find motion arrays by common names
            n_joints = self._detect_joint_count(data)
            T = self._detect_frame_count(data)

            # Try to find rotation data
            local_rot_mats = None
            for key in ("local_rot_mats", "rots", "rotations", "joint_rots"):
                if key in data:
                    local_rot_mats = data[key]
                    break

            root_positions_arr = None
            for key in ("root_positions", "root_trans", "trans", "root"):
                if key in data:
                    root_positions_arr = data[key]
                    break

            posed_joints_arr = None
            for key in ("posed_joints", "joints", "positions", "joint_positions"):
                if key in data:
                    posed_joints_arr = data[key]
                    break

            output_dict = {
                "posed_joints": (posed_joints_arr if posed_joints_arr is not None
                                 else np.zeros((T, n_joints, 3)))[None],
                "root_positions": (root_positions_arr if root_positions_arr is not None
                                   else np.zeros((T, 3)))[None],
                "foot_contacts": np.zeros((1, T, 4), dtype=np.float32),
                "global_root_heading": np.zeros((1, T, 1), dtype=np.float32),
            }

            if local_rot_mats is not None:
                output_dict["local_rot_mats"] = local_rot_mats[None]
                output_dict["global_rot_mats"] = local_rot_mats[None]
            else:
                output_dict["local_rot_mats"] = np.zeros((1, T, n_joints, 3, 3), dtype=np.float32)
                output_dict["global_rot_mats"] = np.zeros((1, T, n_joints, 3, 3), dtype=np.float32)

            output_dict["smooth_root_pos"] = output_dict["root_positions"]

        # --- Joint names / parents ---
        if skeleton is not None:
            joint_names = skeleton.joint_names
            joint_parents = skeleton.joint_parents
            neutral_joints = skeleton.neutral_joints
            skel_name = skeleton.skeleton_name
        else:
            # Auto-detect by joint count
            joint_names, joint_parents, neutral_joints, skel_name = \
                self._auto_skeleton(n_joints)

        # --- FPS ---
        if fps <= 0:
            fps = 30.0

        motion_data = KimodoMotionData(
            output_dict=output_dict,
            model_name="npz_loaded",
            skeleton_name=skel_name,
            fps=fps,
            texts=["Loaded from NPZ"],
            num_frames=[T],
            num_samples=1,
            joint_parents=joint_parents,
            joint_names=joint_names,
            neutral_joints=neutral_joints,
            skeleton=None,
            constraint_lst=[],
        )

        skel_data = KimodoSkeletonData(
            joint_names=joint_names,
            joint_parents=joint_parents,
            neutral_joints=neutral_joints,
            skeleton_name=skel_name,
            skeleton=None,
        )

        summary = f"Loaded NPZ: {T} frames, {n_joints} joints, {fps} FPS"
        return motion_data, skel_data, summary

    @staticmethod
    def _detect_joint_count(data) -> int:
        for key in ("posed_joints", "joints", "local_rot_mats", "global_rot_mats"):
            if key in data:
                arr = data[key]
                if arr.ndim >= 3:
                    return arr.shape[-3] if arr.ndim == 4 else arr.shape[-2]
        return 30

    @staticmethod
    def _detect_frame_count(data) -> int:
        for key in ("posed_joints", "joints", "local_rot_mats", "root_positions", "trans"):
            if key in data:
                arr = data[key]
                if arr.ndim >= 2:
                    idx = -2 if arr.ndim >= 3 else -1
                    # Skip batch dim if present
                    return arr.shape[0] if arr.ndim <= 2 else arr.shape[0]
        return 1

    @staticmethod
    def _auto_skeleton(n_joints: int):
        """Auto-detect skeleton by joint count, or create generic names."""
        # Try to use kimodo's built-in skeleton definitions
        try:
            from kimodo.skeleton.registry import build_skeleton
            skel = build_skeleton(n_joints)
            jn = list(skel.bone_order_names)
            jp = skel.joint_parents.cpu().tolist()
            nj = skel.neutral_joints.cpu().numpy() if hasattr(skel, 'neutral_joints') else None
            return jn, jp, nj, skel.name
        except Exception:
            pass

        # Fallback: generic linear hierarchy
        jn = [f"joint_{i}" for i in range(n_joints)]
        jp = [-1] + [i - 1 for i in range(1, n_joints)]
        return jn, jp, None, f"unknown_{n_joints}"


# ---------------------------------------------------------------------------
# Node: Motion Path (curve-controlled waypoints → constraints JSON)
# ---------------------------------------------------------------------------

class Kimodo_MotionPath:
    """Generate root2d constraints from a 3D path defined by control points.

    Takes 3D control points (x, y, z) in motion space, distributes evenly-spaced
    waypoints along the arc-length of the control polygon, computes heading
    (direction of travel) per waypoint, and writes the result as a Kimodo
    constraints JSON file that can be passed to the Sampler node.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "points": ("STRING", {
                    "default": "0.0, 0.0, 0.0\n2.0, 1.0, 2.0\n4.0, 0.0, 4.0",
                    "multiline": True,
                    "tooltip": (
                        "Control points: one 'x, y, z' per line. "
                        "Waypoints are evenly distributed along the path."
                    ),
                }),
                "num_waypoints": ("INT", {
                    "default": 8, "min": 2, "max": 30, "step": 1,
                    "tooltip": "Number of evenly-spaced waypoints along the path.",
                }),
                "start_frame": ("INT", {
                    "default": 0, "min": 0, "max": 100000,
                    "tooltip": "First Kimodo frame index for the constraint.",
                }),
                "end_frame": ("INT", {
                    "default": 90, "min": 1, "max": 100000,
                    "tooltip": "Last Kimodo frame index for the constraint.",
                }),
            },
            "optional": {
                "auto_canonicalize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Offset X and Z so the first waypoint is at (0,0). "
                        "Y (height) is preserved as-is. Keeps motion centered at origin."
                    ),
                }),
                "compute_heading": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Auto-compute heading from path direction. "
                        "When off, uses fixed_heading."
                    ),
                }),
                "fixed_heading": ("FLOAT", {
                    "default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0,
                    "tooltip": (
                        "Fixed heading angle in degrees (0 = +Z forward). "
                        "Only used when compute_heading is off."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("constraints_json",)
    FUNCTION = "build"
    CATEGORY = "Kimodo/Constraints"

    def build(self, points, num_waypoints=8, start_frame=0, end_frame=90,
              auto_canonicalize=True, compute_heading=True, fixed_heading=0.0):
        # 1. Parse control points (support 2D and 3D)
        ctrl = self._parse_points(points)
        if len(ctrl) < 2:
            raise ValueError("Need at least 2 control points")

        # Normalize to 3D: ensure each point has exactly 3 components
        ctrl = [p if len(p) >= 3 else [p[0], 0.0, p[1]] for p in ctrl]

        # 2. 3D arc-length parameterization of the control polygon
        arc = [0.0]
        for i in range(1, len(ctrl)):
            dx = ctrl[i][0] - ctrl[i - 1][0]
            dy = ctrl[i][1] - ctrl[i - 1][1]
            dz = ctrl[i][2] - ctrl[i - 1][2]
            arc.append(arc[-1] + math.sqrt(dx * dx + dy * dy + dz * dz))
        total = arc[-1]

        # 3. Evenly-spaced waypoints along the arc (3D interpolation)
        if total < 1e-8:
            wps = [ctrl[0]] * num_waypoints
        else:
            wps = []
            for i in range(num_waypoints):
                target = i / (num_waypoints - 1) * total if num_waypoints > 1 else 0.0
                seg = 0
                while seg < len(arc) - 2 and arc[seg + 1] < target:
                    seg += 1
                seg_len = arc[seg + 1] - arc[seg]
                frac = (target - arc[seg]) / seg_len if seg_len > 0 else 0.0
                x = ctrl[seg][0] + frac * (ctrl[seg + 1][0] - ctrl[seg][0])
                y = ctrl[seg][1] + frac * (ctrl[seg + 1][1] - ctrl[seg][1])
                z = ctrl[seg][2] + frac * (ctrl[seg + 1][2] - ctrl[seg][2])
                wps.append([x, y, z])

        # 4. Headings (direction of travel, computed from XZ components)
        last_angle = math.radians(fixed_heading)
        headings = []
        for i in range(num_waypoints):
            if compute_heading:
                if i + 1 < num_waypoints:
                    dx = wps[i + 1][0] - wps[i][0]
                    dz = wps[i + 1][2] - wps[i][2]
                elif i > 0:
                    dx = wps[i][0] - wps[i - 1][0]
                    dz = wps[i][2] - wps[i - 1][2]
                else:
                    dx, dz = 0.0, 0.0
                if abs(dx) > 1e-8 or abs(dz) > 1e-8:
                    last_angle = math.atan2(dx, dz)
                headings.append(last_angle)
            else:
                headings.append(math.radians(fixed_heading))

        # 5. Canonicalize (subtract first waypoint XZ)
        ox = wps[0][0] if auto_canonicalize else 0.0
        oz = wps[0][2] if auto_canonicalize else 0.0

        # 6. Frame mapping: evenly distributed across [start_frame, end_frame]
        total_frames = end_frame - start_frame
        frame_indices = []
        smooth_root_2d = []
        global_root_heading = []
        for i in range(num_waypoints):
            frac = i / (num_waypoints - 1) if num_waypoints > 1 else 0.0
            kf = start_frame + round(frac * total_frames)
            frame_indices.append(kf)
            smooth_root_2d.append([wps[i][0] - ox, wps[i][1], wps[i][2] - oz])
            global_root_heading.append([math.cos(headings[i]), math.sin(headings[i])])

        # 7. Build constraint dict
        constraints = [{
            "type": "root2d",
            "frame_indices": frame_indices,
            "smooth_root_2d": smooth_root_2d,
            "global_root_heading": global_root_heading,
        }]

        # 8. Write to temp JSON file so the Sampler can load it via
        #    load_constraints_lst(filepath, skeleton)
        import json, tempfile
        fd, json_path = tempfile.mkstemp(suffix=".json", prefix="kimodo_path_")
        os.close(fd)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(constraints, f, indent=2)

        print(f"[Kimodo_MotionPath] {num_waypoints} waypoints, "
              f"frames {start_frame}–{end_frame} → {json_path}", flush=True)
        return (json_path,)

    @staticmethod
    def _parse_points(text: str) -> list[list[float]]:
        """Parse control points from text input.

        Accepts one 'x, y, z' per line (3D) or 'x, z' per line (2D),
        or comma-separated x1,y1,z1,x2,y2,z2,...
        """
        text = text.strip()
        if not text:
            return []

        points = []
        for line in text.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.replace(",", " ").split() if p.strip()]
            if len(parts) >= 3:
                try:
                    points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    continue
            elif len(parts) >= 2:
                try:
                    points.append([float(parts[0]), float(parts[1])])
                except ValueError:
                    continue

        # Fallback: comma-separated flat list
        if not points and "," in text:
            flat = [p.strip() for p in text.replace("\n", ",").split(",") if p.strip()]
            if len(flat) >= 6 and len(flat) % 3 == 0:
                for i in range(0, len(flat), 3):
                    try:
                        points.append([float(flat[i]), float(flat[i + 1]), float(flat[i + 2])])
                    except ValueError:
                        break
            if not points and len(flat) >= 4 and len(flat) % 2 == 0:
                for i in range(0, len(flat), 2):
                    try:
                        points.append([float(flat[i]), 0.0, float(flat[i + 1])])
                    except ValueError:
                        break
        return points


# ---------------------------------------------------------------------------
# Node: Curve → Points (visual curve editor)
# ---------------------------------------------------------------------------

class Kimodo_CurveToPoints:
    """3D curve editor — place control points to define a 3D motion path.

    Uses a three.js 3D viewport for interactive editing.
    Outputs a point string (one ``x, y, z`` per line) that connects directly
    to ``Kimodo_MotionPath`` for waypoint generation.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "num_samples": ("INT", {
                    "default": 16, "min": 0, "max": 512, "step": 1,
                    "tooltip": (
                        "Number of evenly-sampled points along the curve. "
                        "0 = output raw control points only."
                    ),
                }),
            },
            "hidden": {
                "curve_json": ("STRING", {"default": "[]"}),
                "node_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("points",)
    FUNCTION = "sample"
    CATEGORY = "Kimodo/Constraints"

    def sample(self, num_samples, curve_json="[]", node_id=None):
        import json
        try:
            ctrl = json.loads(curve_json)
        except (json.JSONDecodeError, TypeError):
            ctrl = []

        if not isinstance(ctrl, list) or len(ctrl) < 2:
            ctrl = [
                {"x": 0.0, "y": 0.0, "z": 0.0},
                {"x": 2.0, "y": 2.0, "z": 2.0},
                {"x": 4.0, "y": 0.0, "z": 4.0},
            ]

        # Parse 3D points; handle backward compat (old format: x=param, y=Z)
        pts = []
        for p in ctrl:
            x = p.get("x", 0.0)
            if "z" in p:
                pts.append([x, p.get("y", 0.0), p["z"]])
            else:
                # Old 2D format: x=param, y=Z → map to x,y=0,z
                pts.append([x, 0.0, p.get("y", 0.0)])

        if num_samples == 0:
            lines = "\n".join(f"{p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f}" for p in pts)
            return (lines,)

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        zs = [p[2] for p in pts]

        # 3D chord-length parameterisation
        chords = [0.0]
        for i in range(1, len(pts)):
            dx = pts[i][0] - pts[i - 1][0]
            dy = pts[i][1] - pts[i - 1][1]
            dz = pts[i][2] - pts[i - 1][2]
            chords.append(chords[-1] + math.sqrt(dx * dx + dy * dy + dz * dz))
        total = chords[-1]

        if total < 1e-8:
            lines = "\n".join(
                f"{pts[0][0]:.6f}, {pts[0][1]:.6f}, {pts[0][2]:.6f}"
                for _ in range(num_samples)
            )
            return (lines,)

        # Normalise to [0, 1]
        t = [c / total for c in chords]

        # Deduplicate t so PchipInterpolator sees strictly increasing values
        # (overlapping control points produce duplicate t values, which scipy rejects)
        uniq_pts = []
        uniq_t = []
        for ti, pt in zip(t, pts):
            if not uniq_t or ti > uniq_t[-1] + 1e-12:
                uniq_t.append(ti)
                uniq_pts.append(pt)
        if len(uniq_t) < 2:
            # All points collapsed — output repeated first point
            lines = "\n".join(
                f"{pts[0][0]:.6f}, {pts[0][1]:.6f}, {pts[0][2]:.6f}"
                for _ in range(num_samples)
            )
            return (lines,)
        uniq_t[-1] = 1.0  # guarantee the endpoint

        xs = [p[0] for p in uniq_pts]
        ys = [p[1] for p in uniq_pts]
        zs = [p[2] for p in uniq_pts]

        t_samples = [i / (num_samples - 1) for i in range(num_samples)]

        from scipy.interpolate import PchipInterpolator
        fx = PchipInterpolator(uniq_t, xs)
        fy = PchipInterpolator(uniq_t, ys)
        fz = PchipInterpolator(uniq_t, zs)

        out_xs = fx(t_samples)
        out_ys = fy(t_samples)
        out_zs = fz(t_samples)

        lines = "\n".join(
            f"{x:.6f}, {y:.6f}, {z:.6f}"
            for x, y, z in zip(out_xs, out_ys, out_zs)
        )
        return (lines,)


# ---------------------------------------------------------------------------
# Node mappings
# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    # Loaders
    "Kimodo_LoadModel": Kimodo_LoadModel,
    "Kimodo_LoadMotion": Kimodo_LoadMotion,
    # Configuration
    "Kimodo_Config": Kimodo_Config,
    # Conditioning
    "Kimodo_TextEncode": Kimodo_TextEncode,
    # Sampling
    "Kimodo_Sampler": Kimodo_Sampler,
    # Post-processing
    "Kimodo_PostProcess": Kimodo_PostProcess,
    # Preview
    "Kimodo_Preview": Kimodo_Preview,
    "Kimodo_Preview3D": Kimodo_Preview3D,
    # Export
    "Kimodo_SaveNPZ": Kimodo_SaveNPZ,
    "Kimodo_ExportBVH": Kimodo_ExportBVH,
    "Kimodo_ExportFBX": Kimodo_ExportFBX,
    # Skeleton operations
    "Kimodo_SkeletonInfo": Kimodo_SkeletonInfo,
    "Kimodo_Retarget": Kimodo_Retarget,
    "Kimodo_SelectBones": Kimodo_SelectBones,
    # Constraints
    "Kimodo_MotionPath": Kimodo_MotionPath,
    "Kimodo_CurveToPoints": Kimodo_CurveToPoints,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Kimodo_LoadModel": "Kimodo Load Model",
    "Kimodo_LoadMotion": "Kimodo Load Motion",
    "Kimodo_Config": "Kimodo Configuration",
    "Kimodo_TextEncode": "Kimodo Text Encode",
    "Kimodo_Sampler": "Kimodo Sampler",
    "Kimodo_PostProcess": "Kimodo Post Process",
    "Kimodo_Preview": "Kimodo Preview (2D)",
    "Kimodo_Preview3D": "Kimodo Preview 3D",
    "Kimodo_SaveNPZ": "Kimodo Save NPZ",
    "Kimodo_ExportBVH": "Kimodo Export BVH",
    "Kimodo_ExportFBX": "Kimodo Export FBX (Mixamo)",
    "Kimodo_SkeletonInfo": "Kimodo Skeleton Info",
    "Kimodo_Retarget": "Kimodo Retarget",
    "Kimodo_SelectBones": "Kimodo Select Bones",
    "Kimodo_MotionPath": "Kimodo Motion Path",
    "Kimodo_CurveToPoints": "Kimodo Curve → Points",
}
