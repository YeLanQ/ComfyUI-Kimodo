# ComfyUI-Kimodo

A ComfyUI plugin that wraps [Kimodo](https://github.com/nv-tlabs/kimodo) — NVIDIA's kinematic motion diffusion model for generating high-quality 3D human and humanoid robot motions from text prompts with optional kinematic constraints.

[中文说明](README_CN.md)

## Features

- **Text-to-Motion Generation** — Describe a motion in natural language, get 3D joint positions and rotations
- **Multiple Skeleton Types** — SOMA human body, SMPLX, and Unitree G1 humanoid robot
- **Kinematic Constraints** — Optional JSON constraints for pose keyframes, end-effector positions, 2D paths
- **Multi-Prompt Segments** — Chain multiple motion descriptions with smooth transitions
- **Multiple Samples** — Generate batch of motion variations from the same prompt
- **Motion Composition** — Append or overwrite motions to build longer sequences
- **NPZ Export** — Save motion data (joint positions, rotations, foot contacts, trajectories)
- **BVH Export** — Export to BVH format for animation software (SOMA skeletons)
- **BVH/NPZ Import** — Load existing motion files for editing or retargeting
- **FBX Export (Mixamo)** — Retarget motion onto Mixamo-rigged FBX characters and export animated FBX
- **Skeleton Operations** — Inspect hierarchy, retarget between skeletons, select bone subsets
- **Interactive Curve Editor** — 3D web-based path editor for motion trajectory control
- **2D Preview** — Skeleton stick-figure visualization as ComfyUI IMAGE output
- **3D Preview** — Three.js interactive skeleton viewer in browser
- **HuggingFace Auto-Download** — Models download automatically on first use (~17GB VRAM)

## Nodes

### Loaders

| Node | Category | Description |
|------|----------|-------------|
| **Kimodo Load Model** | Loaders | Load a Kimodo model variant (auto-downloads from HuggingFace) |
| **Kimodo Load Motion** | Loaders | Load BVH or NPZ motion files for editing, retargeting, or export |

### Configuration

| Node | Description |
|------|-------------|
| **Kimodo Configuration** | Set text encoder mode (local / API / auto), base model, and adapter paths |

### Conditioning

| Node | Description |
|------|-------------|
| **Kimodo Text Encode** | Encode text prompt → reusable conditioning (swap seeds without re-encoding) |

### Sampling

| Node | Description |
|------|-------------|
| **Kimodo Sampler** | Diffusion sampling with conditioning + optional constraints → motion. Supports composition (append/overwrite) with existing motions |

### Post-Processing

| Node | Description |
|------|-------------|
| **Kimodo Post Process** | Foot-skate cleanup with configurable root margin (optional, requires motion_correction module) |

### Preview & Export

| Node | Description |
|------|-------------|
| **Kimodo Preview (2D)** | Render 2D skeleton stick-figure for a specific frame |
| **Kimodo Preview 3D** | Interactive 3D skeleton visualization in browser |
| **Kimodo Save NPZ** | Save motion data as NPZ files |
| **Kimodo Export BVH** | Export motion to BVH format (SOMA skeletons only) |
| **Kimodo Export FBX (Mixamo)** | Retarget and export motion to a Mixamo-rigged FBX character |

### Skeleton Operations

| Node | Description |
|------|-------------|
| **Kimodo Skeleton Info** | Inspect joint names, parent hierarchy, and rest-pose positions |
| **Kimodo Retarget** | Remap/reorder joints via explicit bone mapping or auto name matching |
| **Kimodo Select Bones** | Keep only specified bones from motion data |

### Constraints

| Node | Description |
|------|-------------|
| **Kimodo Motion Path** | Generate root2d constraints from 3D control points with even waypoint distribution |
| **Kimodo Curve → Points** | Interactive 3D curve editor (web UI) that outputs control points for Motion Path |

## Installation

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jtydhr88/ComfyUI-Kimodo.git
```

Install dependencies:

```bash
cd ComfyUI-Kimodo
pip install -r requirements.txt
```

The kimodo package itself will be auto-installed on first launch if needed.

Restart ComfyUI. The **Kimodo** nodes will appear under the `Kimodo` category.

### Models

Models download automatically from HuggingFace on first use:

| Model | Skeleton | Dataset | Description |
|-------|----------|---------|-------------|
| Kimodo-SOMA-RP-v1 | SOMA (30 joints) | Rigplay (700h) | Human body, recommended |
| Kimodo-SMPLX-RP-v1 | SMPLX (22 joints) | Rigplay (700h) | SMPLX human body |
| Kimodo-G1-RP-v1 | G1 (34 joints) | Rigplay (700h) | Unitree G1 robot |
| Kimodo-SOMA-SEED-v1 | SOMA | SEED (288h) | Human body, SEED dataset |
| Kimodo-G1-SEED-v1 | G1 | SEED (288h) | G1 robot, SEED dataset |

### Manual Model Download

Kimodo's text encoder uses Meta Llama 3 8B, which is a **gated model** on HuggingFace. You need to:

1. Visit https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct and request access
2. Create a token at https://huggingface.co/settings/tokens
3. Log in and download all required models:

```bash
# Log in to HuggingFace
huggingface-cli login

# Text encoder: Llama 3 base model (gated, requires access approval)
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct

# Text encoder: LLM2Vec adapters
huggingface-cli download McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp
huggingface-cli download McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised

# Kimodo model (pick the one you want to use)
huggingface-cli download nvidia/Kimodo-SOMA-RP-v1
```

### Motion Correction (Optional)

The `motion_correction` C++ module provides foot-skate cleanup post-processing. You have two options:

**Option A: Use prebuilt binary (Windows + Python 3.11 only)**

```bash
# Copy the prebuilt files into your Python environment
cp -r prebuilt/win_amd64_cp311 <your-python-env>/Lib/site-packages/motion_correction
```

Or add the `prebuilt/win_amd64_cp311` directory to your Python path.

**Option B: Build from source (any platform)**

Requires CMake 3.15+ and a C++17 compiler (MSVC / GCC / Clang).

```bash
cd kimodo/MotionCorrection
pip install -e .
```

Verify: `python -c "import motion_correction; print('OK')"`

> Without this module, set `post_processing = False` in the Generate node. The motion will still work but may have foot-sliding artifacts.

### FBX Export (Optional)

To use the **Kimodo Export FBX (Mixamo)** node, install the FBX SDK Python bindings:

```bash
pip install fbxsdkpy --extra-index-url https://gitlab.inria.fr/api/v4/projects/18692/packages/pypi/simple
```

You also need a Mixamo-rigged FBX character file. Download one from [Mixamo](https://www.mixamo.com/) (select "Without Skin" or "T-Pose" for best results).

### Hardware Requirements

- CUDA GPU, ~17GB VRAM (text encoder + diffusion model)
- Tested on: RTX 3090, RTX 4090, A100

## Usage

### Modular Workflow (Recommended)

```
Load Model → Text Encode → Sampler → Post Process → Export/Preview
                                ↑
                         (constraints_json)
```

1. Add **Kimodo Load Model** — select a model variant
2. Add **Kimodo Text Encode** — enter text prompt (reusable across different seeds)
3. Add **Kimodo Sampler** — set duration, seed, diffusion steps
4. Add **Kimodo Post Process** — optional foot-skate cleanup
5. Add preview/export nodes

### Parameters

#### Sampler

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prompt` | — | Text description of the motion (via Text Encode node) |
| `duration` | 5.0 | Duration in seconds per segment |
| `seed` | 42 | Random seed for reproducibility |
| `num_samples` | 1 | Number of motion variations to generate |
| `diffusion_steps` | 100 | Denoising steps (more = better quality, slower) |
| `constraints_json` | — | Optional path to kinematic constraints JSON |
| `composition_mode` | new | `new` = fresh generation; `append` = concatenate after existing; `overwrite` = replace frames |
| `overwrite_frame` | 0 | Frame index for overwrite mode |

#### FBX Export

| Parameter | Default | Description |
|-----------|---------|-------------|
| `yaw_offset` | 0.0 | Rotate character around Y-axis (degrees) |
| `scale` | 0.0 | Force scale multiplier (0 = auto height-based scaling) |

#### Post Process

| Parameter | Default | Description |
|-----------|---------|-------------|
| `root_margin` | 0.04 | Root correction margin in meters |

### Multi-Prompt

Separate motion segments with periods in the prompt:

```
A person walks forward. They stop and wave hello. They turn around and sit down.
```

Each segment gets the specified duration.

### Motion Composition

Connect an existing motion to the Sampler's `existing_motion` input and set `composition_mode` to `append` or `overwrite`:

- **Append**: new motion frames are concatenated after the last frame of existing motion
- **Overwrite**: new motion replaces frames in existing motion starting at `overwrite_frame`

### Skeleton Operations

1. **Kimodo Skeleton Info** — connect a motion to inspect joint hierarchy, names, and rest pose
2. **Kimodo Retarget** — reorder/remap bones via text mapping (e.g. `left_hand -> hand_l`), auto name matching, or identity pass-through
3. **Kimodo Select Bones** — keep only specified bones (comma/newline separated names)

### Constraints: Motion Path

1. Add **Kimodo Curve → Points** for interactive 3D path editing
2. Connect its output to **Kimodo Motion Path** to generate evenly-spaced waypoints
3. Connect Motion Path's `constraints_json` to the Sampler

### Output Format

The NPZ output contains:
- `posed_joints` — Joint positions `[T, J, 3]`
- `global_rot_mats` — Joint rotation matrices `[T, J, 3, 3]`
- `root_positions` — Root trajectory `[T, 3]`
- `foot_contacts` — Foot contact labels `[T, 4]`
- `global_root_heading` — Root heading angle `[T]`

## Workflows

Example workflow JSON files are in the `workflows/` directory:

- `kimodo-basic.json` — Basic text-to-motion generation
- `kimodo-fbx.json` — Generation with FBX export
- `kimodo-with-post-process.json` — Generation with foot-skate correction

## Credits

This plugin wraps [Kimodo](https://github.com/nv-tlabs/kimodo) from NVIDIA Toronto AI Lab.

## Connect

- **GitHub Issues** — Bug reports, feature requests, and discussions: https://github.com/jtydhr88/ComfyUI-Kimodo/issues
- **Original Kimodo** — NVIDIA's motion diffusion model: https://github.com/nv-tlabs/kimodo
- **NVIDIA Toronto AI Lab** — Research lab behind Kimodo: https://www.torontoai-lab.com
- **HuggingFace Models** — Pre-trained Kimodo models: https://huggingface.co/nvidia

## License

[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)
