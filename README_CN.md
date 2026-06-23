# ComfyUI-Kimodo

一个将 [Kimodo](https://github.com/nv-tlabs/kimodo) 集成到 ComfyUI 的插件 — NVIDIA 的运动学运动扩散模型，通过文本提示和可选的运动学约束生成高质量的 3D 人体和人形机器人动作。

[English](README.md)

## 功能特性

- **文本生成动作** — 用自然语言描述动作，获得 3D 关节位置和旋转
- **多种骨骼类型** — SOMA 人体、SMPLX 和 Unitree G1 人形机器人
- **运动学约束** — 可选的 JSON 约束：姿态关键帧、末端执行器位置、2D 路径
- **多段提示** — 用句号分隔多个动作描述，自动平滑过渡
- **多样本生成** — 从同一提示生成多个动作变体
- **动作合成** — 追加或覆盖动作以构建更长的序列
- **NPZ 导出** — 保存动作数据（关节位置、旋转、足部接触、轨迹）
- **BVH 导出** — 导出 BVH 格式用于动画软件（仅限 SOMA 骨骼）
- **BVH/NPZ 导入** — 加载现有动作文件进行编辑或重定向
- **FBX 导出 (Mixamo)** — 将动作重定向到 Mixamo 绑定的 FBX 角色并导出动画 FBX
- **骨骼操作** — 检查层级结构、骨骼重定向、选择骨骼子集
- **交互式曲线编辑器** — 基于 Web 的 3D 路径编辑器，用于运动轨迹控制
- **2D 预览** — 骨架简笔画可视化，作为 ComfyUI IMAGE 输出
- **3D 预览** — 浏览器中的 Three.js 交互式骨架查看器
- **HuggingFace 自动下载** — 首次使用时自动下载模型（约需 17GB 显存）

## 节点说明

### 加载器

| 节点 | 分类 | 说明 |
|------|------|------|
| **Kimodo Load Model** | 加载器 | 加载 Kimodo 模型变体（自动从 HuggingFace 下载） |
| **Kimodo Load Motion** | 加载器 | 加载 BVH 或 NPZ 动作文件进行编辑、重定向或导出 |

### 配置

| 节点 | 说明 |
|------|------|
| **Kimodo Configuration** | 设置文本编码器模式（本地 / API / 自动）、基础模型和适配器路径 |

### 条件

| 节点 | 说明 |
|------|------|
| **Kimodo Text Encode** | 编码文本提示 → 可复用的条件（换 seed 不用重新编码） |

### 采样

| 节点 | 说明 |
|------|------|
| **Kimodo Sampler** | 扩散采样 + 可选约束 → 动作数据。支持合成模式（追加/覆盖） |

### 后处理

| 节点 | 说明 |
|------|------|
| **Kimodo Post Process** | 脚滑修正，可配置根关节边距（可选，需要 motion_correction 模块） |

### 预览 & 导出

| 节点 | 说明 |
|------|------|
| **Kimodo Preview (2D)** | 渲染指定帧的 2D 骨架简笔画 |
| **Kimodo Preview 3D** | 浏览器中的交互式 3D 骨架可视化 |
| **Kimodo Save NPZ** | 保存动作数据为 NPZ 文件 |
| **Kimodo Export BVH** | 导出 BVH 格式（仅限 SOMA 骨骼） |
| **Kimodo Export FBX (Mixamo)** | 将动作重定向并导出到 Mixamo 绑定的 FBX 角色 |

### 骨骼操作

| 节点 | 说明 |
|------|------|
| **Kimodo Skeleton Info** | 检查关节名称、父级层级结构和 T-Pose 位置 |
| **Kimodo Retarget** | 通过显式骨骼映射或自动名称匹配重新映射/排序关节 |
| **Kimodo Select Bones** | 仅保留指定的骨骼 |

### 约束

| 节点 | 说明 |
|------|------|
| **Kimodo Motion Path** | 从 3D 控制点生成 root2d 约束，均匀分布路径点 |
| **Kimodo Curve → Points** | 交互式 3D 曲线编辑器（Web UI），输出控制点给 Motion Path |

## 安装

将此仓库克隆到 ComfyUI 的 `custom_nodes` 目录：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jtydhr88/ComfyUI-Kimodo.git
```

安装依赖：

```bash
cd ComfyUI-Kimodo
pip install -r requirements.txt
```

kimodo 包本身会在首次启动时自动安装（如果需要）。

重启 ComfyUI，**Kimodo** 节点将出现在 `Kimodo` 分类下。

### 模型

首次使用时自动从 HuggingFace 下载：

| 模型 | 骨骼 | 数据集 | 说明 |
|------|------|--------|------|
| Kimodo-SOMA-RP-v1 | SOMA (30 关节) | Rigplay (700h) | 人体，推荐使用 |
| Kimodo-SMPLX-RP-v1 | SMPLX (22 关节) | Rigplay (700h) | SMPLX 人体 |
| Kimodo-G1-RP-v1 | G1 (34 关节) | Rigplay (700h) | Unitree G1 机器人 |
| Kimodo-SOMA-SEED-v1 | SOMA | SEED (288h) | 人体，SEED 数据集 |
| Kimodo-G1-SEED-v1 | G1 | SEED (288h) | G1 机器人，SEED 数据集 |

### 手动下载模型

Kimodo 的文本编码器使用 Meta Llama 3 8B，这是 HuggingFace 上的**受限模型**。你需要：

1. 访问 https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct 申请访问权限
2. 在 https://huggingface.co/settings/tokens 创建 token
3. 登录并下载所有需要的模型：

```bash
# 登录 HuggingFace
huggingface-cli login

# 文本编码器：Llama 3 基础模型（受限，需要审批）
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct

# 文本编码器：LLM2Vec 适配器
huggingface-cli download McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp
huggingface-cli download McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised

# Kimodo 模型本身（选你要用的）
huggingface-cli download nvidia/Kimodo-SOMA-RP-v1
```

### Motion Correction 动作修正（可选）

`motion_correction` 是一个 C++ 模块，用于脚滑修正等后处理。两种安装方式：

**方式 A：使用预编译文件（仅限 Windows + Python 3.11）**

```bash
# 将预编译文件复制到 Python 环境
cp -r prebuilt/win_amd64_cp311 <你的Python环境>/Lib/site-packages/motion_correction
```

或者将 `prebuilt/win_amd64_cp311` 目录加入 Python 路径。

**方式 B：从源码编译（任意平台）**

需要 CMake 3.15+ 和 C++17 编译器（MSVC / GCC / Clang）。

```bash
cd kimodo/MotionCorrection
pip install -e .
```

验证：`python -c "import motion_correction; print('OK')"`

> 不安装此模块时，将 Sampler 节点的 `post_processing` 设为 `False` 即可。动作仍可正常生成，但可能有脚滑现象。

### FBX 导出（可选）

使用 **Kimodo Export FBX (Mixamo)** 节点需要安装 FBX SDK Python 绑定：

```bash
pip install fbxsdkpy --extra-index-url https://gitlab.inria.fr/api/v4/projects/18692/packages/pypi/simple
```

还需要一个 Mixamo 绑定的 FBX 角色文件，可从 [Mixamo](https://www.mixamo.com/) 下载（建议选 T-Pose 导出）。

### 硬件要求

- CUDA GPU，约需 17GB 显存（文本编码器 + 扩散模型）
- 测试硬件：RTX 3090、RTX 4090、A100

## 使用方法

### 模块化工作流（推荐）

```
Load Model → Text Encode → Sampler → Post Process → Export/Preview
                                ↑
                         (constraints_json)
```

1. 添加 **Kimodo Load Model** — 选择模型变体
2. 添加 **Kimodo Text Encode** — 输入文本提示（可在不同 seed 间复用）
3. 添加 **Kimodo Sampler** — 设置时长、seed、扩散步数
4. 添加 **Kimodo Post Process** — 可选的脚滑修正
5. 添加预览/导出节点

### 参数说明

#### Sampler

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `prompt` | — | 动作的文本描述（通过 Text Encode 节点） |
| `duration` | 5.0 | 每段时长（秒） |
| `seed` | 42 | 随机种子 |
| `num_samples` | 1 | 生成的动作变体数量 |
| `diffusion_steps` | 100 | 去噪步数（越多质量越好，速度越慢） |
| `constraints_json` | — | 可选的运动学约束 JSON 文件路径 |
| `composition_mode` | new | `new` = 全新生成；`append` = 追加到现有动作后；`overwrite` = 覆盖帧 |
| `overwrite_frame` | 0 | 覆盖模式下的起始帧索引 |

#### FBX 导出

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `yaw_offset` | 0.0 | 绕 Y 轴旋转角色（度） |
| `scale` | 0.0 | 强制缩放倍数（0 = 自动基于高度缩放） |

#### 后处理

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `root_margin` | 0.04 | 根关节修正边距（米） |

### 多段提示

在提示中用句号分隔不同的动作段落：

```
A person walks forward. They stop and wave hello. They turn around and sit down.
```

每段使用指定的时长。

### 动作合成

将现有动作连接到 Sampler 的 `existing_motion` 输入，设置 `composition_mode` 为 `append` 或 `overwrite`：

- **Append**：新动作帧追加到现有动作末尾
- **Overwrite**：在 `overwrite_frame` 处开始覆盖现有动作的帧

### 骨骼操作

1. **Kimodo Skeleton Info** — 连接动作数据，检查关节层级、名称和 T-Pose
2. **Kimodo Retarget** — 通过文本映射重排骨骼（如 `left_hand -> hand_l`），自动匹配或保持原样
3. **Kimodo Select Bones** — 仅保留指定的骨骼（逗号或换行分隔）

### 约束：运动路径

1. 添加 **Kimodo Curve → Points** 进行交互式 3D 路径编辑
2. 将其输出连接到 **Kimodo Motion Path** 生成均匀分布的路径点
3. 将 Motion Path 的 `constraints_json` 输出连接到 Sampler

### 输出格式

NPZ 输出包含：
- `posed_joints` — 关节位置 `[T, J, 3]`
- `global_rot_mats` — 关节旋转矩阵 `[T, J, 3, 3]`
- `root_positions` — 根轨迹 `[T, 3]`
- `foot_contacts` — 足部接触标签 `[T, 4]`
- `global_root_heading` — 根朝向角度 `[T]`

## 工作流

`workflows/` 目录中包含示例工作流 JSON 文件：

- `kimodo-basic.json` — 基础文本生成动作
- `kimodo-fbx.json` — 生成并导出 FBX
- `kimodo-with-post-process.json` — 生成并脚滑修正

## 致谢

本插件封装了 [NVIDIA Toronto AI Lab](https://github.com/nv-tlabs) 的 [Kimodo](https://github.com/nv-tlabs/kimodo) 项目。

## 联系

- **GitHub Issues** — 报告问题、功能请求和讨论：https://github.com/jtydhr88/ComfyUI-Kimodo/issues
- **Kimodo 原项目** — NVIDIA 运动扩散模型：https://github.com/nv-tlabs/kimodo
- **NVIDIA Toronto AI Lab** — Kimodo 背后的研究团队：https://www.torontoai-lab.com
- **HuggingFace 模型** — 预训练的 Kimodo 模型：https://huggingface.co/nvidia

## 许可证

[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)
