# AGENTS.md

## 项目概述

基于 W2V-BERT 2.0 的说话人验证 (Speaker Verification) 研究项目，包含 LoRA 微调、多阶段训练和知识蒸馏引导的结构化剪枝。对应论文：[arXiv:2510.04213](https://arxiv.org/abs/2510.04213)。

纯研究代码，无 CI/CD、无测试套件、无 linter 配置。许可证为 CC BY-NC-SA 4.0。

## 仓库结构

```
deeplab/                  # 核心框架库（被所有训练脚本通过 sys.path.append 引用）
  core/trainer.py         # 基础训练器（DDP、AMP、梯度裁剪、检查点保存）
  core/scheduler.py       # 自定义学习率调度器（Warmup+StepDecay、WarmupCosine 等）
  dataio/audio.py         # 音频处理（截断、加噪、混响、速度扰动、编解码增强）
  metric/eer.py           # 说话人验证指标：EER 和 minDCF 计算
  utils/                  # 文件I/O、语料加载、SCP/trial 格式处理
  pretrained/audio2vector/
    api.py                # AudioEncoder 核心类，封装模型加载、LoRA配置、冻结/解冻
    forward_impl.py       # W2V-BERT 和 Whisper 的自定义 forward 实现
    ckpts/facebook/w2v-bert-2.0/  # 模型配置文件（不含权重，需手动下载）
    module/transformers/  # ★ 魔改版 HuggingFace transformers（支持 Hard Concrete 剪枝门）

recipes/DeepASV/          # ★ 所有命令的执行目录（含有 train.py 等入口文件）
  train.py                # 主训练脚本（继承 Trainer，实现 speaker model 训练）
  train_prune_s1.py       # 剪枝阶段1：知识蒸馏引导的结构化剪枝
  train_prune_s2.py       # 剪枝阶段2：进一步蒸馏
  run.sh / run_prune.sh   # 示例命令（全部被注释，仅供参考）
  conf/w2v-bert/          # 训练 YAML 配置：s1(冻结编码器+LoRA), s2(联合微调), s3(LMFT)
  conf/prune/             # 剪枝 YAML 配置：dis_prune_s1, dis_prune_s2, s1-s3(剪枝后微调)
  local/
    spk_model.py          # 说话人模型：Audio2Vec_based_Adapter(主力), Audio2Vec_based_Prune
    spk_classifier.py     # ArcFace 分类器
    dataset.py            # Train_Dataset / Valid_Dataset（VoxCeleb 格式）
    sampler.py            # WavBatchSampler（按时长动态分批）
    data_pipe.py          # 在线数据增强管线
    modules/              # ECAPA-TDNN、ASP 等池化层
  utils/
    lora_merge.py         # 合并 LoRA 权重到基础模型
    apply_prune_s1.py     # 执行结构化剪枝并生成新的模型配置
    apply_prune_s2.py     # 提取蒸馏后的学生模型权重
    get_embd_w2v.py       # 提取说话人嵌入并计算 EER
  analysis/               # 大规模嵌入提取流水线（独立于训练，不依赖 CWD）
    step1_run_embedding_extraction.sh
    step2_run_compute_speaker_embeddings.sh
    step3_run_compute_speaker_similarities.sh
```

## 环境搭建

```bash
conda create -y -n asv python=3.9
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip uninstall transformers          # ★ 必须卸载系统 transformers
conda install -c conda-forge sox   # torchaudio sox_effects 依赖
```

**关键：必须卸载 pip 安装的 transformers**。本仓库使用魔改版 transformers（位于 `deeplab/pretrained/audio2vector/module/transformers/src`），通过 `sys.path.append` 方式引入。官方 transformers 共存时，`peft` 的 `from transformers import ...` 会优先命中官方版，而官方版 `Wav2Vec2BertConfig` 没有 Hard Concrete 剪枝门字段，导致类型检查失败。

### 预训练权重下载

从 HuggingFace 下载 `model.safetensors` 放入：
```
deeplab/pretrained/audio2vector/ckpts/facebook/w2v-bert-2.0/model.safetensors
```

## 训练流程（严格有序）

**所有命令必须在 `recipes/DeepASV/` 目录下执行**，因为：
1. 训练脚本通过 `sys.path.append('../..')` 引用 `deeplab/` 包
2. YAML 配置中的路径相对于 `recipes/DeepASV/`
3. `api.py:5` 中的 `sys.path.append('./module/transformers/src')` 依赖调用脚本预先通过绝对路径添加了魔改版 transformers 到 sys.path

### 完整训练（3阶段）

```bash
# Stage 1：冻结编码器 + LoRA 适配器训练
OMP_NUM_THREADS="16" CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
torchrun --nnodes 1 --nproc_per_node=8 --master_port=12885 train.py \
  --tag vox2_ --is_distributed true --yaml conf/w2v-bert/s1.yaml

# ★ 阶段间操作：合并 LoRA 参数（在 recipes/DeepASV/utils/ 下执行）
cd utils && python3 lora_merge.py && cd ..

# Stage 2：联合微调（编码器解冻）
torchrun --nnodes 1 --nproc_per_node=8 --master_port=12885 train.py \
  --tag vox2_ --is_distributed true --yaml conf/w2v-bert/s2.yaml \
  --pretrain /path/stage1/merge_lora.pth

# Stage 3：Large Margin Fine-Tuning (LMFT)
torchrun --nnodes 1 --nproc_per_node=8 --master_port=12885 train.py \
  --tag vox2_ --is_distributed true --yaml conf/w2v-bert/s3.yaml \
  --pretrain /path/stage2/best_ckpt.pth
```

**各阶段配置文件对照：**

| 阶段 | YAML | encoder_config | frozen_encoder | peft_config |
|------|------|----------------|----------------|-------------|
| s1 | `conf/w2v-bert/s1.yaml` | `config_tea.json` | true | LoRA (r=64) |
| s2 | `conf/w2v-bert/s2.yaml` | `config_prune_tea.json` | false | null |
| s3 | `conf/w2v-bert/s3.yaml` | `config_prune_tea.json` | false | null |

### 剪枝流程（严格有序，每步不可跳过）

```
dis_prune_s1 → apply_prune_s1.py → dis_prune_s2 → apply_prune_s2.py → prune/s1 → prune/s2 → prune/s3
```

每个 `apply_prune_*.py` 脚本在 `recipes/DeepASV/utils/` 目录下执行，会修改模型权重和配置文件。

```bash
# 剪枝 Stage 1：需先编辑 conf/prune/dis_prune_s1.yaml 第 59 行的教师路径
OMP_NUM_THREADS="12" CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
torchrun --nnodes 1 --nproc_per_node=8 --master_port=12885 train_prune_s1.py \
  --tag prune_ --is_distributed true --yaml conf/prune/dis_prune_s1.yaml

# ★ 剪枝间操作（在 utils/ 下执行）
cd utils && python3 apply_prune_s1.py && cd ..

# 剪枝 Stage 2：
torchrun --nnodes 1 --nproc_per_node=8 --master_port=12885 train_prune_s2.py \
  --tag prune_ --is_distributed true --yaml conf/prune/dis_prune_s2.yaml \
  --pretrain /path/prune_s1/prune_update.pth

# ★ 剪枝间操作（在 utils/ 下执行）
cd utils && python3 apply_prune_s2.py && cd ..

# 剪枝后微调（3 阶段，使用 conf/prune/s1.yaml → s2.yaml → s3.yaml）
torchrun --nnodes 1 --nproc_per_node=8 --master_port=12886 train.py \
  --tag prune_ft_ --is_distributed true --yaml conf/prune/s1.yaml \
  --pretrain /path/prune_s2/prune_dis.pth
```

### 推理/测试

```bash
cd recipes/DeepASV/utils
python3 get_embd_w2v.py
```

**注意：`get_embd_w2v.py` 要求 checkpoint 同目录下存在 `train.yaml`**（脚本会通过该 yaml 重建模型结构）。

大规模嵌入提取流水线在 `recipes/DeepASV/analysis/` 下，使用 `__file__` 计算绝对路径，不依赖 CWD。脚本顺序为 step1 → step2 → step3。

## 关键技术细节

### sys.path 依赖

所有训练脚本通过 `sys.path.append('../..')` 和 `sys.path.append('../../deeplab/pretrained/audio2vector/module/transformers/src')` 引用 `deeplab/` 包和魔改版 transformers。**必须从 `recipes/DeepASV/` 目录启动**，否则导入会失败。

注意：`api.py:5` 内部还有 `sys.path.append('./module/transformers/src')`，这是一个相对于 CWD 的路径，实际由调用方脚本预先设置的正确 sys.path 覆盖，不要依赖这行。

`utils/` 下的脚本（`lora_merge.py`、`get_embd_w2v.py` 等）使用 `sys.path.append('..')` 和 `sys.path.append('../../..')`，需要从 `utils/` 目录执行。

### YAML 配置格式

使用 `hyperpyyaml`（SpeechBrain 风格），不是标准 YAML：
- `!new:local.spk_model.Audio2Vec_based_Adapter` — 实例化类
- `!name:torch.optim.AdamW` — 延迟实例化（传 callable）
- `!apply:deeplab.utils.misc.set_random_seed` — 立即调用函数
- `!ref <var>` — 引用同文件中的变量

### 模型配置文件区别 ★ 容易踩坑

`deeplab/pretrained/audio2vector/ckpts/facebook/w2v-bert-2.0/` 下有多个配置：

| 配置文件 | prune 字段 | 用途 |
|---------|-----------|------|
| `config_tea.json` | `false` | **仅 Stage 1**，加载 safetensors 权重 |
| `config_prune_tea.json` | `false` | Stage 2/3 教师、剪枝教师 |
| `config_prune_stu.json` | `true` | 剪枝学生（训练时启用 Hard Concrete 门） |
| `config_prune_stu_0.8.json` | `false` | 剪枝后学生（由 `apply_prune_s1.py` 生成，初始不存在） |
| `config.json` | — | HuggingFace 原始配置，由 `config.py` 读取生成上述文件 |
| `config.py` | — | 生成 `config_prune_stu.json` 和 `config_prune_tea.json` 的脚本 |

**关键判断逻辑（`api.py:165`）：**
```python
if 'prune' not in self.model_config:
    # 加载 safetensors 预训练权重
```
即：配置**文件名**中不含字符串 `prune` → 加载 safetensors → 使用 `config_tea.json`。
文件名含 `prune` → 跳过 safetensors → 需通过 `--pretrain` 提供权重。

**易错场景：**
- **Stage 1 必须用 `config_tea.json`**，误用 `config_prune_tea.json` 会跳过 safetensors 初始化，导致从随机权重开始训练
- **lora_merge.py 必须用 `config_prune_tea.json`**：因为 LoRA 合并后模型结构与 `config_tea.json` 不兼容

### 需手动更新的检查点路径

`utils/` 下的脚本和 `conf/prune/dis_prune_s1.yaml` 中包含 `YOUR_*_DIR` 占位符，训练后需替换为实际目录名：

| 文件 | 占位符 | 说明 |
|------|-------|------|
| `lora_merge.py:31` | `YOUR_STAGE1_DIR` | Stage 1 检查点目录 |
| `dis_prune_s1.yaml:59` | `YOUR_STAGE2_DIR` | Stage 2 的 `best_ckpt.pth`（剪枝需要已训练的 Stage 2 模型作为教师） |
| `apply_prune_s1.py:10` | `YOUR_PRUNE_S1_DIR` | 剪枝 Stage 1 检查点目录 |
| `apply_prune_s2.py:10` | `YOUR_PRUNE_S2_DIR` | 剪枝 Stage 2 检查点目录 |
| `apply_prune_s2.py:37` | `YOUR_STAGE2_DIR` | 原始 Stage 2 的 `best_ckpt.pth` |
| `get_embd_w2v.py:15` | 硬编码 `../../../models/model_lmft_0.14.pth` | 可自行修改 |

### 检查点格式

```python
{
    'modules': {
        'spk_model': OrderedDict,  # 说话人模型 state_dict
        'classifier': OrderedDict  # ArcFace 分类器 state_dict
    },
    'epoch_idx': int
}
```

加载检查点时，`classifier` 权重支持大小不匹配的部分加载（`trainer.py:409-424`）：Stage 3 的 `out_features=5994` 小于 Stage 1/2 的 `17982`，加载时只取前 5994 个 speaker 的权重。

### 数据目录结构

所有硬编码绝对路径已改为基于仓库根目录的相对路径。数据需按以下结构放置：

```
data/
├── AudioData/
│   ├── musan/              # MUSAN 噪声语料
│   └── rirs_noise/         # RIR 混响数据
├── vox2dev/
│   └── dev.spk2utt         # VoxCeleb2 dev 集（tab 分隔：spk_id\tutt_path）
└── test_vox/
    └── vox1-o/
        ├── wav_copy.scp     # 测试音频 SCP（tab 分隔：utt_id\twav_path）
        └── trials           # trial 文件（空格分隔：label utt1 utt2）
```

- 训练数据：`.spk2utt` 文件（tab 分隔：`spk_id\tutt_path`）
- 验证数据：`.scp` 文件（tab 分隔：`utt_id\twav_path`）+ trial 文件（空格分隔：`label utt1 utt2`）
- 音频采样率：16000 Hz
- 噪声增强需要 MUSAN 和 RIRs 数据集

### DDP 兼容性处理

- `api.py:192`：加载 W2V-BERT 时会 `delattr(self.encoder, 'masked_spec_embed')`，避免 DDP 中未使用参数报错
- 训练默认使用 8 GPU + NCCL 后端
- 默认使用 bfloat16 混合精度训练
- `OMP_NUM_THREADS` 控制 CPU 端数据加载的线程数（stage1/stage2 用 16，prune 用 12）

### 魔改版 transformers

`deeplab/pretrained/audio2vector/module/transformers/` 是 HuggingFace transformers 的 fork，主要修改：
- `Wav2Vec2BertModel` 中添加了 Hard Concrete 剪枝门（`log_alpha` 参数）
- 支持按层级控制 attention head 数、FFN 维度、Conv 维度的结构化剪枝
- 模型配置中新增 `prune`、`intermediate_size_group`、`num_attention_heads_group`、`conv_group`、`use_feed_forward`、`use_attention` 等字段
- **不要**用 pip 安装的 transformers 替换或更新此目录

## 常见操作

| 操作 | 命令 / 位置 |
|------|------------|
| 计算模型 FLOPs | `python3 recipes/DeepASV/local/spk_model.py` |
| 查看训练日志 | `results/checkpoints/<tag>_<timestamp>/logs.json` |
| 查看训练配置 | `results/checkpoints/<tag>_<timestamp>/train.yaml` |
| 修改 speaker 数量 | YAML 中 `classifier.out_features`（需匹配数据集） |
| 修改 LoRA rank | YAML 中 `peft_config.r` 和 `lora_alpha` |
| 启用/禁用 wandb | YAML 中添加/删除 `wandb_cfgs` 字段 |
| 推理 EER 测试 | `cd utils && python3 get_embd_w2v.py`（需 checkpoint 同目录有 `train.yaml`） |
| 大规模嵌入提取 | `cd analysis && bash step1_run_embedding_extraction.sh` |
