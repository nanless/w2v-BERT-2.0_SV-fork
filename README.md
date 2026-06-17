# W2V-BERT 2.0 说话人验证系统 — 完整技术文档

基于 W2V-BERT 2.0 的端到端说话人验证 (Speaker Verification) 研究项目。实现 LoRA 微调、多阶段渐进式训练、知识蒸馏引导的 Hard Concrete 结构化剪枝，以及大规模嵌入提取与相似度分析流水线。

> 📄 **论文**：[arXiv:2510.04213](https://arxiv.org/abs/2510.04213)
>
> 🏷️ **许可证**：CC BY-NC-SA 4.0

---

## 目录

1. [环境搭建](#1-环境搭建)
2. [仓库结构](#2-仓库结构)
3. [模型架构详解](#3-模型架构详解)
4. [预训练权重](#4-预训练权重)
5. [数据准备](#5-数据准备)
6. [完整训练流程](#6-完整训练流程)
7. [结构化剪枝流程](#7-结构化剪枝流程)
8. [推理与嵌入提取](#8-推理与嵌入提取)
9. [大规模相似度分析流水线](#9-大规模相似度分析流水线)
10. [关键技术与常见陷阱](#10-关键技术与常见陷阱)
11. [预训练模型下载](#11-预训练模型下载)
12. [常见操作速查](#12-常见操作速查)
13. [引用](#13-引用)

---

## 1. 环境搭建

### 1.1 Conda 环境

```bash
conda create -y -n asv python=3.9
conda activate asv
```

### 1.2 PyTorch

```bash
# CUDA 11.8 版本
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu118
```

### 1.3 依赖安装

```bash
pip install -r requirements.txt
```

`requirements.txt` 完整说明：

| 包名 | 版本要求 | 用途 |
|------|---------|------|
| `calflops` | — | 计算模型浮点运算量 (FLOPs) |
| `hyperpyyaml` | — | SpeechBrain 风格 YAML（支持 `!new:` / `!name:` / `!apply:` / `!ref`） |
| `librosa` | — | 备用音频特征提取 |
| `numpy` | — | 数值计算 |
| `pydub` | — | MP3/非标准格式编解码 |
| `scikit_learn` | — | `accuracy_score` |
| `scipy` | — | 信号卷积、DET 曲线计算 |
| `soundfile` | — | WAV/FLAC 音频 I/O |
| `tqdm` | — | 进度条 |
| `peft` | — | LoRA (`LoraConfig`, `get_peft_model`, `merge_and_unload`) |
| `wandb` | — | 可选训练可视化 |
| `importlib_metadata` | — | 包版本检查 |

### 1.4 ★ 关键操作：卸载官方 transformers

```bash
pip uninstall transformers
```

**不执行此步骤会导致无法运行。**

本项目使用的是魔改版 HuggingFace transformers（位于 `deeplab/pretrained/audio2vector/module/transformers/src`），通过 `sys.path.append` 的方式导入。官方 `transformers` 包和魔改版同时存在时，`peft` 的 `from transformers import ...` 会优先命中官方版本——而官方版 `Wav2Vec2BertConfig` 没有 Hard Concrete 剪枝门的 `prune` / `intermediate_size_group` 等字段，导致类型检查失败。

### 1.5 sox

```bash
conda install -c conda-forge sox
```

`torchaudio.sox_effects.apply_effects_file` 依赖系统 sox。

---

## 2. 仓库结构

```
w2v-BERT-2.0_SV-fork/
│
├── deeplab/                               # 核心框架库
│   ├── core/
│   │   ├── trainer.py                     # 基础训练器
│   │   └── scheduler.py                  # 三种调度器实现
│   ├── dataio/
│   │   └── audio.py                       # 7 种音频增强操作
│   ├── metric/
│   │   └── eer.py                         # EER / minDCF 计算
│   ├── utils/
│   │   ├── fileio.py                      # 音频加载、SCP/trial I/O、hyperpyyaml
│   │   ├── corpus.py                      # 训练语料/MUSAN/RIR 加载
│   │   └── misc.py                        # 种子、参数计数、时间格式化
│   └── pretrained/audio2vector/
│       ├── api.py                         # AudioEncoder 类
│       ├── forward_impl.py                # forward_w2v_bert / forward_whisper
│       ├── ckpts/facebook/w2v-bert-2.0/   # 配置 + safetensors 权重
│       └── module/transformers/           # ★ 魔改版 HuggingFace 源码
│
├── recipes/DeepASV/                       # ★ 工作目录
│   ├── train.py                           # 主训练脚本
│   ├── train_prune_s1.py                  # 剪枝训练（含稀疏度控制 + λ 拉格朗日）
│   ├── train_prune_s2.py                  # 剪枝后蒸馏微调
│   ├── run.sh / run_prune.sh              # 全部注释，仅供参考
│   ├── conf/
│   │   ├── w2v-bert/                      # 训练配置
│   │   │   ├── s1.yaml                    # Stage 1: 冻结 + LoRA
│   │   │   ├── s2.yaml                    # Stage 2: 联合微调
│   │   │   └── s3.yaml                    # Stage 3: LMFT
│   │   └── prune/                         # 剪枝配置
│   │       ├── dis_prune_s1.yaml           # 剪枝训练阶段1
│   │       ├── dis_prune_s2.yaml           # 剪枝训练阶段2
│   │       ├── s1/s2/s3.yaml               # 剪枝后微调
│   ├── local/
│   │   ├── spk_model.py                   # 3 种说话人模型
│   │   ├── spk_classifier.py              # ArcFace
│   │   ├── dataset.py                     # Train/Valid Dataset
│   │   ├── sampler.py                     # WavBatchSampler
│   │   ├── data_pipe.py                   # SCP/trial 加载包装
│   │   └── modules/
│   │       ├── pooling_v2.py              # GSP / ASP
│   │       └── ecapa_tdnn.py              # ECAPA-TDNN（备用）
│   ├── utils/
│   │   ├── lora_merge.py                  # LoRA 合并
│   │   ├── apply_prune_s1.py              # Hard Concrete 剪枝执行
│   │   ├── apply_prune_s2.py              # 学生权重提取
│   │   └── get_embd_w2v.py                # 标准 EER 评估
│   └── analysis/                          # ★ 大规模嵌入提取流水线
│       ├── step1_run_embedding_extraction.sh
│       ├── step2_run_compute_speaker_embeddings.sh
│       ├── step3_run_compute_speaker_similarities.sh
│       └── local/
│           ├── extract_embeddings.py
│           ├── compute_speaker_embeddings.py
│           └── compute_speaker_similarities.py
│
├── data/                                  # 训练/测试数据
├── models/                                # 下载的检查点
├── assets/                                # 论文图表
├── requirements.txt
├── AGENTS.md
└── README.md
```

---

## 3. 模型架构详解

### 3.1 完整前向流程

```
原始音频 (16kHz, 单声道)
│    shape: (n_samples,)  例：3 秒音频 = (48000,)
│
▼
┌──────────────────────────────────────────────────────┐
│  特征提取器 (Wav2Vec2FeatureExtractor)               │
│  · mel 滤波器组 (160 个 mel bins)                    │
│  · 帧长 25ms, 帧移 20ms                              │
│  · 输出: (1, T_feat, 160)                            │
└──────────────────────────────────────────────────────┘
│
▼
┌──────────────────────────────────────────────────────┐
│  特征投影 (FeatureProjection)                        │
│  · Conv1d(160 → 1024, kernel=1) + LayerNorm + Dropout│
│  · 输出: (1, T_feat, 1024)                           │
└──────────────────────────────────────────────────────┘
│
▼
┌──────────────────────────────────────────────────────┐
│  W2V-BERT 2.0 编码器 (25 层 Conformer)               │
│  · hidden_size=1024, intermediate_size=4096          │
│  · num_attention_heads=16, head_dim=64               │
│  · 每层: FFN(1024→4096→1024) + MHSA + Conv           │
│  · 参数: ~580M                                        │
│  · 各层 hidden_states: (1, T_feat, 1024)             │
│                                                      │
│  hidden_states = [h0 (输入投影), h1..h25 (25层输出)] │
│  共 26 个中间表示                                     │
└──────────────────────────────────────────────────────┘
│
▼  n_mfa_layers = -1 → 取全部 26 层
│  在 dim=-1 上拼接：[h0 | h1 | ... | h25]
│  shape: (1, T_feat, 1024×26) = (1, T, 26624)
│
▼
┌──────────────────────────────────────────────────────┐
│  逐层 Adapter (仅 Audio2Vec_based_Adapter)            │
│  每层: Linear(1024 → 128) → LayerNorm → ReLU         │
│        → Linear(128 → 128)                            │
│  26 个并行的 Adapter，每个输出 (1, T, 128)            │
│  在 dim=-1 上拼接: (1, T, 128×26) = (1, T, 3328)     │
└──────────────────────────────────────────────────────┘
│
▼
┌──────────────────────────────────────────────────────┐
│  ASP 池化 (Attentive Statistics Pooling)              │
│                                                      │
│  attention = Conv1d(3328→128→3328) → Softmax          │
│  w = attention(x.T).T  ,  w.shape = (1, T, 3328)     │
│  μ   = Σ(x · w, dim=1)            → (1, 3328)        │
│  σ   = √(Σ(x² · w) - μ²)          → (1, 3328)        │
│  out = [μ | σ]                     → (1, 6656)       │
│                                                      │
│  备选: GSP (Global Statistics Pooling)                │
│  out = [mean(x,dim=1) | std(x,dim=1)] → (1, 6656)   │
└──────────────────────────────────────────────────────┘
│
▼
┌──────────────────────────────────────────────────────┐
│  Bottleneck: Linear(6656 → 256)                       │
│  → L2 Normalized Embedding: shape (1, 256)            │
└──────────────────────────────────────────────────────┘
│                                         ▼ (仅训练时)
│                              ┌────────────────────────┐
│                              │ ArcFace 分类器          │
│                              │ L2_Norm(emb) @ L2_Norm(W)│
│                              │ + margin + s=32 缩放   │
│                              │ out: (batch, 5994/17982) │
│                              └────────────────────────┘
```

### 3.2 W2V-BERT 2.0 编码器参数

```json
{
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "num_attention_heads": 16,
    "num_hidden_layers": 25,
    "hidden_act": "swish",
    "attention_dropout": 0.0,
    "hidden_dropout": 0.0,
    "feat_proj_dropout": 0.0,
    "feature_projection_input_dim": 160,
    "conv_depthwise_kernel_size": 31
}
```

### 3.3 三种说话人模型对比

| 类名 | 用途 | 编码器 | Adapter | 池化 | 前向输出形状 |
|------|------|--------|---------|------|-------------|
| `Audio2Vec_based` | 基础模型 | 1 个 | 无 | GSP/ASP | `(B, 256)` |
| `Audio2Vec_based_Adapter` | ★ 主力（训练+推理） | 1 个 | 每层 128D | GSP/ASP | `(B, 256)` |
| `Audio2Vec_based_Weighted_ECAPATDNN` | 备选变体 | 1 个 | 无 | ECAPA-TDNN | `(B, 256)` |
| `Audio2Vec_based_Prune` | 剪枝训练 | 1 教师 + 1 学生 | 无 | 无 | `(L, B, T, 1024)` 元组 |
| `Lambda` | 剪枝辅助 | — | — | — | λ1, λ2 参数 |

**`Audio2Vec_based_Prune` 前向详析**：

```python
def forward(self, x):
    x_teacher = self.teacher_front(x).hidden_states  # 26 层 × (B, T, 1024)
    x_student = self.student_front(x).hidden_states
    # 取 distillation_layers（默认全部 25 层）
    x_t_out = [x_teacher[idx] for idx in self.distillation_layers]
    x_s_out = [x_student[idx] for idx in self.distillation_layers]
    return torch.stack(x_t_out), torch.stack(x_s_out)
    # 返回形状: (L, B, T, 1024), (L, B, T, 1024)  其中 L=len(distillation_layers)
```

### 3.4 LoRA 参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `r` | 64 | 低秩分解的秩 |
| `lora_alpha` | 128 | 缩放因子（LoRA 更新 × alpha/r） |
| `target_modules` | `["linear_q", "linear_v"]` | 仅对 Q 和 V 投影矩阵添加 LoRA |
| `lora_dropout` | 0.0 | LoRA 层的 dropout |
| `bias` | `"none"` | 不对 bias 添加 LoRA |

LoRA 仅应用于 W2V-BERT 编码器的 `linear_q` (Query) 和 `linear_v` (Value) 线性层。Key 投影不加 LoRA，FFN 不加 LoRA。合并后的可训练参数约 6.2M（占总模型 ~1%）。

### 3.5 ArcFace 分类器

```python
ArcFace(in_features=256, out_features=N, s=32.0, m=0.2, easy_margin=False)
```

**前向计算**：

```
1. cosine = L2(emb) @ L2(W).T     # 余弦相似度
2. sine   = sqrt(1 - cosine²)
3. phi    = cosine × cos(m) - sine × sin(m)     # cos(θ + m)
4. one_hot[label] = phi[label], 其余 = cosine   # 仅正类加 margin
5. output = one_hot × s                          # 缩放
```

**margin 选择**：
- Stage 1/2: `m = 0.2`（标准 margin）
- Stage 3 (LMFT): `m = 0.5`（更大 margin，更强调区分度）

### 3.6 池化层详解

**GSP (Global Statistics Pooling)**：
```python
out = [x.mean(dim=1) | x.std(dim=1)]    # (B, D) → (B, 2D)
```
无参数，直接拼接均值和标准差。

**ASP (Attentive Statistics Pooling)**：
```python
w = Conv1d(D→D_hidden→D) → Softmax(dim=2)     # 注意力权重 (B, T, D)
μ = Σ(x · w, dim=1)                              # 加权均值
σ = √(Σ(x² · w, dim=1) - μ²)                    # 加权标准差
out = [μ | σ]                                     # (B, 2D)
```
~0.5M 额外参数，通过注意力学习更重要的帧。

---

## 4. 预训练权重

从 HuggingFace 下载 W2V-BERT 2.0 预训练权重：

```
URL：https://huggingface.co/facebook/w2v-bert-2.0/blob/main/model.safetensors
目标路径：
deeplab/pretrained/audio2vector/ckpts/facebook/w2v-bert-2.0/model.safetensors
```

该目录下已有的配置文件：

| 文件 | 用途 |
|------|------|
| `config.json` | 未使用（HuggingFace 原始配置） |
| `config_tea.json` | Stage 1 教师：含 `intermediate_size_group` 等字段，`prune: false` |
| `config_prune_tea.json` | Stage 2/3 教师：与 tea 基本相同，但文件名含 `prune` → 跳过 safetensors |
| `config_prune_stu.json` | 剪枝学生：`prune: true`，启用 Hard Concrete 门 |
| `config_prune_stu_0.8.json` | 剪枝后学生：空的初始模板，由 `apply_prune_s1.py` 填入实际参数 |
| `preprocessor_config.json` | 特征提取器配置（mel 滤波器组参数） |

**关键判断**：`api.py:165` — 如果配置文件路径字符串中**不包含** `"prune"` 子串，则从 `model.safetensors` 加载预训练权重。因此 `config_tea.json` 会加载权重，而 `config_prune_*.json` 都跳过 safetensors 加载。

---

## 5. 数据准备

### 5.1 目录结构

```
data/
├── AudioData/
│   ├── musan/                         # MUSAN 噪声语料
│   │   ├── noise/free-sound/          # 环境噪声
│   │   ├── noise/sound-bible/         # 环境噪声
│   │   ├── music/fma/                 # 音乐（仅无 vocal 的文件）
│   │   ├── music/jamendo/             # 音乐
│   │   ├── speech/librivox/           # 多人语音（babble noise）
│   │   └── speech/us-gov/             # 多人语音
│   └── rirs_noise/                    # RIR 房间冲激响应
│       └── simulated_rirs/
│           ├── mediumroom/
│           └── smallroom/
├── vox2dev/
│   └── dev.spk2utt                    # VoxCeleb2 dev 训练列表
└── test_vox/
    └── vox1-o/
        ├── wav_copy.scp               # VoxCeleb1-O 测试文件列表
        └── trials                     # VoxCeleb1-O 标准 trial
```

### 5.2 文件格式规范

**`.spk2utt`（训练数据）**：Tab 分隔，每行一条 utterance

```
spk_id_001\t/home/data/vox2dev/wav/id10270/abc123.wav
spk_id_001\t/home/data/vox2dev/wav/id10270/def456.wav
spk_id_002\t/home/data/vox2dev/wav/id10271/ghi789.wav
```

训练时，数据集通过 `load_audio_corpus(dataset_dir, subsets)` 加载，返回 `spk2utt: dict[str, list[str]]`。

**`.scp`（验证/测试数据）**：Tab 分隔

```
utt_id_001\t/home/data/vox1-o/wav/utt1.wav
utt_id_002\t/home/data/vox1-o/wav/utt2.wav
```

**`trials`（验证/测试数据）**：空格分隔

```
1 utt_id_001 utt_id_002      # 正样本 (同说话人)
0 utt_id_001 utt_id_003      # 负样本 (不同说话人)
```

### 5.3 音频要求

- **采样率**：16000 Hz（`load_audio` 自动重采样非 16kHz 文件）
- **格式**：WAV / FLAC / MP3（WAV 和 FLAC 通过 `soundfile`，MP3 通过 `pydub`）
- **通道**：默认取第 0 通道 (`channels=0`)，立体声自动取左声道
- **裁剪**：训练时截断到 `dur_range`（如 2~3 秒的随机段），推理时截断到 `max_valid_dur`

### 5.4 Speaker 数量配置

YAML 中 `classifier.out_features` 必须匹配实际数据：

| 配置 | Speaker 数 | 包含速度扰动 |
|------|-----------|-------------|
| `out_features: 5994` | VoxCeleb2 dev 原始 5994 人 | × |
| `out_features: 17982` | 5994 × 3 (speed 0.9/1.0/1.1) | √ |

Stage 1/2 用 17982，Stage 3 (LMFT) 用 5994（不包含速度扰动扩展）。

### 5.5 数据增强管线

`Train_Dataset.__getitem__` 中的数据增强顺序：

```
1. load_audio(wav_path, sr=16000)        # 加载 + 重采样
2. truncate_audio_random(signal, dur)    # 随机截取 dur 秒
3. speed_augmentation(signal, sr, shift) # 速度扰动 (0.9× / 1.0× / 1.1×)
4. 随机选择增强类型:
   - 'none'   → 不增强
   - 'noise'  → MUSAN 加噪 (SNR 5~20 dB)
   - 'reverb' → RIR 混响
5. torch.from_numpy(signal).float()      # → tensor
```

### 5.6 大规模推理数据格式

`extract_embeddings.py` 使用 `os.walk` 遍历，`parse_speaker_utt` 提取 speaker_id：

```
audio/
  {dataset}/                # 第一级：数据集名
    {speaker_id}/           # 第二级：说话人 ID（取此目录名作为 speaker_id）
      {utterance}.wav
  {dataset}/
    {speaker_id}/
      sub/                  # 可选更深子目录
        {utterance}.flac
```

---

## 6. 完整训练流程

> ⚠️ **所有命令必须在 `recipes/DeepASV/` 目录下执行。**

### 6.1 WavBatchSampler 分批算法

训练中的关键组件——确保每批内所有音频等长，避免 padding：

```python
class WavBatchSampler:
    def __iter__(self):
        batch, dur = [], random.uniform(2, 3)  # 随机选取 2~3 秒的目标长度
        for idx in self.sampler:
            batch.append((idx, dur))           # 每个样本带相同的 dur
            if len(batch) == 64:               # batch_size
                yield batch
                batch, dur = [], random.uniform(2, 3)  # 新批 → 新随机长度
```

`Dataset.__getitem__` 收到 `(idx, dur)` 后调用 `truncate_audio_random(signal, dur * sr)` 截断到精确的 `dur` 秒 → 同一批内所有 tensor 等长 → PyTorch 默认 collate 直接 stack → `(B, T)` shape。

### 6.2 检查点命名规则

```
results/checkpoints/{tag}{YYMMDDHHmmss}/
  例：results/checkpoints/vox2_251005144134/
    ckpt_0001.pth           # epoch 1 结束 (S1: 15 个, S2: 4 个, S3: 2 个)
    ckpt_0002.pth
    ckpt_0002_1000item.pth  # items_save=true 时，每 1000 iter 保存
    ckpt_0002_2000item.pth
    ...
    merge_lora.pth          # lora_merge.py 输出
    train.yaml              # 训练配置副本
    logs.json               # 训练日志
```

### 6.3 Stage 1 — 冻结编码器 + LoRA 训练

**目标**：仅训练 ~6.2M LoRA + Adapter + 池化 + 分类器参数，编码器冻结

**配置详解** (`conf/w2v-bert/s1.yaml`)：

```yaml
seed: 24
use_amp: true                          # bfloat16 混合精度
use_gradient_clipping: true            # max_norm=1.0, L2
gradient_accumulation: 1               # 不累积梯度
cudnn_benchmark: false

optimizer: !name:torch.optim.AdamW
    lr: 0.0001                         # LoRA + Adapter 用较高学习率
    weight_decay: 0.0001

scheduler: !name: WarmupLR_withStepDecay
    warmup_step: 5                     # 前 5 epoch 线性 warmup
    decay_step: 5                      # 每 5 epoch lr × 0.1
    gamma: 0.1

num_epochs: 15
batch_size: 64                         # 每卡 64 条
valid_batch_size: 1                    # 验证逐条处理
num_workers: 16                        # CPU 数据加载线程
dur_range: [2, 3]                      # 随机截断 2~3 秒段
max_valid_dur: 60                      # 验证最多用 60 秒
speed_perturbation: [0.9, 1.1]        # 速度扰动扩展
data_aug: true                         # MUSAN + RIR 增强
embd_dim: 256

peft_config: !apply:create_lora_config
    model_type: 'w2v-bert'
    r: 64                               # LoRA 秩
    lora_alpha: 128                     # 缩放
    target_modules: ["linear_q","linear_v"]

spk_model: !new:Audio2Vec_based_Adapter
    frozen_encoder: true                # ★ 冻结 580M 参数
    encoder_config: 'config_tea.json'   # ★ 必须用 tea 版本加载 safetensors
    n_mfa_layers: -1                    # 全部 26 层
    pooling_layer: 'ASP'

classifier: !new:ArcFace
    out_features: 17982                 # 5994 × 3 (速度扰动)
    s: 32, m: 0.2
```

```bash
OMP_NUM_THREADS="16" CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
torchrun --nnodes 1 --nproc_per_node=8 --master_port=12885 train.py \
  --tag vox2_ --is_distributed true --yaml conf/w2v-bert/s1.yaml
```

### 6.4 ★ 阶段间操作 1：合并 LoRA

```bash
cd recipes/DeepASV/utils
# 编辑 lora_merge.py 第 31 行，修改 ckpt_path
python3 lora_merge.py
cd ../..
```

**`lora_merge.py` 逐行解析**：

```python
# 1. 用 config_prune_tea.json 重建模型（★ 不能用 config_tea.json）
model = Audio2Vec_based_Adapter(
    encoder_config='config_prune_tea.json',  # 含 group 字段，兼容合并后结构
    frozen_encoder=True, n_mfa_layers=-1, pooling_layer='ASP',
    peft_config=create_lora_config(r=64, lora_alpha=128))

# 2. 加载 Stage 1 权重
ckpt = torch.load('.../ckpt_0001.pth')
model.load_state_dict(ckpt['modules']['spk_model'], strict=False)

# 3. 合并 LoRA → 基础权重中
model.front.encoder = model.front.encoder.merge_and_unload()
# 此后 model 是一个标准 Wav2Vec2BertModel（无 PEFT 包装）

# 4. 保存
ckpt['modules']['spk_model'] = model.state_dict()
torch.save(ckpt, 'merge_lora.pth')
```

### 6.5 Stage 2 — 联合微调

**变化（对比 Stage 1）**：

| 字段 | S1 值 | S2 值 | 原因 |
|------|-------|-------|------|
| `frozen_encoder` | `true` | `false` | 解冻编码器 |
| `peft_config` | LoRA | `null` | LoRA 已合并到基础权重 |
| `encoder_config` | `config_tea.json` | `config_prune_tea.json` | 文件名含 prune → 不加载 safetensors |
| `optimizer.lr` | `1e-4` | `1e-5` | 全模型微调用更低学习率 |
| `scheduler` | `WarmupLR_withStepDecay` | `WarmupCosineScheduler` | 余弦衰减更平滑 |
| `num_epochs` | 15 | 4 | 快速适应 |
| `items_save` | `false` | `true` | 保存中间检查点 |

**WarmupCosineScheduler 行为**（`scheduler.py:74-116`）：

```
warmup_epoch=0, fix_epoch=2:
  step < warmup_step (0):  lr = max_lr × step/warmup (即时 lr = max_lr)
  warmup ≤ step < fix:     lr = min_lr + 0.5(max_lr-min_lr)(1+cos(π·(step-warmup)/(fix-warmup)))
  step ≥ fix:              lr = min_lr (保持最低)
```

```bash
OMP_NUM_THREADS="16" CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
torchrun --nnodes 1 --nproc_per_node=8 --master_port=12885 train.py \
  --tag vox2_ --is_distributed true --yaml conf/w2v-bert/s2.yaml \
  --pretrain /path/stage1/merge_lora.pth
```

### 6.6 Stage 3 — LMFT

**变化**：

| 字段 | S2 值 | S3 值 |
|------|-------|-------|
| `classifier.out_features` | 17982 | 5994 |
| `classifier.m` | 0.2 | 0.5 |
| `num_epochs` | 4 | 2 |
| `batch_size` | 64 | 32 |
| `dur_range` | [2,3] | [5,6] |
| `speed_perturbation` | [0.9,1.1] | [] |
| `data_aug` | true | false |
| `scheduler_lmft` | — | `WarmupCosineScheduler(0, 1)` |

```bash
OMP_NUM_THREADS="16" CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
torchrun --nnodes 1 --nproc_per_node=8 --master_port=12885 train.py \
  --tag vox2_ --is_distributed true --yaml conf/w2v-bert/s3.yaml \
  --pretrain /path/stage2/best_ckpt.pth
```

### 6.7 训练循环详解

**`Trainer.train_one_epoch` 核心代码**：

```python
for iter_idx, inputs in enumerate(train_dataloader, start=1):
    inputs = self.scatter_data(inputs)   # 移至 GPU

    if use_amp:
        with ExitStack() as stack:
            if is_distributed and iter_idx % grad_accum != 0:
                # 梯度累积中间步：跳过 DDP 梯度同步
                for m in modules: stack.enter_context(m.no_sync())

            with torch.amp.autocast('cuda', torch.bfloat16):
                predictions = compute_forward(inputs)      # 子类定义
                loss_dict = loss_fn(inputs, predictions)   # 子类定义
                loss = sum(loss_dict.values()) / grad_accum

            amp_scaler.scale(loss).backward()

        if iter_idx % grad_accum == 0:    # 累积完成 → 更新
            amp_scaler.unscale_(optimizer)
            clip_gradient(max_norm=1.0, norm_type=2)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            optimizer.zero_grad()

    # 每 iter 记录
    train_logs = update_logs(train_logs, loss_dict)
    train_logs = update_logs(train_logs, eval_fn(inputs, predictions))

    # LMFT 调度器 (每 iter 步进)
    if scheduler_lmft: scheduler_lmft.step()

    # 中间检查点保存
    if items_save and iter_idx % item_save_steps == 0:
        valid_logs = validate_once(epoch_idx)
        ...

# epoch 结束：完整验证 + epoch 级调度器步进
valid_logs = validate_once(epoch_idx)
if scheduler: scheduler.step()
```

**`validate_once` (train.py)**：逐条提取嵌入 → 各 GPU all_reduce 汇总 → 构建 utt2embd → 在 trial 上计算 EER

**`load_checkpoints`**：支持 classifier 维度不匹配（LMFT 阶段 17982 → 5994），自动截断/补齐。

---

## 7. 结构化剪枝流程

> 完整顺序：`dis_prune_s1 → apply_prune_s1.py → dis_prune_s2 → apply_prune_s2.py → prune/s1 → s2 → s3`

### 7.1 前置条件

剪枝 Stage 1 需要已完成 Stage 2 训练的教师模型。至少完成 [§6.5](#65-stage-2--联合微调)。

### 7.2 Hard Concrete 剪枝门机制

每个可剪枝的组件（FFN 神经元组 / Attention Head / Conv 通道）分配一个 `log_alpha` 参数。前向时通过 Hard Concrete 分布采样得到门控值：

```
u ~ Uniform(0, 1)
s = Sigmoid((log(u) - log(1-u) + log_alpha) / β)   # β 温度参数
z = clamp((ζ-γ)s + γ, 0, 1)                          # Hard Concrete → 0 或 1
output = input × z                                    # 门控乘法
```

训练中 `log_alpha` 逐渐收敛：
- `log_alpha << 0` → `z ≈ 0` → 该组件被剪掉
- `log_alpha > 0` → `z ≈ 1` → 保留

训练结束后 `model.prune()` 永久移除 `log_alpha > 0`（即 z ≈ 0）的组件。

### 7.3 Prune Stage 1 — 知识蒸馏 + 剪枝门训练

**Loss 函数**（`train_prune_s1.py:89-94`）：

```python
loss_l1  = L1(x_student, x_teacher)                                    # 逐元素 L1
loss_cos = -CosineSimilarity(x_student, x_teacher).mean()             # 余弦相似度
loss_reg = λ₁(sparsity - target_sparsity) + λ₂(sparsity - target_sparsity)²  # 拉格朗日

total_loss = loss_l1 + loss_cos + loss_reg
```

**λ 更新机制（递增拉格朗日乘子法）**：

```python
# trainer.py:157-179
param_groups = [
    {'params': main_params,        'lr': +2e-4, 'name': 'main'},      # 蒸馏主参数
    {'params': log_alpha_params,   'lr': +2e-2, 'name': 'log_alpha'}, # 剪枝门参数
    {'params': lambda_params,      'lr': -2e-2, 'name': 'lambda'},    # λ₁, λ₂（负 lr → 自动增大惩罚）
]
```

`log_alpha` 和 `λ` 的 lr 符号相反——`λ` 越来越大（增强稀疏惩罚），`log_alpha` 越来越小（更多门关闭）。

**稀疏度预热**（`train_prune_s1.py:69-74`）：

```python
def get_target_sparsity(self):
    if cur_steps >= 10000:
        return 0.8                   # 最终目标：80% 稀疏度
    return 0.8 × (cur_steps / 10000) # 前 10000 步线性增加
```

**配置**：

```yaml
prune_opt:
    distill_lr: 2e-4
    reg_lr: 2e-2
target_sparsity: 0.8
warmup_steps: 10000
num_epochs: 20
batch_size: 16
dur_range: [4, 4]
speed_perturbation: []
data_aug: false
```

```bash
# ★ 先编辑 conf/prune/dis_prune_s1.yaml 第 59 行：
#   pretrain_encoder: 'results/checkpoints/YOUR_STAGE2_DIR/best_ckpt.pth'

OMP_NUM_THREADS="12" CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
torchrun ... train_prune_s1.py --yaml conf/prune/dis_prune_s1.yaml
```

### 7.4 ★ 剪枝间操作 1

```bash
cd utils
# 编辑 apply_prune_s1.py:10 → YOUR_PRUNE_S1_DIR
python3 apply_prune_s1.py
cd ..
```

**脚本逻辑**：

```
1. 加载 prune_s1 检查点
2. 用 config_prune_stu.json 重建学生模型
3. 加载训练好的 log_alpha → model.prune()
   → 遍历所有层，log_alpha > 0 的组件被永久删除
   → 返回 model.config（实际剪枝后的维度）
4. 将 pruned 后权重写回 → prune_update.pth
5. 将 model.config 写入 config_prune_stu_0.8.json (prune=false)
```

### 7.5 Prune Stage 2 — 蒸馏微调

用剪枝后的配置 `config_prune_stu_0.8.json` 进行第二轮蒸馏微调（无剪枝门，纯蒸馏）。

```bash
torchrun ... train_prune_s2.py --yaml conf/prune/dis_prune_s2.yaml \
  --pretrain /path/prune_update.pth
```

### 7.6 ★ 剪枝间操作 2

```bash
cd utils
# 编辑 YOUR_PRUNE_S2_DIR 和 YOUR_STAGE2_DIR
python3 apply_prune_s2.py
cd ..
```

**逻辑**：取 prune_s2 的学生编码器权重 → 注入到原始 Stage 2 检查点的 `front.encoder` 位置 → 输出 `prune_dis.pth`。

### 7.7 剪枝后微调

```bash
# 三个微调阶段与完整训练流程对应但配置不同
# encoder_config: 'config_prune_stu_0.8.json'
torchrun ... train.py --yaml conf/prune/s1.yaml --pretrain /path/prune_dis.pth
torchrun ... train.py --yaml conf/prune/s2.yaml --pretrain /path/prune_ft_s1/best_ckpt.pth
torchrun ... train.py --yaml conf/prune/s3.yaml --pretrain /path/prune_ft_s2/best_ckpt.pth
```

### 7.8 占位符汇总

| 文件:行号 | 占位符 | 替换为 |
|----------|--------|--------|
| `utils/lora_merge.py:31` | `YOUR_STAGE1_DIR` | Stage 1 检查点目录名 |
| `conf/prune/dis_prune_s1.yaml:59` | `YOUR_STAGE2_DIR` | Stage 2 目录（剪枝教师来源） |
| `utils/apply_prune_s1.py:10` | `YOUR_PRUNE_S1_DIR` | 剪枝 Stage 1 目录名 |
| `utils/apply_prune_s2.py:10` | `YOUR_PRUNE_S2_DIR` | 剪枝 Stage 2 目录名 |
| `utils/apply_prune_s2.py:37` | `YOUR_STAGE2_DIR` | 原始 Stage 2 目录（注入目标） |

---

## 8. 推理与嵌入提取

### 8.1 标准 EER 评估

```bash
cd recipes/DeepASV/utils
python3 get_embd_w2v.py
```

**必需条件**：检查点 `.pth` 文件同一目录下存在 `train.yaml`。

**`get_embd_w2v.py` 流程**：

```
1. read_hyperyaml(train.yaml) → 重建 modules (spk_model + classifier)
2. 加载 checkpoint → 逐参数匹配 → load_state_dict (strict=False 的部分匹配)
3. 逐条 load_audio → truncate → spk_model(audio) → 嵌入 (256,)
4. 对 trial 中的每对 (utt1, utt2) 计算 cosine similarity
5. compute_eer(target_scores, non_target_scores) → EER / minDCF
```

### 8.2 推理脚本的模型加载要点

`spk_model` 实例化时：

- **encoder_config**：指定使用的 config JSON 文件（如 `config_prune_tea.json`、`config_prune_stu_0.8.json`）
- **frozen_encoder**：推理时设为 `false`（不需要冻结/解冻逻辑）
- **peft_config**：Stage 1 的 LoRA 模型需要对应 peft_config，Stage 2+ 设为 `null`

---

## 9. 大规模相似度分析流水线

`recipes/DeepASV/analysis/` — 完全独立于训练的推理流水线。

### 9.1 架构总览

```
Step 1:  多 GPU 并行提取 utterance 嵌入    → .pkl 文件
Step 2:  多进程 CPU 分组取均值              → 说话人 .pkl 文件
Step 3:  多进程 CPU 余弦相似度 + 统计        → JSON 报告
```

### 9.2 Step 1 — 嵌入提取

**输入格式**：`{DATA_ROOT}/{dataset}/{speaker_id}/{audio}.wav`（通过 `os.walk` 遍历）

**输出格式**：`{OUTPUT_DIR}/{speaker_id}/{utt_name}.pkl`，pickle 格式 numpy `(256,)`

**核心特性**：

| 特性 | 实现 |
|------|------|
| 多 GPU | `NUM_GPUS × PROCS_PER_GPU` 个进程，`file_stride` 分片 |
| 并行 I/O | DataLoader `num_workers` 并行 `load_audio` |
| 精度 | `batch_size=1`，无 padding，与 `get_embd_w2v.py` 完全一致 |
| 断点续跑 | `--skip_existing` 跳过已有 `.pkl` |
| 可中断 | Ctrl+C 终止后改 `MAX_FILES` 或保留现有结果重新启动 |

**分片示例**（4 GPU × 4 proc = 16 总进程）：

```
文件列表: [f0, f1, ..., f15, f16, ..., f31, ...]

GPU0  proc0  offset=0   →  f0,  f16, f32, ...
      proc1  offset=4   →  f4,  f20, f36, ...
      proc2  offset=8   →  f8,  f24, f40, ...
      proc3  offset=12  →  f12, f28, f44, ...

GPU1  proc0  offset=1   →  f1,  f17, f33, ...
      ...
```

**tmux 后台运行**：

```bash
tmux new-session -d -s emb_w2v "conda activate asv && \
  cd recipes/DeepASV/analysis && \
  bash step1_run_embedding_extraction.sh 2>&1 | tee /tmp/emb_master.log"

tmux a -t emb_w2v              # 查看进度
tail -f /tmp/emb_master.log    # 主日志
watch -n 10 'find {OUTPUT_DIR} -name "*.pkl" | wc -l'  # 完成数
```

### 9.3 Step 2 — 说话人嵌入

```bash
bash step2_run_compute_speaker_embeddings.sh
```

**算法**：对每个 speaker 目录下的所有 `.pkl` 嵌入取 `np.mean`：

```python
embeddings = [pickle.load(f) for f in embedding_files]
avg = np.mean(np.stack(embeddings), axis=0)   # (N, 256) → (256,)
```

**可配置选项**：
- `MIN_UTTERANCES`：最少 utterance 数门槛
- `EXCLUDE_VOICEPRINT_PREFIX`：排除特定前缀的文件
- `EXCLUDE_CLONE_PATTERN`：glob 模式排除（如 `*_clone_text_*`）
- `NUM_PROCESSES` / `CHUNK_SIZE`：并行度调优

### 9.4 Step 3 — 相似度分析

```bash
bash step3_run_compute_speaker_similarities.sh
```

**算法**：

```
1. 加载所有说话人嵌入 → L2 归一化 → emb_matrix (N, 256)
2. 多进程分片计算 Top-K 相似邻居（每行计算与全部列的余弦）
3. 采样统计 (至多 10M 对) → 均值/中值/标准差/阈值分布
4. 采样 2M 对 → argsort → Top-100 极端相似对
```

**输出文件**：

| 文件 | 内容 |
|------|------|
| `analysis_summary.json` | N_speakers, 均值, 中位数, 标准差, min/max |
| `threshold_statistics.json` | {0.60~0.95} 各阈值以上的 pair 数量与占比 |
| `extreme_similarity_pairs.json` | Top-100 最相似的 (speaker1, speaker2, similarity) |
| `speaker_top_similarities.json` | 每个 speaker 的 Top-K 相似邻居 |
| `speaker_keys_mapping.json` | 说话人 ID → 矩阵索引 |

---

## 10. 关键技术与常见陷阱

### 10.1 sys.path 依赖

全部训练脚本和推理脚本通过 `sys.path.insert(0, ...)` 引入两个关键路径：
- 仓库根目录（使 `deeplab.*` 和 `local.*` 可导入）
- `deeplab/pretrained/audio2vector/module/transformers/src`（使 `import transformers` 指向魔改版）

训练脚本依赖 CWD 相对路径 → **必须从 `recipes/DeepASV/` 启动**。

`analysis/` 下的推理脚本使用 `__file__` 计算绝对路径，不依赖 CWD。

### 10.2 模型配置文件陷阱

| 错误 | 后果 | 正确做法 |
|------|------|---------|
| Stage 1 用 `config_prune_tea.json` | 跳过 safetensors → 随机权重训练 | 用 `config_tea.json` |
| lora_merge 用 `config_tea.json` | 合并后结构与 tea 不兼容 | 用 `config_prune_tea.json` |
| 推理用 prune YAML | `Audio2Vec_based_Prune` 返回元组，不是嵌入 | 用训练 YAML (Adapter 模型) |

### 10.3 魔改版 transformers

**新增的 Wav2Vec2BertConfig 字段**：

```json
{
    "prune": true / false,
    "intermediate_size_group": [4096, 4096, ...],   // 可剪枝的 FFN 维组
    "num_attention_heads_group": [16, 16, ...],     // 可剪枝的 Head 组
    "conv_group": [32, 32, ...],                    // 可剪枝的 Conv 通道组
    "use_feed_forward": [true, true, ...],           // 每层是否使用 FFN
    "use_attention": [true, true, ...]               // 每层是否使用 Attention
}
```

**不要更新或替换此目录**。它和官方 transformers 不兼容。

### 10.4 检查点加载逻辑

**`load_checkpoints` (trainer.py:401-444)**：

```python
for key, module in self.modules.items():
    if key not in ckpt_data['modules']:
        print('<Not found>')          # 跳过，保持初始权重
    elif key == 'classifier':
        if curr_len == ckpt_len:      # 完全匹配
            load_state_dict()
        elif ckpt_len > curr_len:     # 截断 (LMFT: 17982→5994)
            curr['weight'] = ckpt['weight'][:curr_len]
        else:                         # 补齐
            curr['weight'][:ckpt_len] = ckpt['weight']
    else:
        # 逐参数形状匹配加载，不匹配则跳过
```

### 10.5 DDP 兼容性

- `api.py:192`：`delattr(self.encoder, 'masked_spec_embed')` 避免 DDP 报未使用参数
- 剪枝阶段 `find_unused_parameters=true`（学生有 log_alpha 但蒸馏 loss 不用它）
- `no_sync()` 在梯度累积中间步避免不必要的 DDP all_reduce

### 10.6 批量推理精度

`batch_size > 1` 需要 zero-padding，但 `forward_impl.py` 当前使用 `return_attention_mask=False`，padding 位置会影响 attention 和池化。**大规模提取默认 `batch_size=1`**，通过 `num_workers` 并行 I/O 补偿吞吐。

---

## 11. 预训练模型下载

| 训练集 | 模型 | V1-O EER | 下载 |
|--------|------|---------|------|
| Vox2+VoxBlink2 | LoRA_Adapter_MFA (base) | 0.23% | [model_base_0.23.pth](https://huggingface.co/zl389/w2v-bert-2.0_SV/blob/main/model_base_0.23.pth) |
| Vox2+VoxBlink2 | LoRA_Adapter_MFA (LMFT) | **0.14%** | [model_lmft_0.14.pth](https://huggingface.co/zl389/w2v-bert-2.0_SV/blob/main/model_lmft_0.14.pth) |

---

## 12. 常见操作速查

| 操作 | 方法 |
|------|------|
| 计算 FLOPs | `python3 recipes/DeepASV/local/spk_model.py` |
| 查看训练日志 | 打开 `results/checkpoints/{tag}_{ts}/logs.json` |
| 修改 speaker 数 | YAML 中 `classifier.out_features` |
| 修改 LoRA rank | YAML 中 `peft_config.r` + `lora_alpha` |
| 启用中间检查点 | YAML 中 `items_save: true` |
| 启用 wandb | YAML 中添加 `wandb_cfgs: {project: xxx, watch: false}` |
| 标准 EER | `cd utils && python3 get_embd_w2v.py` |
| 大规模提取 | `cd analysis && bash step1_run_embedding_extraction.sh` |
| GPU 日志 | `ls /tmp/emb_w2v_gpu*_p*.log` |
| 实时监控 | `watch -n 10 'find {OUTPUT_DIR} -name "*.pkl" \| wc -l'` |
| 调并行度 | 编辑 shell 顶部 `NUM_GPUS` / `PROCS_PER_GPU` / `NUM_WORKERS` |
| 断点续跑 | `SKIP_EXISTING=true` 自动生效 |
| 单文件调试 | `--max_files 1 --file_stride 1 --file_offset 0 --num_workers 0` |

---

## 13. 附录 A：全部配置详解

### A.1 config.py —— 配置生成脚本

`deeplab/pretrained/audio2vector/ckpts/facebook/w2v-bert-2.0/config.py` (30 行)

```python
import json

with open('./config.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 添加 Hard Concrete 剪枝门参数
data['limit_l'] = -0.1               # 门控下界
data['limit_r'] = 1.1                # 门控上界
data['temperature'] = 2/3            # Hard Concrete 温度参数 β

# FFN 可剪枝组：每层 FFN 的 intermediate_size 拆成 2 组
layer_num = data['num_hidden_layers']  # 25
data['intermediate_size_group'] = [[data['intermediate_size'], data['intermediate_size']]
                                    for _ in range(layer_num)]  # [[4096,4096], ...]
data['use_feed_forward'] = [[True, True] for _ in range(layer_num)]

# Conv 可剪枝组：每层 Conv 的 hidden_size 拆成 2 组
data['conv_group'] = [[data['hidden_size'], data['hidden_size']]
                       for _ in range(layer_num)]  # [[1024,1024], ...]

# Attention Head 可剪枝：每个 head 单独可剪
data['num_attention_heads_group'] = [data['num_attention_heads']
                                      for _ in range(layer_num)]  # [16, 16, ...]
data['use_attention'] = [True for _ in range(layer_num)]

# 生成 config_prune_stu.json (prune=true)
data['prune'] = True
with open('config_prune_stu.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

# 生成 config_prune_tea.json (prune=false)
data['prune'] = False
with open('config_prune_tea.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
```

### A.2 所有 YAML 配置对比表

**完整训练（3 阶段）**：

| 字段 | s1.yaml | s2.yaml | s3.yaml |
|------|---------|---------|---------|
| `encoder_config` | `config_tea.json` | `config_prune_tea.json` | `config_prune_tea.json` |
| `frozen_encoder` | `true` | `false` | `false` |
| `peft_config` | LoRA r=64 alpha=128 | `null` | `null` |
| `optimizer` | AdamW | AdamW | AdamW |
| `opt.lr` | 1e-4 | 1e-5 | 1e-5 |
| `scheduler` | Warmup+StepDecay(5,5,0.1) | WarmupCosine(0,2) | WarmupCosine(0,1) |
| `num_epochs` | 15 | 4 | 2 |
| `batch_size` | 64 | 64 | 32 |
| `dur_range` | [2,3] | [2,3] | [5,6] |
| `speed_perturbation` | [0.9,1.1] | [0.9,1.1] | [] |
| `data_aug` | true | true | false |
| `classifier.m` | 0.2 | 0.2 | 0.5 |
| `classifier.out_features` | 17982 | 17982 | 5994 |
| `items_save` | false | true | true |

**剪枝训练（2 个 distillation 阶段 + 3 个微调阶段）**：

| 字段 | dis_prune_s1.yaml | dis_prune_s2.yaml | prune/s1.yaml | prune/s2.yaml | prune/s3.yaml |
|------|-------------------|-------------------|---------------|---------------|---------------|
| `spk_model` | Prune(tea+stu) | Prune(tea+stu) | Adapter | Adapter | Adapter |
| `tea_config` | `config_prune_tea.json` | `config_prune_tea.json` | — | — | — |
| `stu_config` | `config_prune_stu.json` | `config_prune_stu_0.8.json` | — | — | — |
| `encoder_config` | — | — | `config_prune_stu_0.8.json` | `config_prune_stu_0.8.json` | `config_prune_stu_0.8.json` |
| `frozen_encoder` | — | — | `true` | `false` | `false` |
| `distill_lr` | 2e-4 | 2e-4 | — | — | — |
| `reg_lr` | 2e-2 | `null` | — | — | — |
| `target_sparsity` | 0.8 | 0.8 | — | — | — |
| `warmup_steps` | 10000 | 0 | — | — | — |
| `num_epochs` | 20 | 20 | 10 | 4 | 2 |
| `batch_size` | 16 | 16 | 64 | 64 | 32 |
| `dur_range` | [4,4] | [4,4] | [2,3] | [2,3] | [5,6] |
| `speed_perturbation` | [] | [] | [0.9,1.1] | [0.9,1.1] | [] |
| `data_aug` | false | false | true | true | false |

### A.3 模型配置文件 JSON 字段对比

```
字段                          config_tea  config_prune_tea  config_prune_stu  config_prune_stu_0.8
────────────────────────────────────────────────────────────────────────────────────────────
hidden_size                    1024        1024              1024              (剪枝后实际值)
intermediate_size              4096        4096              4096              (剪枝后实际值)
num_attention_heads            16          16                16                (剪枝后实际值)
num_hidden_layers              25          25                25                25
prune                          false       false             true              false
limit_l                        —           -0.1              -0.1              -0.1
limit_r                        —           1.1               1.1               1.1
temperature                    —           2/3               2/3               2/3
intermediate_size_group        —           [[4096,4096],…]   [[4096,4096],…]   (剪枝后实际值)
num_attention_heads_group      —           [16,16,…]         [16,16,…]         (剪枝后实际值)
conv_group                     —           [[1024,1024],…]   [[1024,1024],…]   (剪枝后实际值)
use_feed_forward               —           [[T,T],…]         [[T,T],…]         (剪枝后实际值)
use_attention                  —           [T,T,…]           [T,T,…]           (剪枝后实际值)
```

---

## 14. 附录 B：模块详解（未在前文中覆盖的）

### B.1 `deeplab/dataio/audio.py` —— 音频处理管线（491 行）

完整的 7 种音频增强操作：

| 函数 | 功能 | 关键参数 |
|------|------|---------|
| `norm_audio` | 标准差归一化 (`std`) 或最大值归一化 (`max`) | `mode='std'/'max'` |
| `norm_audio_to_int16` | 归一化到 [-32768, 32767] 范围（int16） | — |
| `pcm2signal` / `signal2pcm` | PCM bytes ↔ numpy float 互转 | — |
| `truncate_audio` | 固定截断（不足补零），支持 head/tail 模式 | `tlen`, `head_first=true` |
| `truncate_audio_random` | 随机截取（不足时交叉淡入淡出自拼接） | `tlen`, `crossfade=0` |
| `cat_audio_with_crossfade` | 两段音频的交叉淡入淡出拼接 | `crossfade` 长度 |
| `resample_audio` | 基于 sox 的重采样 | `resample` 目标采样率 |
| `add_reverberation` | RIR 卷积混响（使用 `fftconvolve`） | `prob`, `path_list` |
| `add_noise` | 单噪声加噪 | `prob`, `snr=[5,20]` |
| `add_noise_from_musan_dict` | MUSAN 三段式加噪 (noise/music/speech) | `snr=[5,20]`, `prob=1.0` |
| `speed_augmentation` | 速度扰动（0.9× / 1.1×） | `speed_shift` |

**交叉淡入淡出拼接** (`cat_audio_with_crossfade`):

```
sig1: [████████████████] crossfade 段末尾线性 fade-out
sig2:                     [████████████████] 开头线性 fade-in
结果: [████████████░░░░████████████████]
               ↑───────↑ crossfade 重叠区，两信号叠加
```

**MUSAN 三段式加噪** (`add_noise_from_musan_dict`):

```python
# 随机抽取噪声类型 (各 1/3 概率)
noise_type = random.choice(['noise', 'music', 'babb'])
if noise_type == 'noise':
    add_noise(signal, musan_dict['noise'], snr=[0, 15])
elif noise_type == 'music':
    add_noise(signal, musan_dict['music'], snr=[5, 15])
elif noise_type == 'babb':
    # 1~3 段随机 babble，SNR 13~20 dB
    add_noise(signal, musan_dict['babb'], snr=[13, 20], max_num=3)
```

### B.2 `deeplab/dataio/feature.py` —— 特征提取（95 行）

**`signal2tensor(signal, norm_mode)`**:
- 可选的 `std`/`max` 归一化
- 单通道：`preemphasis(signal, 0.97)` → tensor
- 多通道：逐通道 preemphasis → stack
- 预加重系数 0.97 保持与标准语音处理一致

**`logFbankCal` 类**:
```python
class logFbankCal(nn.Module):
    def forward(self, x, is_aug=[]):
        out = MelSpectrogram(x)    # (B, T) → (B, freq, time)
        out = torch.log(out + 1e-6)  # 对数压缩
        out = out - out.mean(axis=2).unsqueeze(2)  # 去均值
        # 可选：频谱增强（随机 mask 一段频率）
        if is_aug[i]:
            offset = random(1/8, 1/4) × freq_bins
            out[i][start:start+offset] *= random()/2
        return out
```
- 使用 `@torch.amp.autocast('cuda', enabled=False)` 禁用混合精度（特征提取始终用 float32）
- 频谱增强（SpecAugment 简化版）随机屏蔽一段频带

### B.3 `deeplab/utils/misc.py` —— 工具函数（45 行）

| 函数 | 功能 |
|------|------|
| `second_to_timeformat(s)` | 秒数 → `"Xh:XXm:XXs"` 格式 |
| `set_random_seed(seed)` | 固定 Python + NumPy + PyTorch + CUDA 全部随机种子 |
| `seed_worker(worker_id)` | DataLoader worker 种子初始化 |
| `trim_time_interval(t1, t2, min_t1, max_t2)` | 时间间隔裁剪 |
| `count_model_parameters(model)` | 只统计 `requires_grad=True` 的参数 |

### B.4 `deeplab/utils/pbar.py` —— 多进程安全进度条（100 行）

```python
class ProgressBar:
    def __init__(self, total):
        # 创建 mp.Manager 共享队列 (进程间通信)
        self.count_queue = m.Queue()
        self.error_queue = m.Queue()
        # 启动 listen 线程 (接收 count) + update 线程 (更新 tqdm 显示)
```

**工作模式**：
- `listen` 线程每秒从 `count_queue` 消费增量，累加到 `cur_steps`
- `update` 线程每秒用 `pbar.update(delta)` 刷新 tqdm
- 各 worker 进程通过 `count_queue.put(N)` 报告完成数
- `close()` 方法停止两个线程

### B.5 `recipes/DeepASV/local/modules/ecapa_tdnn.py` —— ECAPA-TDNN（518 行）

完整的 ECAPA-TDNN 实现，作为 `Audio2Vec_based_Weighted_ECAPATDNN` 的池化后端使用：

```
输入: (B, n_mels, T) [mel 频谱]

  SE-Res2Block_1  (C=512, scale=8)
  SE-Res2Block_2  (C=512, scale=8)
  SE-Res2Block_3  (C=512, scale=8)

     ↓ concat + Conv1d → [B, 1536, T]

  Attentive Statistics Pooling
  (同 ASP: 注意力加权均值+标准差) → [B, 3072]

     ↓ FC → BN → [B, 192]

  AAM-Softmax 输出层
```

**SE-Res2Block** = Squeeze-Excitation + Res2Net 多尺度卷积

### B.6 `deeplab/utils/corpus.py` —— 语料加载（78 行）

| 函数 | 功能 |
|------|------|
| `init_spk2utt(dataset_dir, subset, spk2utt)` | 解析 `.spk2utt` 文件 → dict |
| `load_musan_dict(dataset_dir)` | 遍历 noise/music/babb 三目录 → `{'noise':[...], 'music':[...], 'babb':[...]}` |
| `load_rirs(dataset_dir)` | 收集 mediumroom + smallroom 的所有 `.wav` |
| `load_audio_corpus(dataset_dir, subsets)` | YAML 可调用的语料加载入口，返回 `spk2utt` dict |

### B.7 `recipes/DeepASV/run.sh` —— 训练参考命令

```bash
# Stage 1: 冻结 + LoRA
torchrun ... train.py --yaml conf/w2v-bert/s1.yaml --tag vox2_

# Stage 2: 联合微调 (需 lora_merge 后)
torchrun ... train.py --yaml conf/w2v-bert/s2.yaml \
  --pretrain results/checkpoints/vox2_251005144134/merge_lora.pth

# Stage 3: LMFT
torchrun ... train.py --yaml conf/w2v-bert/s3.yaml \
  --pretrain results/checkpoints/vox2_251005145628/ckpt_0002.pth
```

### B.8 `recipes/DeepASV/run_prune.sh` —— 剪枝参考命令

```bash
# dis_prune_s1
torchrun ... train_prune_s1.py --yaml conf/prune/dis_prune_s1.yaml

# dis_prune_s2 (需 apply_prune_s1 后)
torchrun ... train_prune_s2.py --yaml conf/prune/dis_prune_s2.yaml \
  --pretrain results/checkpoints/prune_s1/prune_update.pth

# prune ft s1 (需 apply_prune_s2 后)
torchrun ... train.py --yaml conf/prune/s1.yaml \
  --pretrain results/checkpoints/prune_s2/prune_dis.pth

# prune ft s2 / s3 同理
```

### B.9 `recipes/DeepASV/local/data_pipe.py` —— 数据管道（26 行）

```python
def prepare_scp_and_trial_list(scp_path, trial_path, group_id=None):
    # 加载 SCP 文件 (utt_id → wav_path)
    # 加载 trial 文件 (label utt1 utt2)
    # 可选：为 utt_id 添加 group_id 前缀（多数据集场景）
    # 返回 (scp_list, trial_list)
```

### B.10 `deeplab/metric/eer.py` —— EER 计算（68 行）

**完整算法**：

```
1. 收集 target_scores (同说话人) 和 non_target_scores (不同说话人)
2. 将所有分数合并并排序 (升序)
3. DET 曲线: 对每个可能的阈值计算 FRR 和 FAR
   FRR[i] = (target_scores ≤ threshold[i] 的数量) / total_targets
   FAR[i] = (non_target_scores > threshold[i] 的数量) / total_non_targets
4. EER = min(|FRR - FAR|) 处的 (FRR+FAR)/2
5. minDCF = min(C_miss×FRR×P_target + C_fa×FAR×(1-P_target)) / C_default
   默认: P_target=0.01, C_miss=1, C_fa=1
```

---

## 15. 附录 C：运行参考命令速查

### 完整训练（从头开始）

```bash
cd /root/code/github_repos/w2v-BERT-2.0_SV-fork/recipes/DeepASV

# Step 0: 确保已下载 model.safetensors
# Step 0: 确保 data/ 目录结构正确

# Stage 1 (15 epochs, 8 GPU)
OMP_NUM_THREADS="16" CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
torchrun --nnodes 1 --nproc_per_node=8 --master_port=12885 train.py \
  --tag vox2_ --is_distributed true --yaml conf/w2v-bert/s1.yaml

# LoRA 合并
cd utils && python3 lora_merge.py && cd ..
# (先编辑 lora_merge.py:31 的 YOUR_STAGE1_DIR)

# Stage 2 (4 epochs)
OMP_NUM_THREADS="16" CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
torchrun --nnodes 1 --nproc_per_node=8 --master_port=12885 train.py \
  --tag vox2_ --is_distributed true --yaml conf/w2v-bert/s2.yaml \
  --pretrain results/checkpoints/vox2_XXXXXXXXXX/merge_lora.pth

# Stage 3 (2 epochs, LMFT)
OMP_NUM_THREADS="16" CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
torchrun --nnodes 1 --nproc_per_node=8 --master_port=12885 train.py \
  --tag vox2_ --is_distributed true --yaml conf/w2v-bert/s3.yaml \
  --pretrain results/checkpoints/vox2_YYYYYYYYYY/best_ckpt.pth
```

### 剪枝完整流程

```bash
# Prune S1 (需编辑 YAML 中的 pretrain_encoder)
OMP_NUM_THREADS="12" CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
torchrun ... train_prune_s1.py --yaml conf/prune/dis_prune_s1.yaml

# 应用剪枝
cd utils && python3 apply_prune_s1.py && cd ..

# Prune S2
torchrun ... train_prune_s2.py --yaml conf/prune/dis_prune_s2.yaml \
  --pretrain results/checkpoints/PRUNE_S1_DIR/prune_update.pth

# 提取权重
cd utils && python3 apply_prune_s2.py && cd ..

# Prune FT S1/S2/S3 (使用 conf/prune/s{1,2,3}.yaml)
torchrun ... train.py --yaml conf/prune/s1.yaml \
  --pretrain results/checkpoints/PRUNE_S2_DIR/prune_dis.pth
# ... 后续类似
```

### 大规模推理

```bash
cd recipes/DeepASV/analysis

# 嵌入提取 (后台)
tmux new-session -d -s emb "conda activate asv && \
  bash step1_run_embedding_extraction.sh 2>&1 | tee /tmp/emb.log"

# 等待完成后
bash step2_run_compute_speaker_embeddings.sh
bash step3_run_compute_speaker_similarities.sh
```

---

## 16. 引用

```bibtex
@article{li2025enhancing,
  title   = {Enhancing Speaker Verification with w2v-BERT 2.0
             and Knowledge Distillation guided Structured Pruning},
  author  = {Li, Ze and Cheng, Ming and Li, Ming},
  journal = {arXiv preprint arXiv:2510.04213},
  year    = {2025}
}
```
