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

recipes/DeepASV/          # ★ 训练脚本的工作目录（所有命令都在此目录下执行）
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
    modules/              # ECAPA-TDNN、ASP 等池化层
  utils/
    lora_merge.py         # 合并 LoRA 权重到基础模型
    apply_prune_s1.py     # 执行结构化剪枝并生成新的模型配置
    apply_prune_s2.py     # 提取蒸馏后的学生模型权重
    get_embd_w2v.py       # 提取说话人嵌入并计算 EER
```

## 环境搭建

```bash
conda create -y -n asv python=3.9
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip uninstall transformers          # ★ 必须卸载系统 transformers
conda install -c conda-forge sox   # torchaudio sox_effects 依赖
```

**关键：必须卸载 pip 安装的 transformers**。本仓库使用魔改版 transformers（位于 `deeplab/pretrained/audio2vector/module/transformers/src`），通过 `sys.path.append` 方式引入，不能与官方 transformers 共存。

### 预训练权重下载

从 HuggingFace 下载 `model.safetensors` 放入：
```
deeplab/pretrained/audio2vector/ckpts/facebook/w2v-bert-2.0/model.safetensors
```

## 训练流程（严格有序）

### 完整训练（3阶段）

所有命令在 `recipes/DeepASV/` 目录下执行：

```bash
# Stage 1：冻结编码器 + LoRA 适配器训练
torchrun --nnodes 1 --nproc_per_node=8 --master_port=12885 train.py \
  --tag vox2_ --is_distributed true --yaml conf/w2v-bert/s1.yaml

# ★ 阶段间操作：合并 LoRA 参数
cd utils && python3 lora_merge.py && cd ..

# Stage 2：联合微调（编码器解冻）
torchrun ... train.py --yaml conf/w2v-bert/s2.yaml --pretrain /path/stage1/merge_lora.pth

# Stage 3：Large Margin Fine-Tuning (LMFT)
torchrun ... train.py --yaml conf/w2v-bert/s3.yaml --pretrain /path/stage2/best_ckpt.pth
```

### 剪枝流程（5+步骤）

```
dis_prune_s1 → apply_prune_s1.py → dis_prune_s2 → apply_prune_s2.py → prune/s1 → s2 → s3
```

每个 `apply_prune_*.py` 脚本在 `recipes/DeepASV/utils/` 目录下执行，会修改模型权重和配置文件。

### 推理/测试

```bash
cd recipes/DeepASV/utils
python3 get_embd_w2v.py
```

## 关键技术细节（Agent 必知）

### sys.path 依赖

所有训练脚本通过 `sys.path.append('../..')` 引用 `deeplab/` 包。**必须从 `recipes/DeepASV/` 目录启动**，否则导入会失败。

### YAML 配置格式

使用 `hyperpyyaml`（SpeechBrain 风格），不是标准 YAML：
- `!new:local.spk_model.Audio2Vec_based_Adapter` — 实例化类
- `!name:torch.optim.AdamW` — 延迟实例化（传 callable）
- `!apply:deeplab.utils.misc.set_random_seed` — 立即调用函数
- `!ref <var>` — 引用同文件中的变量

### 模型配置文件区别

`deeplab/pretrained/audio2vector/ckpts/facebook/w2v-bert-2.0/` 下有多个配置：

| 配置文件 | 用途 | 关键差异 |
|---------|------|---------|
| `config_tea.json` | Stage 1 教师（加载 safetensors 权重） | `"prune": false`，无剪枝门 |
| `config_prune_tea.json` | Stage 2/3 及剪枝教师 | 增加 `intermediate_size_group` 等字段 |
| `config_prune_stu.json` | 剪枝学生（训练时） | `"prune": true`，启用 Hard Concrete 门 |
| `config_prune_stu_0.8.json` | 剪枝后学生（推理） | 剪枝完成后由 `apply_prune_s1.py` 生成 |

**★ Stage 1 必须使用 `config_tea.json` 而非 `config_prune_tea.json`**，否则会跳过加载 safetensors 初始化权重（`api.py:165` 中的 `'prune' not in self.model_config` 判断）。

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

### 需手动更新的检查点路径

`utils/` 下的脚本中检查点路径使用了 `YOUR_*_DIR` 占位符，训练后需替换为实际目录名：
- `lora_merge.py` → `YOUR_STAGE1_DIR`（Stage 1 检查点目录）
- `apply_prune_s1.py` → `YOUR_PRUNE_S1_DIR`（剪枝 Stage 1 检查点目录）
- `apply_prune_s2.py` → `YOUR_PRUNE_S2_DIR` 和 `YOUR_STAGE2_DIR`
- `conf/prune/dis_prune_s1.yaml` → `YOUR_STAGE2_DIR`（`pretrain_encoder` 字段）
- `get_embd_w2v.py` 已默认指向 `models/model_lmft_0.14.pth`（下载的最佳模型）

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

加载检查点时，`classifier` 权重支持大小不匹配的部分加载（为 LMFT 阶段设计，Stage 3 的 `out_features=5994` 小于 Stage 1/2 的 `17982`）。

### DDP 兼容性处理

- `api.py:192`：加载 W2V-BERT 时会 `delattr(self.encoder, 'masked_spec_embed')`，避免 DDP 中未使用参数报错
- 训练默认使用 8 GPU + NCCL 后端
- 默认使用 bfloat16 混合精度训练

### 数据格式要求

- 训练数据：`.spk2utt` 文件（tab 分隔：`spk_id\tutt_path`）
- 验证数据：`.scp` 文件（tab 分隔：`utt_id\twav_path`）+ trial 文件（空格分隔：`label utt1 utt2`）
- 音频采样率：16000 Hz
- 噪声增强需要 MUSAN 和 RIRs 数据集

### 魔改版 transformers

`deeplab/pretrained/audio2vector/module/transformers/` 是 HuggingFace transformers 的 fork，主要修改：
- `Wav2Vec2BertModel` 中添加了 Hard Concrete 剪枝门（`log_alpha` 参数）
- 支持按层级控制 attention head 数、FFN 维度、Conv 维度的结构化剪枝
- 模型配置中新增 `prune`、`intermediate_size_group`、`num_attention_heads_group`、`conv_group`、`use_feed_forward`、`use_attention` 等字段
- **不要**用 pip 安装的 transformers 替换或更新此目录

## 常见操作

| 操作 | 命令 / 位置 |
|------|------------|
| 计算模型 FLOPs | `spk_model.py` 底部的 `__main__` 块，使用 `calflops` |
| 查看训练日志 | `results/checkpoints/<tag>_<timestamp>/logs.json` |
| 查看训练配置 | `results/checkpoints/<tag>_<timestamp>/train.yaml` |
| 修改 speaker 数量 | YAML 中 `classifier.out_features`（需匹配数据集） |
| 修改 LoRA rank | YAML 中 `peft_config.r` 和 `lora_alpha` |
| 启用/禁用 wandb | YAML 中添加/删除 `wandb_cfgs` 字段 |
