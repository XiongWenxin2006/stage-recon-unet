# StageRecon U-Net

基于**分阶段自监督图像重建预训练**的模块化 U-Net 医学图像分割研究框架。

默认自监督任务仅为**图像重建**（L1 / MSE / 组合损失）。除非配置明确启用，否则不包含 SimCLR / InfoNCE 等对比学习损失。

## 项目介绍

将 U-Net 拆分为可替换模块：

```text
Encoder → Bottleneck → Decoder → Task Head
```

使用无标签医学图像完成前两个重建预训练阶段，第三阶段联合重建，最后迁移到有标签分割任务。三个阶段共享同一个 `ModularUNet`，通过 `StageSpec` 控制模块级参数继承与冻结策略。

### 研究目标

1. 分阶段训练相较随机初始化更快收敛  
2. 提高预训练参数在下游分割中的泛化能力  
3. 对比随机初始化、端到端重建预训练与三阶段重建预训练  
4. 支持 Encoder / Bottleneck / Decoder / Head / Loss / Dataset 快速替换  
5. 支持消融、少样本与多随机种子实验  
6. 支持 WebDataset 流式训练（本地 / S3 / HTTP / pipe）

## 三阶段训练流程

```text
Stage 1  Encoder 表征重建
  Encoder/Bottleneck/Decoder/ReconHead ← random
  重点获得 Encoder + Bottleneck
  保存 stage1_best.pt

        │ 仅迁移 Bottleneck
        ▼

Stage 2  Decoder 重建协作
  Encoder/Decoder/ReconHead ← random
  Bottleneck ← Stage 1
  重点训练 Decoder（与继承后的 Bottleneck）
  保存 stage2_best.pt

        │ Encoder←S1, Bottleneck+Decoder←S2
        ▼

Stage 3  完整 U-Net 联合重建
  Encoder ← Stage 1
  Bottleneck/Decoder ← Stage 2
  ReconHead ← random
  保存 stage3_best.pt

        │ Encoder+Bottleneck+Decoder → 下游
        ▼

Downstream Segmentation
  SegHead ← random（不迁移 ReconHead）
  策略: full finetune / freeze encoder / linear probe
```

### 参数继承矩阵

| 模块 | Stage 1 | Stage 2 | Stage 3 |
|------|---------|---------|---------|
| Encoder | random | random（不继承 S1） | ← Stage 1 |
| Bottleneck | random | ← Stage 1 | ← Stage 2（默认） |
| Decoder | random | random | ← Stage 2 |
| Reconstruction Head | random | random | random |
| Segmentation Head | 不参与 | 不参与 | 不参与 |

## 目录结构

```text
configs/          # Hydra 配置（data / model / pretrain / downstream / experiments）
src/stagerecon/   # 核心包
  data/           # 数据集、在线 corruption、WebDataset
  models/         # 模块化 U-Net（blocks / encoders / bottlenecks / decoders / heads）
  objectives/     # 重建与分割损失
  training/       # StageSpec、ParameterController、Trainer、CheckpointManager
  evaluation/     # 分割指标与 HD95
  experiments/    # Pipeline / Ablation / Seed 编排
  utils/          # 配置、设备、日志、校验
scripts/          # CLI 入口
tests/            # pytest（CPU smoke + 单元测试）
outputs/          # 运行输出（gitignored，保留 .gitkeep）
```

## 安装

需要 Python 3.10+。

```bash
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu   # 或 CUDA wheel
pip install -r requirements-dev.txt
pip install -e .
```

可选依赖：

- `pip install -e ".[ssim]"` — SSIM 重建损失（`pytorch-msssim`）  
- `pip install -e ".[nifti]"` — NIfTI 读取（`nibabel`，分片工具可选）  
- `pip install -e ".[dicom]"` — DICOM 读取（`pydicom`，分片工具可选）

## 合成数据 Smoke Test

CPU 可运行的极小流水线（合成 64×64 数据，少量 step）：

```bash
PYTHONPATH=src python scripts/run_experiment.py --config-name experiments/smoke_test
```

或：

```bash
PYTHONPATH=src python scripts/pretrain_pipeline.py --config-name experiments/smoke_test
```

单元 / 集成测试：

```bash
pytest -q
```

## 三阶段预训练

```bash
# 完整三阶段 + 下游（推荐）
PYTHONPATH=src python scripts/run_experiment.py --config-name experiments/staged_pretrain

# 分阶段入口
PYTHONPATH=src python scripts/pretrain_stage1.py --config-name experiments/staged_pretrain
PYTHONPATH=src python scripts/pretrain_stage2.py --config-name experiments/staged_pretrain
PYTHONPATH=src python scripts/pretrain_stage3.py --config-name experiments/staged_pretrain

# 端到端重建预训练基线
PYTHONPATH=src python scripts/run_experiment.py --config-name experiments/end_to_end_pretrain

# 跳过 Stage 2 消融
PYTHONPATH=src python scripts/run_ablation.py --config-name experiments/ablation_skip_stage2
```

可通过 Hydra 覆盖输出目录与超参，例如：

```bash
PYTHONPATH=src python scripts/run_experiment.py \
  --config-name experiments/staged_pretrain \
  paths.output_dir=outputs/my_run \
  trainer.max_epochs=50
```

## 下游分割训练

```bash
# 随机初始化基线
PYTHONPATH=src python scripts/train_segmentation.py --config-name experiments/random_init_baseline

# 使用 staged_pretrain 产生的 Stage3 权重做全量微调（需先完成预训练）
PYTHONPATH=src python scripts/train_segmentation.py \
  --config-name experiments/staged_pretrain \
  experiment.stages=[downstream]
```

下游冻结策略配置：

- `configs/downstream/finetune_full.yaml` — 训练 Encoder+Bottleneck+Decoder+SegHead  
- `configs/downstream/freeze_encoder.yaml` — 冻结 Encoder  
- `configs/downstream/linear_probe.yaml` — 仅训练 SegHead  

在实验 YAML 的 `stages.downstream` 中组合这些策略字段。

## WebDataset 流式读取

实现目录：`src/stagerecon/data/streaming/`

| 文件 | 职责 |
|------|------|
| `shard_url_builder.py` | brace 展开；`s3://` / `gs://` / HTTP / `pipe:` 规范化 |
| `sample_decoders.py` | `.npy` / `.json` 解码为 tensor + metadata |
| `webdataset_factory.py` | 构建可迭代数据集（shuffle / cache / steps_per_epoch） |
| `error_handlers.py` | `warn_and_continue` / `reraise` |

配置目录：`configs/data_source/`

- `webdataset_local.yaml` — 本地 TAR  
- `webdataset_s3.yaml` — S3（默认改写为 `pipe:aws s3 cp ... -`）  
- `webdataset_http.yaml` — HTTP(S)  

Hydra 会把 `data`（语义）与 `data_source`（传输）合并：当 `data_source.type` 为 webdataset 时，自动路由到流式工厂，**不依赖** `len(dataset)`，用 `steps_per_epoch` / `val_steps` 控制 epoch 长度。

### 样本格式

```text
__key__
image.npy
mask.npy          # 分割任务；重建预训练可省略
meta.json
```

```json
{"sample_id": "case_0001", "split": "train"}
```

### 创建 shards

从 manifest CSV（列：`sample_id,image_path,mask_path,split`）生成：

```bash
PYTHONPATH=src python3 scripts/create_webdataset_shards.py \
  --manifest path/to/manifest.csv \
  --output-dir path/to/shards \
  --maxcount 1000 \
  --maxsize 1000000000 \
  --sha256
```

工具会：按 split 分目录写 shard（禁止混写）、检查 key 唯一性、写出 `summary.json`（可选 SHA-256）。当前完整支持 `.npy` 输入。

### 本地 shard

```bash
PYTHONPATH=src python3 scripts/pretrain_stage1.py \
  --config-name experiments/staged_pretrain \
  data_source=webdataset_local \
  'data_source.shards.train=/data/shards/train/train-{000000..000009}.tar' \
  'data_source.shards.val=/data/shards/val/val-{000000..000001}.tar' \
  data_source.cache_dir=/tmp/stagerecon_wds_cache \
  data_source.steps_per_epoch=100 \
  data_source.val_steps=20
```

### S3 shard

```bash
# 推荐：写 s3://，框架默认改写为 pipe:aws s3 cp ... -
export AWS_PROFILE=your-profile
PYTHONPATH=src python3 scripts/pretrain_stage1.py \
  --config-name experiments/staged_pretrain \
  data_source=webdataset_s3 \
  'data_source.shards=s3://BUCKET/shards/train/train-{000000..000009}.tar' \
  data_source.s3_transport=pipe_aws \
  data_source.cache_dir=/tmp/stagerecon_wds_cache
```

也可用显式 pipe（不再二次改写）：

```bash
'data_source.shards=pipe:aws s3 cp s3://BUCKET/shards/train/train-000000.tar -'
```

可选：`data_source.s3_transport=pipe_rclone` 或 `raw`（需自定义 gopen handler）。

**不要**把云密钥写进仓库或 YAML；使用环境变量 / IAM 角色 / `aws` CLI 配置。

### HTTP shard

```bash
PYTHONPATH=src python3 scripts/pretrain_stage1.py \
  --config-name experiments/staged_pretrain \
  data_source=webdataset_http \
  'data_source.shards=https://example.com/shards/train-{000000..000009}.tar'
```

## 如何新增 Encoder

1. 在 `src/stagerecon/models/encoders/` 实现类，继承 `BaseEncoder`，`forward` 返回高→低分辨率特征列表  
2. 在 `encoder_registry.py` 注册名称  
3. YAML：

```yaml
model:
  encoder:
    name: my_encoder
    channels: [32, 64, 128, 256]
```

## 如何新增 Decoder

1. 实现 `BaseDecoder`：`forward(bottleneck, skip_features) -> decoded_feature`  
2. 注册到 `decoder_registry`  
3. YAML 指定 `model.decoder.name`

## 如何新增损失函数

1. 在 `objectives/` 实现 `nn.Module`，`forward(pred, target) -> Tensor`  
2. 在 `loss_factory.py` 注册名称  
3. 阶段配置：`loss_name: my_loss`

## 如何新增实验配置

在 `configs/experiments/` 新建 YAML，用 `defaults` 组合 `base` / `data` / `model`，并在 `stages` 中声明各阶段的：

- `module_initialization`  
- `trainable_modules` / `frozen_modules`  
- `forward_mode` / `loss_name` / `checkpoint_output`

参考 `smoke_test.yaml` 与 `staged_pretrain.yaml`。

## 检查点迁移说明

检查点按**模块**保存：

```python
{
  "stage": "...",
  "model": {
    "encoder": state_dict,
    "bottleneck": state_dict,
    "decoder": state_dict,
    "reconstruction_head": state_dict,
    ...
  },
  "optimizer": ...,
  "epoch": ...,
  "best_metric": ...,
  "config": ...,
  "seed": ...,
}
```

阶段初始化顺序：

1. 构建完整 `ModularUNet`（随机初始化）  
2. `CheckpointManager.initialize_modules(...)` 按模块覆盖  
3. `ParameterController` 设置冻结  
4. 仅将 `requires_grad=True` 的参数交给优化器  

禁止对阶段迁移执行整模 `model.load_state_dict(ckpt["model"])`（扁平全量加载会被拒绝）。

Stage 3 示例：

```yaml
module_initialization:
  encoder:
    source: checkpoint
    checkpoint_path: ${paths.stage1_checkpoint}
    source_module: encoder
  bottleneck:
    source: checkpoint
    checkpoint_path: ${paths.stage2_checkpoint}
    source_module: bottleneck
  decoder:
    source: checkpoint
    checkpoint_path: ${paths.stage2_checkpoint}
    source_module: decoder
  reconstruction_head:
    source: random
```

## 指标定义与 HD95 空 mask 策略

二分类指标（logits → sigmoid → 阈值 0.5；混淆矩阵固定 `labels=[0,1]`）：

- Accuracy / Sensitivity(Recall) / Specificity / Precision / Dice(F1)  
- **Foreground IoU** = `TP / (TP + FP + FN)`  
- Background IoU  
- **mIoU** = `(foreground IoU + background IoU) / 2`

HD95：

- 基于边界点的**双向最近表面距离**（距离变换或 KDTree）  
- 对两侧距离合并后取 95% 分位  
- **禁止**对完整两两距离矩阵 flatten 后取分位  
- 两侧皆空 → `0.0`  
- 单侧为空 → 默认 `NaN`（可配置替代值）  
- 聚合使用 `nanmean`

## 当前实现限制

- 标准二维 Modular U-Net、三阶段流程、合成数据 smoke test 为完整支持路径  
- Attention U-Net / Residual U-Net / 3D U-Net：提供可实例化结构与配置，但未做大规模真实数据验证  
- BraTS / ISIC 等配置为占位语义配置，需用户提供本地或 WebDataset 路径  
- NIfTI / DICOM 读取、SSIM 损失为可选依赖，默认未接入训练主路径  
- 分布式多 GPU：提供基础 stub，未实现完整 DDP 训练循环  
- 默认不包含对比学习损失  

## 许可证

MIT — 见 [LICENSE](LICENSE)。
