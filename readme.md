# PSPNet-Lightning

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.google.com/url?sa=E&q=https%3A%2F%2Fwww.python.org%2F)  
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9-ee4c2c.svg)](https://www.google.com/url?sa=E&q=https%3A%2F%2Fpytorch.org%2F)  
[![Lightning](https://img.shields.io/badge/Lightning-2.6-792ee5.svg)](https://www.google.com/url?sa=E&q=https%3A%2F%2Flightning.ai%2F)  
[![Hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd.svg)](https://www.google.com/url?sa=E&q=https%3A%2F%2Fhydra.cc%2F)

---

<a name="english"></a>

## 🇬🇧 English

### Introduction

**PSPNet-Lightning** is a semantic segmentation project based on the **PyTorch Lightning** framework and **Hydra** configuration management system. It reproduces the classic **PSPNet** (Pyramid Scene Parsing Network) architecture with a **ResNet50** backbone, specifically optimized for the **Oxford-IIIT Pet** dataset.

> **Note**: This project was developed as part of the author's deep learning study journey. While it aims to implement best practices, there may be imperfections or areas for improvement. Feedback and suggestions are welcome.

### 📂 Project Structure

```text
Project_Root
├── conf/                # Hydra configurations (dataset, model, config, etc.)
├── dataloader/          # Data loading, Preprocessing & Augmentation
├── model/               # PSPNet implementation & LightningModule wrapper
├── scripts/             # Entry points for Training & Testing
├── utils/               # Callbacks (Visualization) & Utilities
└── logs/                # Training logs and Checkpoints
```

### 🚀 Getting Started

#### 1. Requirements

- **Python**: 3.10.19
- **CUDA**: 12.8

Install dependencies using `requirements.txt`:

```bash
pip install -r requirements.txt
```

_Key Libraries Versions:_

- `pytorch-lightning`: 2.6.0
- `torch`: 2.9.1
- `torchvision`: 0.24.1
- `hydra-core`: 1.3.2
- `torchmetrics`: 1.8.2

#### 2. Data Preparation

This project uses the [Oxford-IIIT Pet Dataset](https://www.google.com/url?sa=E&q=https%3A%2F%2Fwww.robots.ox.ac.uk%2F~vgg%2Fdata%2Fpets%2F).

1. Download the dataset (images and trimaps).
2. Randomly split the dataset into **Train**, **Validation**, and **Test** sets with a ratio of **8:1:1**.
3. Organize your directory as follows:
    
    ```text
    data_split/
    ├── train/
    │   ├── images/  (.jpg)
    │   └── trimaps/ (.png)
    ├── val/
    │   ├── images/
    │   └── trimaps/
    └── test/
        ├── images/
        └── trimaps/
    ```
    
4. Update the path in `conf/dataset/pet.yaml` if necessary.

#### 3. Training

Run the training script directly. Configuration is managed by `conf/config.yaml`.

```bash
# Standard training (ResNet50 + SGD + Poly Scheduler)
python scripts/train.py

# Override parameters via command line (Hydra syntax)
python scripts/train.py optimizer.lr=0.005 trainer.max_epochs=50
```

Logs and checkpoints will be saved to `logs/OxfordPet/PSPNet_ResNet50/...`.

#### 4. Testing

Run testing with a specific checkpoint. Visualization results will be saved automatically.

```bash
python scripts/test.py ckpt_path="/path/to/your/best_model.ckpt"
```

### 📊 Features

- **Visualization**: Automatically saves segmentation overlay masks during testing.
- **Logging**: Integrated **TensorBoard** support for tracking Loss, mIoU, and Pixel Accuracy.
- **Optimization**:
    - Supports `bf16-mixed` precision.
    - Optimized with `torch.set_float32_matmul_precision('medium')`.
    - Implements **Poly Learning Rate Scheduler**.

### 🙏 Acknowledgements

- **Guidance**: This project was completed under the guidance of **@Chandery**.
- Original Paper: [Pyramid Scene Parsing Network (CVPR 2017)](https://www.google.com/url?sa=E&q=https%3A%2F%2Farxiv.org%2Fabs%2F1612.01105)

---

<a name="chinese"></a>

## 🇨🇳 中文

### 项目简介

**PSPNet-Lightning** 是一个基于 **PyTorch Lightning** 框架和 **Hydra** 配置管理系统的语义分割项目。本项目复现了经典的 **PSPNet** (Pyramid Scene Parsing Network) 架构（使用 **ResNet50** 主干），并针对 **Oxford-IIIT Pet** 宠物数据集进行了优化。

> **说明**：本项目是作者在深入学习深度学习过程中的实践作品。虽然力求规范，但难免存在不完善之处，恳请批评指正。

### 📂 项目结构

```text
Project_Root
├── conf/                # Hydra 配置文件 (dataset, model, config 等)
├── dataloader/          # 数据加载、预处理与增强
├── model/               # PSPNet 网络实现与 Lightning 封装
├── scripts/             # 训练与测试脚本入口
├── utils/               # 回调函数 (可视化) 与工具
└── logs/                # 训练日志与模型权重
```

### 🚀 快速开始

#### 1. 环境要求

- **Python**: 3.10.19
- **CUDA**: 12.8

使用 `requirements.txt` 安装依赖：

```bash
pip install -r requirements.txt
```

_关键库版本：_

- `pytorch-lightning`: 2.6.0
- `torch`: 2.9.1
- `torchvision`: 0.24.1
- `hydra-core`: 1.3.2
- `torchmetrics`: 1.8.2

#### 2. 数据准备

本项目使用 [Oxford-IIIT Pet 数据集](https://www.google.com/url?sa=E&q=https%3A%2F%2Fwww.robots.ox.ac.uk%2F~vgg%2Fdata%2Fpets%2F)。

1. 下载数据集（包含 images 和 trimaps）。
2. 按照 **8:1:1** 的比例将数据随机划分为 `train` (训练集), `val` (验证集), `test` (测试集)。
3. 整理目录结构如下：
    
    ```text
    data_split/
    ├── train/
    │   ├── images/  (.jpg)
    │   └── trimaps/ (.png)
    ├── val/
    │   ├── images/
    │   └── trimaps/
    └── test/
        ├── images/
        └── trimaps/
    ```
    
4. 如有需要，请修改 `conf/dataset/pet.yaml` 中的路径配置。

#### 3. 训练

直接运行训练脚本，配置由 `conf/config.yaml` 统一管理。

```bash
# 标准训练 (ResNet50 + SGD + Poly 调度器)
python scripts/train.py

# 通过命令行覆盖参数 (Hydra 语法)
python scripts/train.py optimizer.lr=0.005 trainer.max_epochs=50
```

日志和模型权重将自动保存到 `logs/OxfordPet/PSPNet_ResNet50/...` 目录下。

#### 4. 测试

指定权重文件进行测试，分割的可视化结果将自动保存。

```bash
python scripts/test.py ckpt_path="/path/to/your/best_model.ckpt"
```

### 📊 特性

- **可视化**：测试过程中自动保存分割结果（Mask 叠加图）。
- **日志记录**：集成 **TensorBoard**，实时追踪 Loss, mIoU 和 Pixel Accuracy。
- **性能优化**：
    - 支持 `bf16-mixed` 混合精度训练。
    - 使用 `torch.set_float32_matmul_precision('medium')` 加速。
    - 实现了 **Poly 学习率衰减策略**。

### 🙏 致谢

- **指导**：本项目是在 **@Chandery** 的悉心指导下完成的。
- 原论文：[Pyramid Scene Parsing Network (CVPR 2017)](https://www.google.com/url?sa=E&q=https%3A%2F%2Farxiv.org%2Fabs%2F1612.01105)