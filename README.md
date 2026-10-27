# DEnet：模块化重构版本

本仓库将原始 [DEnet 项目](https://github.com/KaitongZeng-web/DEnet)（Knowledge-Augmented GAT，知识增强图注意力网络）整理为更清晰、常见的 Python 项目结构。

**原始仓库没有被修改。** 本次主要调整代码组织方式和模块导入路径，保留原项目的模型实现及 QM9 数据集上的 HOMO/LUMO 预测任务。

## 一、项目目录结构

```text
DEnet-refactored/
├── knowledge_aug_gat/       # 模型与数据处理核心包
│   ├── models/              # 图编码器、图像编码器、特征融合、预测头
│   ├── utils/               # QM9 数据处理、SMILES 分子字符串处理
│   ├── constants.py         # 常量定义
│   └── losses.py            # 损失函数
├── engine/
│   └── train.py             # 模型训练与快速烟雾测试
├── evaluation/
│   └── test.py              # 测试集评估
├── inference/
│   └── predict.py           # 单个 SMILES 输入推理
├── tests/                   # 基础结构和语法检查
├── configs/
│   └── default.yaml         # 示例配置
├── scripts/                 # 分析脚本
├── legacy/
│   └── DEnet_total.py       # 原始单文件实现备份
├── data/                    # 数据目录（数据文件不提交到 Git）
├── checkpoints/             # 模型权重保存目录
├── main.py                  # 命令行程序入口
├── requirements.txt         # Python 依赖
└── .gitignore               # Git 忽略规则
```

## 二、环境安装

建议使用独立的 Python 虚拟环境。PyTorch、PyTorch Geometric 和 RDKit 的安装方式可能因操作系统、Python 版本及 CUDA 版本而异。如果需要使用 GPU，请先按照本机环境安装匹配的 PyTorch 版本。

安装项目依赖：

```bash
python -m pip install -r requirements.txt
```

如果安装 PyTorch Geometric 或 RDKit 时遇到兼容性问题，请优先参考相应项目的官方安装说明，并确认 Python 与 PyTorch 版本兼容。

## 三、运行方式

请在仓库根目录执行以下命令。

### 1. 快速烟雾测试

尝试执行少量数据上的前向传播、损失计算和反向传播：

```bash
python main.py --smoke-test
```

首次运行时，程序可能需要下载或处理 QM9 数据集。此测试需要完整安装相关依赖。

### 2. 训练模型

```bash
python main.py --train --epochs 100 --batch-size 32
```

训练轮数可通过 `--epochs` 调整，批次大小可通过 `--batch-size` 调整。数据默认存放在 `./data/QM9`，模型检查点默认保存到 `./checkpoints/best.pt`（具体保存行为以训练代码为准）。

### 3. 测试模型

使用已经训练好的检查点进行测试：

```bash
python main.py --test --checkpoint ./checkpoints/best.pt
```

执行前请确认指定的模型权重文件存在，并且与当前模型结构及参数配置兼容。

### 4. 输入 SMILES 进行预测

例如，对乙醇的 SMILES 字符串 `CCO` 进行推理：

```bash
python main.py --predict --checkpoint ./checkpoints/best.pt --smiles CCO
```

程序会根据 SMILES 构造模型输入，并输出对应的预测结果。预测前请确保模型权重文件有效。

### 5. 运行基础检查

```bash
python -m unittest discover -s tests -v
```

当前测试主要检查项目关键文件是否存在，以及 Python 文件是否能够通过语法解析；它**不等同于**完整的模型训练测试或预测精度验证。

## 四、重要说明

1. **数据集划分：** 当前训练代码采用固定顺序划分：前 10,000 个样本用于训练，接下来的 1,000 个用于验证，再接下来的 1,000 个用于测试。这沿用了原脚本的简化做法，不代表随机划分或标准基准划分。
2. **推理兼容性：** SMILES 推理代码会重新构建输入特征。使用已有权重前，应确认输入特征维度、模型超参数和检查点与训练时保持一致。
3. **验证状态：** 本次重构完成了代码文件整理，并添加了基础结构检查，但尚未在此环境中完成全量模型训练、烟雾测试和预测结果验证。因此，不能将本次整理视为所有运行环节均已验证通过。
4. **分析脚本：** 原有的 LUMO 与 Yield 相关性绘图脚本保存在 `scripts/plot_LUMO_Yield_Correlation.py`。该脚本还需要 pandas、SciPy 和 matplotlib 等依赖。
5. **数据与权重：** 数据集和模型权重文件通常体积较大，已通过 `.gitignore` 规则排除，避免直接提交到 Git 仓库。

## 五、目录用途速查

| 目录或文件 | 用途 |
|---|---|
| `knowledge_aug_gat/models/` | 定义模型结构 |
| `knowledge_aug_gat/utils/` | 数据读取与特征处理 |
| `knowledge_aug_gat/losses.py` | 定义训练损失函数 |
| `engine/train.py` | 训练模型 |
| `evaluation/test.py` | 评估模型表现 |
| `inference/predict.py` | 使用训练好的模型进行推理 |
| `tests/` | 检查代码结构和语法 |
| `configs/` | 保存示例配置 |
| `scripts/` | 存放独立的数据分析脚本 |
| `legacy/` | 保留原始单文件版本以便对照 |
| `main.py` | 统一的命令行入口 |

