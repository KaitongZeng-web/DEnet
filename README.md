# DEnet
<img width="1918" height="963" alt="image" src="https://github.com/user-attachments/assets/44b5e570-42b3-437e-960c-8f766c46742c" />

## 项目介绍
本项目实现**知识增强图注意力网络（Knowledge-Augmented Graph Attention Network）**，基于PyTorch、PyTorch Geometric、RDKit 搭建，专门针对 QM9 分子数据集完成 HOMO、LUMO 轨道能级多任务回归预测。

模型采用**双分支特征提取架构**，同时融合分子拓扑结构信息与化学先验图像信息，弥补单一图网络仅依靠原子连接关系的特征缺陷：
1. 拓扑图分支：以分子原子3D坐标构建自定义位置编码Pm，拼接原子特征后送入多头QKV自注意力层捕捉全局原子关联；后续堆叠多层GCN提取基础拓扑特征，再通过一维卷积重构节点表征；最后使用带残差连接的多层GATConv学习动态可微分边注意力，聚合得到分子全局图特征。
2. 分子图像分支：利用RDKit将SMILES转换标准化2D分子结构图，通过多层CNN提取视觉化学特征，经全局池化得到图像表征。

两支特征通过**GATE自适应门控融合模块**完成动态加权融合，门控权重由网络自主学习，自动平衡拓扑结构与2D图像先验的贡献比例；融合后的特征送入多层全连接多任务预测头，输出分子HOMO、LUMO预测结果。

项目完整配套数据加载、训练、验证、测试、单分子推理全流程代码，支持导出训练损失曲线、预测结果CSV，同时开放多头自注意力权重、GAT边注意力权重、门控融合权重提取接口，方便开展模型可解释性分析。代码做模块化拆分，区分模型、工具、损失、常量文件，结构清晰易读，适合深度学习入门、分子图学习相关研究复现与二次开发。

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

