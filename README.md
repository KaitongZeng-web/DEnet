#DEnet
**KnowledgeAugGAT
<img width="1918" height="963" alt="image" src="https://github.com/user-attachments/assets/44b5e570-42b3-437e-960c-8f766c46742c" />

## 项目介绍
本项目实现**知识增强图注意力网络（Knowledge-Augmented Graph Attention Network）**，基于 PyTorch、PyTorch Geometric、RDKit 搭建，专门针对 QM9 分子数据集完成 HOMO、LUMO 轨道能级多任务回归预测。

模型采用**双分支特征提取架构**，同时融合分子拓扑结构信息与化学先验图像信息，弥补单一图网络仅依靠原子连接关系的特征缺陷：
1. 拓扑图分支：以分子原子3D坐标构建自定义位置编码Pm，拼接原子特征后送入多头QKV自注意力层捕捉全局原子关联；后续堆叠多层GCN提取基础拓扑特征，再通过一维卷积重构节点表征；最后使用带残差连接的多层GATConv学习动态可微分边注意力，聚合得到分子全局图特征。
2. 分子图像分支：利用RDKit将SMILES转换标准化2D分子结构图，通过多层CNN提取视觉化学特征，经全局池化得到图像表征。

两支特征通过**GATE自适应门控融合模块**完成动态加权融合，门控权重由网络自主学习，自动平衡拓扑结构与2D图像先验的贡献比例；融合后的特征送入多层全连接多任务预测头，输出分子HOMO、LUMO预测结果。

项目完整配套数据加载、训练、验证、测试、单分子推理全流程代码，支持导出训练损失曲线、预测结果CSV，同时开放多头自注意力权重、GAT边注意力权重、门控融合权重提取接口，方便开展模型可解释性分析。代码做模块化拆分，区分模型、工具、损失、常量文件，结构清晰易读，适合深度学习入门、分子图学习相关研究复现与二次开发。

## 环境依赖
```bash
pip install -r requirements.txt
```

## 运行命令
1. 环境快速校验（无需数据集）
```bash
python main.py --smoke-test
```
2. 启动模型训练
```bash
python main.py --train --epochs 10
```
3. 测试集评估模型
```bash
python main.py --test --checkpoint ./checkpoints/best.pt
```
4. 输入SMILES单分子推理预测
```bash
python main.py --predict --smiles CCO --checkpoint ./checkpoints/best.pt
```

## 项目结构
项目统一以`main.py`作为唯一运行入口，内部模块化拆分：模型编码器/融合层/预测头、QM9与SMILES处理工具、自定义损失函数、全局常量配置，逻辑分层清晰，便于修改、拓展与复用。
