"""Single-SMILES inference."""
import os
import torch
from knowledge_aug_gat import QM9_HOMO_LUMO_INDICES, QM9_TARGET_NAMES, KnowledgeAugGAT, smiles_to_inference_data

def predict(args):
    """单SMILES推理"""
    if not args.checkpoint or not os.path.exists(args.checkpoint):
        raise ValueError("推理模式必须指定有效检查点路径")

    print(f"=== 推理 SMILES: {args.smiles} ===")
    # 加载模型（需要先获取node_input_dim，这里简化为105=101+4）
    node_input_dim = 105  # QM9节点特征(101) + 位置编码(4)
    model = KnowledgeAugGAT(
        node_input_dim=node_input_dim,
        hidden_dim=args.hidden_dim,
        num_tasks=len(QM9_HOMO_LUMO_INDICES),
        num_attention_heads=args.num_attention_heads,
        num_gcn_layers=args.num_gcn_layers,
        num_gat_layers=args.num_gat_layers,
        gat_heads=args.gat_heads,
        dropout=args.dropout,
    ).to(args.device)

    # 加载权重
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["model"])

    # 构建推理数据
    data = smiles_to_inference_data(
        smiles=args.smiles,
        node_input_dim=node_input_dim,
        image_size=args.image_size,
    ).to(args.device)

    # 推理
    model.eval()
    with torch.no_grad():
        pred = model(data)
        pred = pred.cpu().numpy().flatten()

    # 输出结果
    for name, value in zip(QM9_TARGET_NAMES, pred):
        print(f"{name.upper()}: {value:.4f}")
    print("=== 推理完成 ===")
