"""Held-out test-set evaluation."""
import os
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from knowledge_aug_gat import QM9_HOMO_LUMO_INDICES, MultiTaskMSELoss, load_qm9_dataset, KnowledgeAugGAT

def test(args):
    """测试模型"""
    if not args.checkpoint or not os.path.exists(args.checkpoint):
        raise ValueError("测试模式必须指定有效检查点路径")

    print("=== 开始测试 ===")
    dataset = load_qm9_dataset(root=args.data_root, image_size=args.image_size)
    test_dataset = dataset[11000:12000]  # 简化：取1000个测试样本
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    node_input_dim = dataset[0].x.size(-1)

    # 加载模型
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

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["model"])
    criterion = MultiTaskMSELoss()

    # 测试
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = batch.to(args.device)
            pred = model(batch)
            loss = criterion(pred, batch.y)
            test_loss += loss.item() * batch.num_graphs

    test_loss /= len(test_dataset)
    print(f"测试平均损失：{test_loss:.4f}")
    print("=== 测试完成 ===")
