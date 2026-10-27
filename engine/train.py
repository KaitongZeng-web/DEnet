"""Training loop and quick smoke test."""
import os
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from knowledge_aug_gat import QM9_HOMO_LUMO_INDICES, MultiTaskMSELoss, load_qm9_dataset, KnowledgeAugGAT

def smoke_test(args):
    """快速前向/反向传播测试"""
    print("=== 烟雾测试：前向+反向传播 ===")
    # 加载少量数据
    dataset = load_qm9_dataset(root=args.data_root, image_size=args.image_size)
    dataloader = DataLoader(dataset[:10], batch_size=2, shuffle=True)
    node_input_dim = dataset[0].x.size(-1)

    # 初始化模型
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

    # 损失函数 + 优化器
    criterion = MultiTaskMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 前向+反向
    model.train()
    for batch in dataloader:
        batch = batch.to(args.device)
        pred = model(batch)
        loss = criterion(pred, batch.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Smoke test loss: {loss.item():.4f}")
        break
    print("=== 烟雾测试完成 ===")

def train(args):
    """训练模型"""
    print("=== 开始训练 ===")
    # 加载数据集
    dataset = load_qm9_dataset(root=args.data_root, image_size=args.image_size)
    train_dataset = dataset[:10000]  # 简化：取前10000个样本
    val_dataset = dataset[10000:11000]
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    node_input_dim = dataset[0].x.size(-1)

    # 初始化模型/损失/优化器
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

    criterion = MultiTaskMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # 加载检查点
    best_val_loss = float("inf")
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        best_val_loss = ckpt["val_loss"]
        print(f"加载检查点：最佳验证损失 {best_val_loss:.4f}")

    # 训练循环
    os.makedirs("./checkpoints", exist_ok=True)
    for epoch in range(args.epochs):
        # 训练
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} (Train)"):
            batch = batch.to(args.device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, batch.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} (Val)"):
                batch = batch.to(args.device)
                pred = model(batch)
                loss = criterion(pred, batch.y)
                val_loss += loss.item() * batch.num_graphs

        # 平均损失
        train_loss /= len(train_dataset)
        val_loss /= len(val_dataset)
        scheduler.step()

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "epoch": epoch,
            }, "./checkpoints/best.pt")
            print(f"保存最佳模型：验证损失 {best_val_loss:.4f}")

    print("=== 训练完成 ===")
