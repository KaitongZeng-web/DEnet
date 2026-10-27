"""CLI entry point for training, testing and inference."""
import argparse
import torch
from engine.train import smoke_test, train
from evaluation.test import evaluate
from inference.predict import predict

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Knowledge-Augmented GAT for QM9")
    # 通用参数
    parser.add_argument("--data-root", default="./data/QM9", help="QM9数据集根目录")
    parser.add_argument("--image-size", type=int, default=128, help="分子图像尺寸")
    parser.add_argument("--hidden-dim", type=int, default=128, help="隐藏层维度")
    parser.add_argument("--batch-size", type=int, default=32, help="批次大小")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout概率")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="设备")
    parser.add_argument("--checkpoint", default=None, help="模型检查点路径")

    # 模式选择
    parser.add_argument("--smoke-test", action="store_true", help="快速前向/反向传播测试")
    parser.add_argument("--train", action="store_true", help="训练模式")
    parser.add_argument("--test", action="store_true", help="测试模式")
    parser.add_argument("--predict", action="store_true", help="单SMILES推理模式")
    parser.add_argument("--smiles", default="CCO", help="推理用SMILES")

    # 模型参数
    parser.add_argument("--num-attention-heads", type=int, default=4, help="MHA头数")
    parser.add_argument("--num-gcn-layers", type=int, default=2, help="GCN层数")
    parser.add_argument("--num-gat-layers", type=int, default=2, help="GAT层数")
    parser.add_argument("--gat-heads", type=int, default=4, help="GAT头数")

    return parser.parse_args()

def main():
    args = parse_args()
    torch.backends.cudnn.enabled = False
    if args.smoke_test:
        smoke_test(args)
    elif args.train:
        train(args)
    elif args.test:
        evaluate(args)
    elif args.predict:
        predict(args)
    else:
        print("请指定运行模式：--smoke-test / --train / --test / --predict")

if __name__ == "__main__":
    main()
