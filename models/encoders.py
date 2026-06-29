"""图编码器 & 图像编码器"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from torch_geometric.utils import to_dense_batch


class MolecularGraphEncoder(nn.Module):
    """分子拓扑图编码器（分支A）"""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_attention_heads: int = 4,
        num_gcn_layers: int = 2,
        num_gat_layers: int = 2,
        gat_heads: int = 4,
        edge_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim % num_attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_attention_heads.")

        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.dropout = dropout

        # 节点特征投影
        self.node_proj = nn.Linear(input_dim, hidden_dim)
        
        # Multi-Head Self-Attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_attention_heads, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # GCN层
        self.gcn_layers = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim) for _ in range(num_gcn_layers)]
        )

        # Conv1d特征重构
        self.reconstruct_conv = nn.Conv1d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1
        )

        # 残差GAT层
        self.gat_layers = nn.ModuleList(
            [
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=gat_heads,
                    concat=False,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    add_self_loops=True,
                )
                for _ in range(num_gat_layers)
            ]
        )

        # 可解释性权重缓存
        self.latest_mha_attention_weights: Optional[torch.Tensor] = None
        self.latest_gat_attention_weights: List[Tuple[torch.Tensor, torch.Tensor]] = []

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor],
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # 节点特征投影
        x = self.node_proj(x.float())

        # Multi-Head Self-Attention（dense格式）
        dense_x, dense_mask = to_dense_batch(x, batch=batch)
        attn_out, attn_weights = self.self_attention(
            query=dense_x, key=dense_x, value=dense_x, key_padding_mask=~dense_mask,
            need_weights=True, average_attn_weights=False
        )
        self.latest_mha_attention_weights = attn_weights

        # Add & Norm
        dense_x = self.attn_norm(dense_x + F.dropout(attn_out, p=self.dropout, training=self.training))
        x = dense_x[dense_mask]

        # GCN层
        for gcn in self.gcn_layers:
            x = gcn(x, edge_index)
            x = F.relu(x)

        # Conv1d特征重构
        dense_x, dense_mask = to_dense_batch(x, batch=batch)
        conv_input = dense_x.transpose(1, 2)
        conv_output = self.reconstruct_conv(conv_input)
        conv_output = F.relu(conv_output).transpose(1, 2)
        x = conv_output[dense_mask]

        # 残差GAT层
        self.latest_gat_attention_weights = []
        for gat in self.gat_layers:
            residual = x
            if self.edge_dim is not None and edge_attr is not None:
                gat_out, attn_info = gat(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
            else:
                gat_out, attn_info = gat(x, edge_index, return_attention_weights=True)
            
            gat_out = F.relu(gat_out)
            x = residual + F.dropout(gat_out, p=self.dropout, training=self.training)
            self.latest_gat_attention_weights.append(attn_info)

        # 全局均值池化
        graph_feat = global_mean_pool(x, batch)
        return graph_feat

    def get_mha_attention_weights(self) -> Optional[torch.Tensor]:
        """返回MHA注意力权重"""
        return self.latest_mha_attention_weights

    def get_gat_attention_weights(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """返回GAT边注意力权重"""
        return self.latest_gat_attention_weights


class MolecularImageEncoder(nn.Module):
    """分子2D图像编码器（分支B）"""
    def __init__(
        self,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        # 多层CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(128, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # 全局均值池化 + 投影
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self.cnn(img.float())
        x = self.global_avg_pool(x)
        img_feat = self.proj(x)
        return img_feat
