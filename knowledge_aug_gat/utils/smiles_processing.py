"""SMILES处理工具（转图像、推理数据构建）"""
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from torch_geometric.data import Data


def smiles_to_rdkit_image_tensor(
    smiles: str,
    image_size: int = 128,
    add_hs: bool = False,
    normalize: bool = True,
) -> torch.Tensor:
    """
    将 SMILES 字符串转为 RDKit 2D 分子结构图像，再转为 PyTorch Tensor。
    返回：img_tensor: shape = [3, image_size, image_size]
    """
    if not isinstance(smiles, str) or len(smiles.strip()) == 0:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    # 关闭严格sanitize，兼容QM9特殊环编号SMILES
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    # 解析失败返回空白全黑图像
    if mol is None:
        return torch.zeros(3, 128, 128, dtype=torch.float32)

    # 基础清洗（避免价态报错）
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_FIND_RINGS)
    except:
        pass

    # 加氢 + 生成2D坐标
    if add_hs:
        mol = Chem.AddHs(mol)
    AllChem.Compute2DCoords(mol)

    # 生成PIL图像并转换为Tensor
    pil_img = Draw.MolToImage(mol, size=(image_size, image_size)).convert("RGB")
    transform_steps = [T.Resize((image_size, image_size)), T.ToTensor()]
    
    if normalize:
        transform_steps.append(
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

    transform = T.Compose(transform_steps)
    return transform(pil_img).float()


def _parse_smiles_for_inference(smiles: str) -> "Chem.Mol":
    """仅用于推理的严格 SMILES 解析"""
    if not isinstance(smiles, str) or len(smiles.strip()) == 0:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as exc:
        raise ValueError(f"RDKit cannot parse SMILES: {smiles!r}") from exc

    if mol is None:
        raise ValueError(f"RDKit cannot parse SMILES: {smiles!r}")
    return mol


def _one_hot_atomic_number(z: torch.Tensor, max_atomic_num: int = 100) -> torch.Tensor:
    """用原子序数构造one-hot节点特征"""
    z = z.long().clamp(min=0, max=max_atomic_num)
    return F.one_hot(z, num_classes=max_atomic_num + 1).float()


def _rdkit_atom_feature_vector(atom: "Chem.Atom") -> List[float]:
    """构建11维原子基础特征（兼容QM9推理）"""
    symbol = atom.GetSymbol()
    hybridization = atom.GetHybridization()

    return [
        float(symbol == "H"),
        float(symbol == "C"),
        float(symbol == "N"),
        float(symbol == "O"),
        float(symbol == "F"),
        float(atom.GetAtomicNum()),
        float(atom.GetIsAromatic()),
        float(hybridization == Chem.HybridizationType.SP),
        float(hybridization == Chem.HybridizationType.SP2),
        float(hybridization == Chem.HybridizationType.SP3),
        float(atom.GetTotalNumHs(includeNeighbors=True)),
    ]


def _build_inference_node_features(
    mol: Chem.Mol,
    base_dim: int,
) -> torch.Tensor:
    """生成与训练兼容的推理节点特征"""
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return torch.zeros((0, max(base_dim, 0)), dtype=torch.float32)

    # one-hot原子序数（base_dim=101）
    if base_dim == 101:
        return torch.stack(
            [_one_hot_atomic_number(torch.tensor(atom.GetAtomicNum()), max_atomic_num=100)
             for atom in mol.GetAtoms()], dim=0
        ).float()

    # 基础特征（11维）+ 补零/截断到base_dim
    base_features = torch.tensor(
        [_rdkit_atom_feature_vector(atom) for atom in mol.GetAtoms()], dtype=torch.float32
    )
    if base_features.size(-1) < base_dim:
        padding = torch.zeros(num_atoms, base_dim - base_features.size(-1), dtype=base_features.dtype)
        base_features = torch.cat([base_features, padding], dim=-1)
    elif base_features.size(-1) > base_dim:
        base_features = base_features[:, :base_dim]

    return base_features


def _rdkit_bond_feature_vector(bond: Chem.Bond) -> List[float]:
    """4维键特征向量"""
    bond_type = bond.GetBondType()
    features = [0.0, 0.0, 0.0, 0.0]
    if bond_type == Chem.BondType.SINGLE:
        features[0] = 1.0
    elif bond_type == Chem.BondType.DOUBLE:
        features[1] = 1.0
    elif bond_type == Chem.BondType.TRIPLE:
        features[2] = 1.0
    elif bond_type == Chem.BondType.AROMATIC:
        features[3] = 1.0
    else:
        features[0] = 1.0
    return features


def _fit_edge_attr_dim(edge_attr: torch.Tensor, edge_dim: Optional[int]) -> torch.Tensor:
    """对齐边特征维度"""
    if edge_dim is None or edge_attr.size(-1) == edge_dim:
        return edge_attr
    if edge_attr.size(-1) < edge_dim:
        padding = torch.zeros(edge_attr.size(0), edge_dim - edge_attr.size(-1), dtype=edge_attr.dtype)
        return torch.cat([edge_attr, padding], dim=-1)
    return edge_attr[:, :edge_dim]


def _build_inference_edges(
    mol: Chem.Mol,
    edge_dim: Optional[int] = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """构建推理用的边索引和边特征"""
    rows: List[int] = []
    cols: List[int] = []
    edge_features: List[List[float]] = []

    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        bond_feat = _rdkit_bond_feature_vector(bond)
        rows.extend([begin_idx, end_idx])
        cols.extend([end_idx, begin_idx])
        edge_features.extend([bond_feat, bond_feat])

    if not rows:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 4), dtype=torch.float32)
    else:
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)

    return edge_index, _fit_edge_attr_dim(edge_attr, edge_dim)


def _build_inference_positions(mol: Chem.Mol) -> torch.Tensor:
    """构建推理用3D坐标（失败则返回全零）"""
    mol_with_hs = Chem.Mol(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42

    try:
        status = AllChem.EmbedMolecule(mol_with_hs, params)
    except Exception:
        status = -1

    if status != 0:
        return torch.zeros((mol_with_hs.GetNumAtoms(), 3), dtype=torch.float32)

    try:
        AllChem.UFFOptimizeMolecule(mol_with_hs)
    except Exception:
        pass

    conformer = mol_with_hs.GetConformer()
    positions = torch.tensor(conformer.GetPositions(), dtype=torch.float32)
    return positions


def smiles_to_inference_data(
    smiles: str,
    node_input_dim: int,
    image_size: int,
    edge_dim: Optional[int] = None,
    normalize_img: bool = True,
) -> Data:
    """将单个SMILES转为可输入模型的PyG Data对象"""
    mol = _parse_smiles_for_inference(smiles)
    mol = Chem.AddHs(mol)

    # 构建节点特征（含位置编码）
    base_dim = max(int(node_input_dim) - 4, 0)
    base_x = _build_inference_node_features(mol, base_dim=base_dim)
    pos = _build_inference_positions(mol)
    edge_index, edge_attr = _build_inference_edges(mol, edge_dim=edge_dim)

    # 导入qm9_utils中的位置编码函数
    from .qm9_utils import build_node_features_with_position_encoding
    data = Data(x=base_x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)
    data.x = build_node_features_with_position_encoding(data)
    
    # 构建图像特征
    data.img = smiles_to_rdkit_image_tensor(
        smiles, image_size=image_size, normalize=normalize_img
    ).unsqueeze(0)
    data.smiles = smiles
    return data
