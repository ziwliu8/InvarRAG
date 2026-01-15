"""
Invar-RAG NQ 单独评估配置
仅评估 Natural Questions 数据集
"""

from mmengine.config import read_base

with read_base():
    # 导入模型配置
    from ..models.my_models.invarrag import models
    
    # 导入数据集配置
    from ..datasets.my_datasets.invarrag_nq import invarrag_nq_datasets


# 只使用 NQ 数据集
datasets = [*invarrag_nq_datasets]

