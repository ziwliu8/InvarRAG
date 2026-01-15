"""
Invar-RAG 完整评估配置
评估 NQ, TriviaQA, PopQA 三个数据集
"""

from mmengine.config import read_base

with read_base():
    # 导入模型配置
    from ..models.my_models.invarrag import models
    
    # 导入数据集配置
    from ..datasets.my_datasets.invarrag_nq import invarrag_nq_datasets
    from ..datasets.my_datasets.invarrag_triviaqa import invarrag_tqa_datasets
    from ..datasets.my_datasets.invarrag_popqa import invarrag_popqa_datasets


# 组合所有数据集
datasets = [
    *invarrag_nq_datasets,
    *invarrag_tqa_datasets,
    *invarrag_popqa_datasets,
]

