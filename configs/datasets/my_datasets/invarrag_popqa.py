"""
Invar-RAG PopQA 数据集配置
"""

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import InvarRAGPopQADataset, InvarRAGEvaluator


# 数据集读取配置
invarrag_popqa_reader_cfg = dict(
    input_columns=['question'],
    output_column='answer'
)

# 推理配置
invarrag_popqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='Question: {question}\nAnswer: '
                ),
            ],
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

# 评估配置
invarrag_popqa_eval_cfg = dict(
    evaluator=dict(type=InvarRAGEvaluator),
    pred_role='BOT'
)

# 数据集配置列表
invarrag_popqa_datasets = [
    dict(
        type=InvarRAGPopQADataset,
        abbr='invarrag_popqa',
        path=None,  # 将使用默认路径
        max_samples=None,  # 使用全部数据
        reader_cfg=invarrag_popqa_reader_cfg,
        infer_cfg=invarrag_popqa_infer_cfg,
        eval_cfg=invarrag_popqa_eval_cfg
    )
]

