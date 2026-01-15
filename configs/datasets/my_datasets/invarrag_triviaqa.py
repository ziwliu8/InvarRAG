"""
Invar-RAG TriviaQA 数据集配置
"""

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import InvarRAGTriviaQADataset, InvarRAGEvaluator


# 数据集读取配置
invarrag_tqa_reader_cfg = dict(
    input_columns=['question'],
    output_column='answer'
)

# 推理配置
invarrag_tqa_infer_cfg = dict(
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
invarrag_tqa_eval_cfg = dict(
    evaluator=dict(type=InvarRAGEvaluator),
    pred_role='BOT'
)

# 数据集配置列表
invarrag_tqa_datasets = [
    dict(
        type=InvarRAGTriviaQADataset,
        abbr='invarrag_tqa',
        path=None,  # 将使用默认路径
        max_samples=500,
        reader_cfg=invarrag_tqa_reader_cfg,
        infer_cfg=invarrag_tqa_infer_cfg,
        eval_cfg=invarrag_tqa_eval_cfg
    )
]

