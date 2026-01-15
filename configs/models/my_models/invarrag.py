"""
Invar-RAG 模型配置
"""

from opencompass.models import InvarRAGModel
import torch


# Invar-RAG 模型配置
models = [
    dict(
        type=InvarRAGModel,
        abbr='invarrag_llama2_7b',
        
        # 检索器参数
        retriever_checkpoint=None,  # 训练后的检索器检查点路径
        llama_path='meta-llama/Llama-2-7b-hf',
        minilm_path='microsoft/MiniLM-L12-H384-uncased',
        corpus_path=None,  # 文档语料库路径
        top_k=5,
        use_retrieval=True,
        
        # 生成器参数
        generator_path='meta-llama/Llama-2-7b-hf',
        generator_checkpoint=None,  # 训练后的生成器检查点路径
        
        # LoRA 参数
        use_lora=True,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        hidden_dim=768,
        
        # 生成参数
        max_seq_len=2048,
        max_out_len=128,
        batch_padding=True,
        device='auto',
        
        generation_kwargs=dict(
            do_sample=False,
            temperature=1.0,
            top_p=0.9,
        ),
        
        # 运行配置
        batch_size=4,
        run_cfg=dict(num_gpus=1),
    )
]

