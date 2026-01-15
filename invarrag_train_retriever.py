"""
Invar-RAG 检索器训练脚本
用于训练双编码器检索模型，包含对齐损失和不变性损失
"""

import os
import json
import argparse
import logging
from typing import Dict, List, Tuple
from pathlib import Path
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
import numpy as np

from opencompass.models.invarrag_retriever import InvarRAGRetriever


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RetrievalDataset(Dataset):
    """
    检索训练数据集
    支持 NQ, TriviaQA, PopQA 等 ODQA 数据集格式
    """
    
    def __init__(
        self,
        data_path: str,
        corpus_path: str,
        max_negatives: int = 5,
        augment_variants: bool = True,
    ):
        """
        Args:
            data_path: 训练数据路径 (JSON/JSONL 格式)
            corpus_path: 文档语料库路径
            max_negatives: 每个查询的最大负样本数
            augment_variants: 是否生成变体用于不变性损失
        """
        self.max_negatives = max_negatives
        self.augment_variants = augment_variants
        
        # 加载数据
        logger.info(f"Loading data from {data_path}")
        self.data = self._load_data(data_path)
        
        logger.info(f"Loading corpus from {corpus_path}")
        self.corpus = self._load_corpus(corpus_path)
        
        logger.info(f"Loaded {len(self.data)} examples and {len(self.corpus)} documents")
    
    def _load_data(self, path: str) -> List[Dict]:
        """加载训练数据"""
        data = []
        
        if path.endswith('.json'):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif path.endswith('.jsonl'):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
        else:
            raise ValueError(f"Unsupported file format: {path}")
        
        return data
    
    def _load_corpus(self, path: str) -> Dict[str, str]:
        """加载文档语料库"""
        corpus = {}
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                corpus[doc['id']] = doc['text']
        
        return corpus
    
    def _generate_query_variants(self, query: str, num_variants: int = 2) -> List[str]:
        """
        生成query变体（用于不变性损失）
        
        注意：这是一个简化版本，实际应使用LLM通过不同prompts改写query。
        在生产环境中，建议预先生成并保存query variants以避免训练时的计算开销。
        
        如果数据中已包含'query_variants'字段，会优先使用预生成的variants。
        
        Args:
            query: 原始查询
            num_variants: 变体数量
            
        Returns:
            查询变体列表
        """
        # 简单的query改写策略（实际应使用LLM）
        variants = []
        
        # 策略1: 改变词序
        words = query.split()
        if len(words) > 2:
            import random
            shuffled = words.copy()
            random.shuffle(shuffled)
            variants.append(' '.join(shuffled))
        
        # 策略2: 添加同义提问词
        question_starters = ['What is', 'Can you tell me', 'Do you know', 'Please explain']
        for starter in question_starters[:num_variants]:
            if not query.lower().startswith(starter.lower()):
                variant = f"{starter} {query.lower()}"
                variants.append(variant)
                if len(variants) >= num_variants:
                    break
        
        return variants[:num_variants]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        返回一个训练样本
        
        Returns:
            {
                'query': str,
                'positive_docs': List[str],
                'negative_docs': List[str],
                'query_variants': List[str]  # 用于不变性损失
            }
        """
        item = self.data[idx]
        
        # 获取查询
        query = item['question']
        
        # 获取或生成query variants（用于不变性损失）
        if 'query_variants' in item and item['query_variants']:
            # 优先使用预生成的variants
            query_variants = item['query_variants']
        elif self.augment_variants:
            # 否则在运行时生成（简化版本）
            query_variants = self._generate_query_variants(query, num_variants=2)
        else:
            query_variants = []
        
        # 获取正样本文档
        positive_doc_ids = item.get('positive_ctxs', [])
        positive_docs = []
        for doc_id in positive_doc_ids[:1]:  # 通常每个查询一个正样本
            if isinstance(doc_id, dict):
                positive_docs.append(doc_id.get('text', ''))
            elif doc_id in self.corpus:
                positive_docs.append(self.corpus[doc_id])
        
        if not positive_docs:
            # 如果没有正样本，使用答案作为正样本
            positive_docs = [item.get('answer', [''])[0]]
        
        # 获取负样本文档
        negative_doc_ids = item.get('negative_ctxs', [])
        negative_docs = []
        
        # 如果没有显式负样本，随机采样
        if not negative_doc_ids:
            corpus_ids = list(self.corpus.keys())
            negative_doc_ids = random.sample(
                corpus_ids, 
                min(self.max_negatives, len(corpus_ids))
            )
        
        for doc_id in negative_doc_ids[:self.max_negatives]:
            if isinstance(doc_id, dict):
                negative_docs.append(doc_id.get('text', ''))
            elif doc_id in self.corpus:
                negative_docs.append(self.corpus[doc_id])
        
        result = {
            'query': query,
            'positive_docs': positive_docs,
            'negative_docs': negative_docs,
            'query_variants': query_variants,  # 添加query variants
        }
        
        return result


def collate_fn(batch: List[Dict]) -> Dict:
    """
    批处理函数
    """
    queries = [item['query'] for item in batch]
    positive_docs = [doc for item in batch for doc in item['positive_docs']]
    negative_docs = [doc for item in batch for doc in item['negative_docs']]
    
    result = {
        'queries': queries,
        'positive_docs': positive_docs,
        'negative_docs': negative_docs,
    }
    
    # 处理query variants（用于不变性损失）
    if 'query_variants' in batch[0]:
        query_variants = []
        for item in batch:
            if item['query_variants']:
                query_variants.append(item['query_variants'])
        result['query_variants'] = query_variants if query_variants else None
    
    return result


class InvarRAGTrainer:
    """
    Invar-RAG 检索器训练器
    """
    
    def __init__(
        self,
        model: InvarRAGRetriever,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        learning_rate: float = 1e-4,
        num_epochs: int = 10,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        alpha_alignment: float = 1.0,
        alpha_invariance: float = 0.5,
        use_simplified_loss: bool = False,
        device: str = "cuda",
        output_dir: str = "./invarrag_checkpoints",
    ):
        """
        Args:
            model: Invar-RAG 检索器模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            learning_rate: 学习率
            num_epochs: 训练轮数
            warmup_steps: 预热步数
            max_grad_norm: 梯度裁剪阈值
            alpha_alignment: 对齐损失权重
            alpha_invariance: 不变性损失权重
            use_simplified_loss: 是否使用简化损失
            device: 设备
            output_dir: 输出目录
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.max_grad_norm = max_grad_norm
        self.alpha_alignment = alpha_alignment
        self.alpha_invariance = alpha_invariance
        self.use_simplified_loss = use_simplified_loss
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # 学习率调度器
        total_steps = len(train_loader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # 训练统计
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def compute_loss(self, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        计算总损失
        
        Returns:
            (total_loss, loss_dict)
        """
        queries = batch['queries']
        positive_docs = batch['positive_docs']
        negative_docs = batch['negative_docs']
        query_variants = batch.get('query_variants', None)
        
        # 组合所有文档
        all_docs = positive_docs + negative_docs
        
        # 判断是否使用query variants
        # 注意：在线生成variants会增加计算开销，建议使用预生成的variants
        generate_variants = query_variants is not None and len(query_variants) > 0
        
        # 前向传播
        outputs = self.model(
            queries=queries,
            documents=all_docs,
            generate_query_variants=generate_variants,
            num_query_variants=2 if generate_variants else 0,
            compute_reference=True,
            use_simplified_loss=self.use_simplified_loss,
            device=self.device
        )
        
        # 提取损失
        alignment_loss = outputs.get('alignment_loss', torch.tensor(0.0, device=self.device))
        invariance_loss = outputs.get('invariance_loss', torch.tensor(0.0, device=self.device))
        
        # 对比损失（正负样本区分）
        similarity_scores = outputs['similarity_scores']
        num_pos = len(positive_docs)
        num_neg = len(negative_docs)
        
        # 为每个查询计算对比损失
        contrastive_loss = torch.tensor(0.0, device=self.device)
        batch_size = len(queries)
        docs_per_query = len(all_docs) // batch_size
        
        for i in range(batch_size):
            # 获取该查询的分数
            query_scores = similarity_scores[i, i*docs_per_query:(i+1)*docs_per_query]
            
            # 正样本分数
            pos_scores = query_scores[:num_pos//batch_size]
            # 负样本分数
            neg_scores = query_scores[num_pos//batch_size:]
            
            # InfoNCE 损失
            if len(pos_scores) > 0 and len(neg_scores) > 0:
                # 组合正负样本
                all_scores = torch.cat([pos_scores, neg_scores])
                labels = torch.zeros(1, dtype=torch.long, device=self.device)  # 第一个是正样本
                
                contrastive_loss += F.cross_entropy(
                    all_scores.unsqueeze(0),
                    labels
                )
        
        contrastive_loss = contrastive_loss / batch_size
        
        # 总损失
        total_loss = (
            contrastive_loss + 
            self.alpha_alignment * alignment_loss + 
            self.alpha_invariance * invariance_loss
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'contrastive': contrastive_loss.item(),
            'alignment': alignment_loss.item(),
            'invariance': invariance_loss.item(),
        }
        
        return total_loss, loss_dict
    
    def train_epoch(self, epoch: int) -> Dict:
        """训练一个 epoch"""
        self.model.train()
        epoch_losses = {
            'total': 0.0,
            'contrastive': 0.0,
            'alignment': 0.0,
            'invariance': 0.0,
        }
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 计算损失
            loss, loss_dict = self.compute_loss(batch)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )
            
            # 更新参数
            self.optimizer.step()
            self.scheduler.step()
            
            # 更新统计
            for key in epoch_losses:
                epoch_losses[key] += loss_dict[key]
            
            self.global_step += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        # 计算平均损失
        num_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate(self) -> Dict:
        """验证"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        val_losses = {
            'total': 0.0,
            'contrastive': 0.0,
            'alignment': 0.0,
            'invariance': 0.0,
        }
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                _, loss_dict = self.compute_loss(batch)
                
                for key in val_losses:
                    val_losses[key] += loss_dict[key]
        
        # 计算平均损失
        num_batches = len(self.val_loader)
        for key in val_losses:
            val_losses[key] /= num_batches
        
        return val_losses
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        # 保存最新检查点
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # 保存最佳模型
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
    
    def train(self):
        """完整训练流程"""
        logger.info("Starting training...")
        logger.info(f"Total epochs: {self.num_epochs}")
        logger.info(f"Total steps: {len(self.train_loader) * self.num_epochs}")
        
        for epoch in range(1, self.num_epochs + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch}/{self.num_epochs}")
            logger.info(f"{'='*50}")
            
            # 训练
            train_losses = self.train_epoch(epoch)
            logger.info(f"Train losses: {train_losses}")
            
            # 验证
            val_losses = self.validate()
            if val_losses:
                logger.info(f"Val losses: {val_losses}")
                
                # 检查是否是最佳模型
                is_best = val_losses['total'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_losses['total']
                    logger.info(f"New best validation loss: {self.best_val_loss:.4f}")
            else:
                is_best = False
            
            # 保存检查点
            if epoch % 1 == 0:  # 每个 epoch 都保存
                self.save_checkpoint(epoch, is_best)
        
        logger.info("\nTraining completed!")


def main():
    parser = argparse.ArgumentParser(description="Train Invar-RAG Retriever")
    
    # 模型参数
    parser.add_argument("--llama_path", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--minilm_path", type=str, default="microsoft/MiniLM-L12-H384-uncased")
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=768)
    
    # 数据参数
    parser.add_argument("--train_data", type=str, required=True, help="Training data path")
    parser.add_argument("--corpus", type=str, required=True, help="Corpus path")
    parser.add_argument("--val_data", type=str, default=None, help="Validation data path")
    parser.add_argument("--max_negatives", type=int, default=5)
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--alpha_alignment", type=float, default=1.0)
    parser.add_argument("--alpha_invariance", type=float, default=0.5)
    parser.add_argument("--use_simplified_loss", action="store_true")
    
    # 其他参数
    parser.add_argument("--output_dir", type=str, default="./invarrag_checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # 创建模型
    logger.info("Creating Invar-RAG Retriever...")
    model = InvarRAGRetriever(
        llama_model_path=args.llama_path,
        minilm_model_path=args.minilm_path,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        hidden_dim=args.hidden_dim,
    ).to(args.device)
    
    # 创建数据集
    logger.info("Loading datasets...")
    train_dataset = RetrievalDataset(
        data_path=args.train_data,
        corpus_path=args.corpus,
        max_negatives=args.max_negatives,
        augment_variants=True,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )
    
    val_loader = None
    if args.val_data:
        val_dataset = RetrievalDataset(
            data_path=args.val_data,
            corpus_path=args.corpus,
            max_negatives=args.max_negatives,
            augment_variants=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
        )
    
    # 创建训练器
    trainer = InvarRAGTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        alpha_alignment=args.alpha_alignment,
        alpha_invariance=args.alpha_invariance,
        use_simplified_loss=args.use_simplified_loss,
        device=args.device,
        output_dir=args.output_dir,
    )
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    import torch.nn.functional as F
    main()

