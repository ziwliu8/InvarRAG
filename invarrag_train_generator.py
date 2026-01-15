"""
Invar-RAG 生成器微调脚本
用于微调 LLM 以更好地利用检索到的文档生成答案
"""

import os
import json
import argparse
import logging
from typing import Dict, List, Optional
from pathlib import Path
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import numpy as np


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGGenerationDataset(Dataset):
    """
    RAG 生成训练数据集
    包含查询、检索到的文档和目标答案
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_context_length: int = 2048,
        max_answer_length: int = 128,
        num_retrieved_docs: int = 5,
    ):
        """
        Args:
            data_path: 数据路径
            tokenizer: tokenizer
            max_context_length: 最大上下文长度
            max_answer_length: 最大答案长度
            num_retrieved_docs: 使用的检索文档数量
        """
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
        self.max_answer_length = max_answer_length
        self.num_retrieved_docs = num_retrieved_docs
        
        logger.info(f"Loading generation data from {data_path}")
        self.data = self._load_data(data_path)
        logger.info(f"Loaded {len(self.data)} examples")
    
    def _load_data(self, path: str) -> List[Dict]:
        """加载数据"""
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
    
    def _format_prompt(
        self, 
        question: str, 
        retrieved_docs: List[str]
    ) -> str:
        """
        格式化提示，包含查询和检索到的文档
        
        根据论文，使用以下格式：
        Context: [Retrieved Documents]
        Question: [Question]
        Answer:
        """
        # 组合检索到的文档
        context = "\n\n".join([
            f"Document {i+1}: {doc}"
            for i, doc in enumerate(retrieved_docs[:self.num_retrieved_docs])
        ])
        
        prompt = f"""You are a helpful assistant. Answer the question based on the given context.

Context:
{context}

Question: {question}

Answer:"""
        
        return prompt
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        返回一个训练样本
        
        Returns:
            {
                'input_ids': torch.Tensor,
                'attention_mask': torch.Tensor,
                'labels': torch.Tensor,
            }
        """
        item = self.data[idx]
        
        # 获取问题和答案
        question = item['question']
        answer = item.get('answer', item.get('answers', [''])[0])
        if isinstance(answer, list):
            answer = answer[0]
        
        # 获取检索到的文档
        retrieved_docs = []
        if 'retrieved_docs' in item:
            retrieved_docs = item['retrieved_docs']
        elif 'positive_ctxs' in item:
            retrieved_docs = [
                ctx.get('text', '') if isinstance(ctx, dict) else ctx
                for ctx in item['positive_ctxs']
            ]
        
        # 如果没有检索文档，使用空列表
        if not retrieved_docs:
            retrieved_docs = ["No relevant documents found."]
        
        # 格式化输入
        prompt = self._format_prompt(question, retrieved_docs)
        
        # Tokenize
        # 输入：prompt
        input_encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_context_length,
            padding=False,
            return_tensors=None,
        )
        
        # 输出：answer
        answer_encoding = self.tokenizer(
            answer,
            truncation=True,
            max_length=self.max_answer_length,
            padding=False,
            return_tensors=None,
        )
        
        # 组合输入和输出
        input_ids = input_encoding['input_ids'] + answer_encoding['input_ids']
        attention_mask = input_encoding['attention_mask'] + answer_encoding['attention_mask']
        
        # 创建 labels（只计算答案部分的损失）
        labels = [-100] * len(input_encoding['input_ids']) + answer_encoding['input_ids']
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch: List[Dict], pad_token_id: int) -> Dict:
    """
    批处理函数，进行动态填充
    """
    # 找到最大长度
    max_length = max(len(item['input_ids']) for item in batch)
    
    # 填充
    input_ids = []
    attention_mask = []
    labels = []
    
    for item in batch:
        seq_len = len(item['input_ids'])
        padding_len = max_length - seq_len
        
        # 填充 input_ids
        input_ids.append(
            torch.cat([
                item['input_ids'],
                torch.full((padding_len,), pad_token_id, dtype=torch.long)
            ])
        )
        
        # 填充 attention_mask
        attention_mask.append(
            torch.cat([
                item['attention_mask'],
                torch.zeros(padding_len, dtype=torch.long)
            ])
        )
        
        # 填充 labels
        labels.append(
            torch.cat([
                item['labels'],
                torch.full((padding_len,), -100, dtype=torch.long)
            ])
        )
    
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask),
        'labels': torch.stack(labels),
    }


class InvarRAGGeneratorTrainer:
    """
    Invar-RAG 生成器训练器
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        device: str = "cuda",
        output_dir: str = "./invarrag_generator_checkpoints",
        gradient_accumulation_steps: int = 1,
    ):
        """
        Args:
            model: 生成模型
            tokenizer: tokenizer
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            learning_rate: 学习率
            num_epochs: 训练轮数
            warmup_steps: 预热步数
            max_grad_norm: 梯度裁剪阈值
            device: 设备
            output_dir: 输出目录
            gradient_accumulation_steps: 梯度累积步数
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # 学习率调度器
        total_steps = (len(train_loader) // gradient_accumulation_steps) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # 训练统计
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def train_epoch(self, epoch: int) -> float:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            # 移动到设备
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 前向传播
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # 梯度累积
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            
            # 更新参数
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # 更新统计
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self) -> float:
        """验证"""
        if self.val_loader is None:
            return float('inf')
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # 移动到设备
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 前向传播
                outputs = self.model(**batch)
                loss = outputs.loss
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        # 保存模型
        model_path = self.output_dir / f"checkpoint_epoch_{epoch}"
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        logger.info(f"Saved checkpoint to {model_path}")
        
        # 保存训练状态
        state_path = self.output_dir / f"training_state_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'global_step': self.global_step,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
        }, state_path)
        
        # 保存最佳模型
        if is_best:
            best_path = self.output_dir / "best_model"
            self.model.save_pretrained(best_path)
            self.tokenizer.save_pretrained(best_path)
            logger.info(f"Saved best model to {best_path}")
    
    def train(self):
        """完整训练流程"""
        logger.info("Starting generator training...")
        logger.info(f"Total epochs: {self.num_epochs}")
        logger.info(f"Total steps: {len(self.train_loader) * self.num_epochs}")
        
        for epoch in range(1, self.num_epochs + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch}/{self.num_epochs}")
            logger.info(f"{'='*50}")
            
            # 训练
            train_loss = self.train_epoch(epoch)
            logger.info(f"Train loss: {train_loss:.4f}")
            
            # 验证
            val_loss = self.validate()
            if val_loss != float('inf'):
                logger.info(f"Val loss: {val_loss:.4f}")
                
                # 检查是否是最佳模型
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    logger.info(f"New best validation loss: {self.best_val_loss:.4f}")
            else:
                is_best = False
            
            # 保存检查点
            self.save_checkpoint(epoch, is_best)
        
        logger.info("\nGenerator training completed!")


def main():
    parser = argparse.ArgumentParser(description="Train Invar-RAG Generator")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    # 数据参数
    parser.add_argument("--train_data", type=str, required=True, help="Training data path")
    parser.add_argument("--val_data", type=str, default=None, help="Validation data path")
    parser.add_argument("--max_context_length", type=int, default=2048)
    parser.add_argument("--max_answer_length", type=int, default=128)
    parser.add_argument("--num_retrieved_docs", type=int, default=5)
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # 其他参数
    parser.add_argument("--output_dir", type=str, default="./invarrag_generator_checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # 加载 tokenizer
    logger.info(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    logger.info(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # 使用 LoRA
    if args.use_lora:
        logger.info("Applying LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # 创建数据集
    logger.info("Loading datasets...")
    train_dataset = RAGGenerationDataset(
        data_path=args.train_data,
        tokenizer=tokenizer,
        max_context_length=args.max_context_length,
        max_answer_length=args.max_answer_length,
        num_retrieved_docs=args.num_retrieved_docs,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
        num_workers=4,
    )
    
    val_loader = None
    if args.val_data:
        val_dataset = RAGGenerationDataset(
            data_path=args.val_data,
            tokenizer=tokenizer,
            max_context_length=args.max_context_length,
            max_answer_length=args.max_answer_length,
            num_retrieved_docs=args.num_retrieved_docs,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
            num_workers=4,
        )
    
    # 创建训练器
    trainer = InvarRAGGeneratorTrainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        device=args.device,
        output_dir=args.output_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()

