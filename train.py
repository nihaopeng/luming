# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications Copyright 2026 Yutao Peng, Northeastern University, liaoning, China

import os
import sys
import time
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from config import MiniMindConfig
from dataloader import PretrainDataset, SFTDataset
from utils import Logger, SkipBatchSampler, TokenConfig, get_lr, init_distributed_mode, init_model, is_main_process, lm_checkpoint, setup_seed,get_args

DTYPE2FN = {
    "float16":torch.float16,
    "float32":torch.float32,
    "float64":torch.float64,
    "bfloat16":torch.bfloat16
}

def train_epoch(epoch, loader, iters, start_step=0, wandb=None, autocast_ctx=None):
    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # PyTorch 自动混合精度（Automatic Mixed Precision, AMP） 的上下文管理器，用于在训练过程中自动选择 float16（半精度）或 float32（单精度）进行计算，从而在保持模型精度的同时显著提升训练速度并减少显存占用。
        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss
            if hasattr(res, 'aux_loss') and res.aux_loss is not None:
                loss = loss + res.aux_loss
            loss = loss / args.accumulation_steps
            # 判断是否NaN，如果出现NaN则提前终止训练。
            if not torch.isfinite(loss):
                Logger("Saving last good checkpoint before exit...")
                raw_model = model.module if hasattr(model, 'module') else model
                raw_model = getattr(raw_model, '_orig_mod', raw_model)
                raw_model.save_pretrained(args.save_weight, safe_serialization=True)
                Logger("!quit train!")
                sys.exit(1)

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if hasattr(res, 'aux_loss') and res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"tmp_loss": loss,"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            # === 提取原始模型 ===
            raw_model = model.module if hasattr(model, 'module') else model # 模型使用DistributedDataParallel初始化，因此需要单独保存。
            # 如果用了 torch.compile，再加一层：
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            # 保存原始模型（safetensors + config.json）
            raw_model.save_pretrained(args.save_weight, safe_serialization=True)
            # lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='./checkpoints')
            model.train()
        del input_ids, labels, res, loss

if __name__ == "__main__":
    args = get_args()
    
    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(os.path.dirname(args.save_weight), exist_ok=True)
    # lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='./checkpoints') if args.from_resume==1 else None

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = DTYPE2FN[args.dtype]
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type,dtype=dtype)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(args)
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    sep = args.sep.split(",")
    assert len(sep)==3,"sep参数需要三个token，用逗号分隔"
    token_config = TokenConfig(sep[0],sep[1],sep[2])
    train_ds = SFTDataset(args.data_path, tokenizer, token_config, max_length=args.max_seq_len) if args.train_mode=="sft"\
         else PretrainDataset(args.data_path, tokenizer, token_config, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.amp.GradScaler(device_type,enabled= args.dtype != 'bfloat16')
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 7. DDP包模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}# 忽略这两个buffer的同步，这个没有梯度，所以没必要同步
        model = DistributedDataParallel(model, device_ids=[local_rank]) # 自动进行参数同步。
    
    # ========== 8. 开始训练 ==========
    start_epoch, start_step = 0, 0
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        # if epoch == start_epoch and start_step > 0: # 第一个epoch且存在检查点
        #     batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
        #     loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        #     Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
        #     train_epoch(epoch, loader, iters=len(loader) + start_step + 1, start_step=start_step,autocast_ctx=autocast_ctx)
        # else: # 默认从头开始
        loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
        train_epoch(epoch, loader, iters=len(loader), start_step=0, autocast_ctx=autocast_ctx)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()
    
    