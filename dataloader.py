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

import torch
import os
import json
import re
from pathlib import Path
from torch.utils.data import Dataset

class JsonlDataset(Dataset):
    def __init__(self, file_path: str, split: str = "train"):
        """
        支持 split 参数，格式如:
          - "train" → 全量
          - "train[0:1000]" → 前 1000 行
          - "train[:10%]" → 前 10%
          - "train[500:]" → 从 500 到末尾
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {file_path}")

        # Step 1: 获取总行数（通过偏移量）
        self.full_offsets = self._build_line_offsets(self.file_path)
        total_lines = len(self.full_offsets)

        # Step 2: 解析 split 并计算实际索引范围
        start_idx, end_idx = self._parse_split(split, total_lines)

        # Step 3: 只保留 [start_idx, end_idx) 范围内的偏移量
        self.offsets = self.full_offsets[start_idx:end_idx]

    def _build_line_offsets(self, file_path: Path):
        """构建每行起始字节偏移量列表"""
        offsets = [0]
        with open(file_path, 'rb') as f:
            while f.readline():
                offsets.append(f.tell())
        return offsets[:-1]  # 移除最后多余的偏移

    def _parse_split(self, split: str, total_lines: int):
        """解析 split 字符串，返回 (start, end) 索引"""
        # 移除 "train" 等前缀，只保留 [...] 部分
        match = re.search(r'\[(.+)\]', split)
        if not match:
            # 默认全量
            return 0, total_lines
        slice_str = match.group(1).strip()
        # 处理百分比
        if '%' in slice_str:
            if ':' in slice_str:
                raise ValueError("Percentage splits do not support ranges like 'a:b%'")
            # 例如 ":10%" → 取前 10%
            pct = float(slice_str.replace('%', '').replace(':', ''))
            if pct < 0 or pct > 100:
                raise ValueError("Percentage must be between 0 and 100")
            end = int(total_lines * pct / 100)
            return 0, end
        # 处理普通切片 "a:b"
        parts = slice_str.split(':')
        if len(parts) == 1:
            # 单个数字？不支持（HF 也不支持）
            raise ValueError(f"Invalid split format: {split}")
        start_str, end_str = parts[0], parts[1]
        # 解析 start
        start = int(start_str) if start_str else 0
        if start < 0:
            start = total_lines + start
        # 解析 end
        if end_str == '':
            end = total_lines
        else:
            end = int(end_str)
            if end < 0:
                end = total_lines + end
        # 边界检查
        start = max(0, min(start, total_lines))
        end = max(start, min(end, total_lines))
        return start, end

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.offsets):
            raise IndexError("Index out of range")
        with open(self.file_path, 'r', encoding='utf-8') as f:
            f.seek(self.offsets[idx])
            line = f.readline().strip()
            if not line:
                raise RuntimeError(f"Empty line at index {idx}")
            return json.loads(line)

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = JsonlDataset(file_path=data_path, split='train') # split = [0:100]/[:x%]/[-100:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        encoding = self.tokenizer(
            str(sample['text']),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding.input_ids.squeeze()
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = JsonlDataset(file_path=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, cs):
        messages = cs.copy()
        tools = cs[0]["functions"] if (cs and cs[0]["role"] == "system" and cs[0].get("functions")) else None
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample['conversations'])
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)
        # # === 调试打印 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # print(f"sft full prompt:{self.tokenizer.decode(input_ids)}")
        # # ================
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
    
    