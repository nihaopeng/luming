# pretrain minimind eval
CUDA_VISIBLE_DEVICES=2 python eval.py --tokenizer_path "./tokenizer/minimind" --from_weight "./out/minimind_pretrain" --sep "<|im_start|>,<|im_end|>,<|endoftext|>" --eval_mode "pretrain" --stream 1

# pretrain qwen0.6B eval
CUDA_VISIBLE_DEVICES=2 python eval.py --tokenizer_path "./tokenizer/qwen0.6Bbase" --from_weight "./out/qwen0.6Bbase" --sep "<|im_start|>,<|im_end|>,<|endoftext|>" --eval_mode "pretrain" --stream 0

# sft minimind eval
CUDA_VISIBLE_DEVICES=2 python eval.py --tokenizer_path "./tokenizer/qwen0.6Bbase" --from_weight "./out/qwen_sft" --sep "<|im_start|>assistant,<|im_end|>,<|endoftext|>" --eval_mode "sft" --stream 1

# sft qwen0.6B eval
CUDA_VISIBLE_DEVICES=2 python eval.py --tokenizer_path "./tokenizer/qwen0.6Bbase" --from_weight "./out/qwen_sft" --sep "<|im_start|>assistant,<|im_end|>,<|endoftext|>" --eval_mode "sft" --stream 1