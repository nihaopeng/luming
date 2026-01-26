# minimind sft
CUDA_VISIBLE_DEVICES=1 python train.py --dtype=float16 --data_path "./dataset/sft_mini_512.jsonl" --tokenizer_path "./tokenizer/minimind" --train_mode "sft" --save_weight "./out/minimind_sft" --from_weight "./out/minimind_pretrain" --hidden_size 768 --num_hidden_layers 16 --use_compile 0 --epochs 2 --sep "<|im_start|>assistant,<|im_end|>,<|endoftext|>"

# qwen0.6b sft
CUDA_VISIBLE_DEVICES=1 python train.py --dtype=float16 --data_path "./dataset/sft_mini_512.jsonl" --tokenizer_path "./tokenizer/qwen0.6Bbase" --train_mode "sft" --save_weight "./out/qwen_sft" --from_weight "./out/qwen0.6Bbase" --use_compile 0 --epochs 2 --sep "<|im_start|>assistant,<|im_end|>,<|endoftext|>" > sft.log 2>&1