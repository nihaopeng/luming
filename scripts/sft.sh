# minimind sft
CUDA_VISIBLE_DEVICES=1 python train.py --dtype=float16 --data_path "./dataset/sft_512.jsonl" --train_mode "sft" --save_weight "sft" --from_weight "pretrain" --hidden_size 768 --num_hidden_layers 16 --use_compile 1 --epochs 2

# qwen0.6b sft
CUDA_VISIBLE_DEVICES=1 python train.py --dtype=float16 --data_path "./dataset/sft_512.jsonl" --tokenizer_path "./tokenizer/qwen0.6B" --train_mode "sft" --save_weight "./out/qwen_sft.pth" --from_weight "./out/qwen" --safetensor 1 --hidden_size 768 --num_hidden_layers 16 --use_compile 0 --epochs 2 