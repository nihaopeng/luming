
case $1 in
"pretrain")
CUDA_VISIBLE_DEVICES=1 python train.py --dtype=float16 --data_path "./dataset/pretrain_hq.jsonl" --save_weight "pretrain" --from_weight "none" --hidden_size 768 --num_hidden_layers 16 --use_compile 1 --epochs 6 --tokenizer_path "./tokenizer/minimind" --safetensor 0 --train_mode "pretrain"
;;
"sft")
CUDA_VISIBLE_DEVICES=1 python train.py --dtype=float16 --data_path "./dataset/sft_512.jsonl" --from_weight "./out/pretrain_768.pth" --hidden_size 768 --num_hidden_layers 16 --use_compile 1 --epochs 2 --tokenizer_path "./tokenizer/minimind" --safetensor 0 --train_mode "sft" --save_weight "./out/tmp.pth"
;;
"eval")
CUDA_VISIBLE_DEVICES=1 python eval.py --save_dir "./out" --weight "sft" --eval_mode "sft" --stream 1
;;
esac