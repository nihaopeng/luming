
case $1 in
"pretrain")
CUDA_VISIBLE_DEVICES=1 python pretrain.py --dtype=float16 --data_path "./dataset/pretrain_hq.jsonl" --save_weight "pretrain" --from_weight "none" --hidden_size 768 --num_hidden_layers 16 --use_compile 1
;;
"sft")
CUDA_VISIBLE_DEVICES=1 python pretrain.py --dtype=float16 --data_path "./dataset/sft_512.jsonl" --save_weight "sft" --from_weight "pretrain" --hidden_size 768 --num_hidden_layers 16 --use_compile 1
;;
"eval")
CUDA_VISIBLE_DEVICES=1 python eval.py
;;
esac