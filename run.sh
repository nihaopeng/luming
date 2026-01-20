
case $1 in
"pretrain")
CUDA_VISIBLE_DEVICES=1 python train.py --dtype=float16 --data_path "./dataset/pretrain_hq.jsonl" --save_weight "pretrain" --from_weight "none" --hidden_size 768 --num_hidden_layers 16 --use_compile 1 --epochs 6
;;
"sft")
CUDA_VISIBLE_DEVICES=1 python train.py --dtype=float16 --data_path "./dataset/sft_512.jsonl" --sft 1 --save_weight "sft" --from_weight "pretrain" --hidden_size 768 --num_hidden_layers 16 --use_compile 1 --epochs 2
;;
"eval")
CUDA_VISIBLE_DEVICES=1 python eval.py --save_dir "./out" --weight "pretrain" --eval_mode "pretrain" --stream 1
;;
esac