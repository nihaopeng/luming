# minimind pretrain_hq pretrain
CUDA_VISIBLE_DEVICES=1 python train.py --dtype=float16 --data_path "./dataset/pretrain_hq.jsonl" --save_weight "pretrain" --from_weight "none" --hidden_size 768 --num_hidden_layers 16 --use_compile 1 --epochs 6
