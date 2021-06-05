# MOBILENET_V2
# TRAIN
# ./runners/imagenet/mobilenet_v2/imagenet_mobilenet_v2_train_explore.sh

# --asni-sigmoid-scale-1 12 \
# --asni-sigmoid-mag-1 92.5 \
# --asni-sigmoid-trans-1 0.5 \
# --asni-perc-max 90 \
# --save-stages \
# --asni-mode sigmoid \


python -m torch.distributed.launch --nproc_per_node=4 \
--master_port=$RANDOM main.py -a mobilenet_v2 --dataname imagenet \
--optimizer SGD+M \
--lr-policy cosine_lr \
--lr 0.4 \
--warmup-length 5 \
--scale-coslr 1.02 \
--weight-decay 0.00004 \
--batch-size 205 \
--stages 150 \
--asni-rest-stage 150 \
--gpu-idx 0 1 2 3 \
--pruning-strategy lottery \
--logterminal \
--config imagenet_mobilenet_v2_train 





