# RESNET50
# ASNI
# ./runners/imagenet/resnet50/imagenet_resnet50_asni_explore.sh

# --asni-perc-max 90 \


python -m torch.distributed.launch --nproc_per_node=4 \
--master_port=$RANDOM main.py -a resnet50 --dataname imagenet \
--optimizer SGD+M \
--lr-policy cosine_lr \
--label-smoothing 0 \
--lr 0.35 \
--scale-coslr 1.02 \
--weight-decay 0.0001 \
--batch-size 205 \
--stages 90 \
--asni-rest-stage 90 \
--gpu-idx 4 5 6 7 \
--asni-sigmoid-scale-1 10 \
--asni-sigmoid-mag-1 81.4 \
--asni-sigmoid-trans-1 0.5 \
--save-stages \
--asni-mode sigmoid \
--pruning-strategy asni \
--logterminal \
--config imagenet_resnet50_asni 



# python -m torch.distributed.launch --nproc_per_node=4 \
# --master_port=$RANDOM main.py -a resnet50 --dataname imagenet \
# --optimizer SGD+M \
# --lr-policy normal_lr \
# --label-smoothing 0 \
# --normal-exp-scale 1440 \
# --lr 0.35 \
# --weight-decay 0.0001 \
# --batch-size 205 \
# --stages 90 \
# --asni-rest-stage 90 \
# --gpu-idx 4 5 6 7 \
# --asni-sigmoid-scale-1 10 \
# --asni-sigmoid-mag-1 81.4 \
# --asni-sigmoid-trans-1 0.5 \
# --save-stages \
# --asni-mode sigmoid \
# --pruning-strategy asni \
# --logterminal \
# --config imagenet_resnet50_asni 

