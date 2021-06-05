# RESNET50
# TRAINSPARSE
# ./runners/imagenet/resnet50/imagenet_resnet50_trainsparse_explore.sh

# --quantize-prepruned \
# --quantize-bias \

# 
# 2021_5_30_1_20_43

scenarios=('2021_5_30_1_20_43')
for i in ${scenarios[@]}
do
python -m torch.distributed.launch --nproc_per_node=4 \
--master_port=$RANDOM main.py -a resnet50 --dataname imagenet \
--optimizer SGD+M \
--lr-policy cosine_lr \
--lr 0.128 \
--label-smoothing 0 \
--scale-coslr 1.04 \
--weight-decay 0.0001 \
--prepruned-model \
--prepruned-scen $i \
--mask-stage 89 \
--nonpruned-percent 19.9 \
--batch-size 205 \
--stages 90 \
--asni-rest-stage 90 \
--gpu-idx 0 1 2 3 \
--save-stages \
--pruning-strategy lottery \
--logterminal \
--config imagenet_resnet50_quantized 
done