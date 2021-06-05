# VGG11
# QUANTIZE
# ./runners/cifar10/vgg11/cifar10_vgg11_quantize_explore.sh

# --quantize-bias \
# --quantize-prepruned \

times=("2021_5_28_1_41_19" "2021_5_28_1_41_21" "2021_5_28_1_41_23")

c=7

for i in $(seq 0 $((${#times[@]}-1)))
do
python -m torch.distributed.launch --nproc_per_node=1 \
--master_port=$RANDOM main.py \
-a vgg11 --dataname cifar10 \
--optimizer SGD+M \
--lr-policy cosine_lr \
--lr 0.05 --weight-decay 0.0005 \
--scale-coslr 1.06 --exp-coslr 1 \
--mask-stage 159 \
--prepruned-model \
--prepruned-scen ${times[i]} \
--nonpruned-percent 3.9 \
--batch-size 128 \
--stages 160 \
--asni-rest-stage 160 \
--gpu-idx $c \
--pruning-strategy lottery \
--config cifar10_vgg11_quantized &
sleep 5
done


# python -m torch.distributed.launch --nproc_per_node=1 \
# --master_port=$RANDOM main.py \
# -a vgg11 --dataname cifar10 \
# --optimizer SGD+M \
# --lr-policy cosine_lr \
# --lr 0.05 --weight-decay 0.0005 \
# --scale-coslr 1.06 --exp-coslr 1 \
# --mask-stage 159 \
# --prepruned-model \
# --prepruned-scen $i \
# --nonpruned-percent 1.9 \
# --batch-size 128 \
# --stages 160 \
# --asni-rest-stage 160 \
# --quantize-prepruned \
# --gpu-idx $c \
# --pruning-strategy lottery \
# --logterminal \
# --config cifar10_vgg11_quantized 

