# ResNet18
# QUANTIZE
# ./runners/cifar10/resnet18/cifar10_resnet18_quantize_explore.sh


# --quantize-bias \
# --quantize-prepruned \

times=("2021_5_28_1_56_41" "2021_5_28_1_56_42" "2021_5_28_1_56_44")

c=7

for i in $(seq 0 $((${#times[@]}-1)))
do
python -m torch.distributed.launch --nproc_per_node=1 \
--master_port=$RANDOM main.py \
-a resnet18 --dataname cifar10 \
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
--config cifar10_resnet18_quantized &
sleep 5
done