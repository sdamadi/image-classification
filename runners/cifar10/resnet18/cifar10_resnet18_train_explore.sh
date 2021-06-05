# VGG16
# TRAINING
# ./runners/cifar10/resnet18/cifar10_resnet18_train_explore.sh


# --logterminal \

c=2
for i in 0 1 2 
do   
python -m torch.distributed.launch --nproc_per_node=1 \
--master_port=$RANDOM main.py \
-a resnet18 --dataname cifar10 \
--optimizer SGD+M \
--lr-policy cosine_lr \
--lr 0.05 \
--weight-decay 0.0005 \
--batch-size 128 \
--stages 160 \
--scale-coslr 1.06 --exp-coslr 1 \
--gpu-idx $c \
--config cifar10_resnet18_train &
sleep 5
done


# python -m torch.distributed.launch --nproc_per_node=1 \
# --master_port=$RANDOM main.py \
# -a resnet18 --dataname cifar10 \
# --optimizer SGD+M \
# --lr-policy cosine_lr \
# --lr 0.05 \
# --weight-decay 0.0005 \
# --batch-size 128 \
# --stages 160 \
# --scale-coslr 1.06 --exp-coslr 1 \
# --gpu-idx 2 \
# --logterminal \
# --config cifar10_resnet18_train


