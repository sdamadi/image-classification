# CONV4
# TRAINING
# ./runners/cifar10/conv4/cifar10_conv4_train_explore.sh

c=3
for i in 0 1 2
do    
python -m torch.distributed.launch --nproc_per_node=1 \
--master_port=$RANDOM main.py \
-a conv4 --dataname cifar10 \
--optimizer Adam \
--lr-policy constant_lr \
--lr 0.0003 \
--batch-size 60 \
--stages 25 \
--gpu-idx $c \
--config cifar10_conv4_train &
sleep 4
done

# c=3  
# python -m torch.distributed.launch --nproc_per_node=1 \
# --master_port=$RANDOM main.py \
# -a conv4 --dataname cifar10 \
# --optimizer Adam \
# --lr-policy constant_lr \
# --lr 0.0003 \
# --batch-size 60 \
# --stages 25 \
# --gpu-idx $c \
# --logterminal \
# --config cifar10_conv4_train
