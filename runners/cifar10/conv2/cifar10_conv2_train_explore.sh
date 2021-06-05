# CONV2
# TRAINING
# ./runners/cifar10/conv2/cifar10_conv2_train_explore.sh

c=4
for i in 0 1 2
do    
python -m torch.distributed.launch --nproc_per_node=1 \
--master_port=$RANDOM main.py \
-a conv2 --dataname cifar10 \
--optimizer Adam \
--lr-policy constant_lr \
--lr 0.0002 \
--batch-size 60 \
--stages 20 \
--gpu-idx $c \
--config cifar10_conv2_train &
sleep 4
done


  
# python -m torch.distributed.launch --nproc_per_node=1 \
# --master_port=$RANDOM main.py \
# -a conv2 --dataname cifar10 \
# --optimizer Adam \
# --lr-policy constant_lr \
# --lr 0.0002 \
# --batch-size 60 \
# --stages 20 \
# --gpu-idx 2 \
# --logterminal \
# --config cifar10_conv2_train


