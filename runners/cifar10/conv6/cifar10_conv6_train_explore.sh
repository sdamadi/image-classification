# Training an CONV4 network on CIFAR10
# run the following to run 5 different process
# ./runners/cifar10/conv6/cifar10_conv6_train_explore.sh

c=7
for i in 0
do    
python -m torch.distributed.launch --nproc_per_node=1 \
--master_port=$RANDOM main.py \
-a conv6 --dataname cifar10 \
--optimizer Adam \
--lr-policy constant_lr \
--lr 0.0003 \
--batch-size 60 \
--stages 30 \
--gpu-idx $c \
--logterminal \
--config cifar10_conv6_train
sleep 4
done

# c=7
# python -m torch.distributed.launch --nproc_per_node=1 \
# --master_port=$RANDOM main.py \
# -a conv6 --dataname cifar10 \
# --optimizer Adam \
# --lr-policy constant_lr \
# --lr 0.0003 \
# --batch-size 60 \
# --stages 30 \
# --gpu-idx $c \
# --logterminal \
# --config cifar10_conv6_train