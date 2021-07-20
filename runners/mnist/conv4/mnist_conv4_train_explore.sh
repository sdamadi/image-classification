# CONV4
# TRAINING
# ./runners/mnist/conv4/mnist_conv4_train_explore.sh

c=6
for i in 0 1 2
do    
python -m torch.distributed.launch --nproc_per_node=1 \
--master_port=$RANDOM main.py \
-a conv4 --dataname mnist \
--optimizer Adam \
--lr-policy constant_lr \
--lr 0.0003 \
--batch-size 60 \
--stages 25 \
--gpu-idx $c \
--config mnist_conv4_train &
sleep 4
done


# python -m torch.distributed.launch --nproc_per_node=1 \
# --master_port=$RANDOM main.py \
# -a conv4 --dataname mnist \
# --optimizer Adam \
# --lr-policy constant_lr \
# --lr 0.0003 \
# --batch-size 60 \
# --stages 25 \
# --gpu-idx 6 \
# --logterminal \
# --config mnist_conv4_train