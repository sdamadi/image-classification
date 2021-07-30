# Training an VGG19 network on CIFAR10
# run the following to run 5 different process
# ./runners/cifar10/vgg19/cifar10_vgg19_train_explore.sh


c=2
for i in 0
do   
python -m torch.distributed.launch --nproc_per_node=1 \
--master_port=$RANDOM main.py \
-a vgg19 --dataname cifar10 \
--optimizer SGD+M \
--lr-policy cosine_lr \
--lr 0.05 \
--weight-decay 0.0005 \
--batch-size 128 \
--scale-coslr 1.06 --exp-coslr 1 \
--gpu-idx $c \
--logterminal \
--config cifar10_vgg19_train
sleep 5
done


# python -m torch.distributed.launch --nproc_per_node=1 \
# --master_port=$RANDOM main.py \
# -a vgg19 --dataname cifar10 \
# --optimizer SGD+M \
# --lr-policy cosine_lr \
# --lr 0.05 \
# --weight-decay 0.0005 \
# --batch-size 128 \
# --stages 160 \
# --scale-coslr 1.06 --exp-coslr 1 \
# --gpu-idx 2 \
# --logterminal \
# --config cifar10_vgg19_train


