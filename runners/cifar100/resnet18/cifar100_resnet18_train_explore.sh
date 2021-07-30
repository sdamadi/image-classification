# Training an RESNET18 network on CIFAR100
# run the following to run 5 different process
# ./runners/cifar100/resnet18/cifar100_resnet18_train_explore.sh


c=2
for i in 0 
do   
python -m torch.distributed.launch --nproc_per_node=1 \
--master_port=$RANDOM main.py \
-a resnet18 --dataname cifar100 \
--optimizer SGD+M \
--lr-policy cosine_lr \
--lr 0.05 \
--weight-decay 0.0005 \
--batch-size 128 \
--scale-coslr 1.06 --exp-coslr 1 \
--gpu-idx $c \
--logterminal \
--config cifar100_resnet18_train
sleep 5
done
