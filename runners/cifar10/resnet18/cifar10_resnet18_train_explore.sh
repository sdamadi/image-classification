# Training an RESNET18 network on CIFAR10
# run the following to run 5 different process
# ./runners/cifar10/resnet18/cifar10_resnet18_train_explore.sh


c=0
for i in 1.02 1.03 1.04 1.05
do   
for j in 0.1 0.05
do 
python -m torch.distributed.launch --nproc_per_node=1 \
--master_port=$RANDOM main.py \
-a resnet18 --dataname cifar10 \
--optimizer SGD+M \
--lr-policy cosine_lr \
--lr $i \
--weight-decay 0.0001 \
--batch-size 128 \
--scale-coslr $j --exp-coslr 1 \
--gpu-idx $c \
--epochs 160 \
--config cifar10_resnet18_train &
sleep 5
((c+=1))
done
done


# --logterminal \
