# Training an VGG13 network on CIFAR10
# run the following to run 5 different process
# ./runners/cifar10/vgg13/cifar10_vgg13_train_explore.sh


c=7
for i in 0
do   
python -m torch.distributed.launch --nproc_per_node=1 \
--master_port=$RANDOM main.py \
-a vgg13 --dataname cifar10 \
--optimizer SGD+M \
--lr-policy cosine_lr \
--lr 0.05 \
--weight-decay 0.0005 \
--batch-size 128 \
--scale-coslr 1.06 --exp-coslr 1 \
--gpu-idx $c \
--logterminal \
--config cifar10_vgg13_train
sleep 5
done

# ((c=c+1))
# sleep 10


# policies="constant_lr multistep_lr cosine_lr"
# pols=($policies)
# lrs=(0.1 0.05)
# decays=(0.0001 0.0005)
# steps=("80 120" "30 60 90")
# scales=(1.02 1.06)
# exps=(1 1)
# gammas=(0.1 0.5) 
# c=0
# for i in 0 1 2
# do
# for j in 0 1
# do     
# python -m torch.distributed.launch --nproc_per_node=1 \
# --master_port=$RANDOM main.py \
# -a vgg16 --dataname cifar10 \
# --lr-policy ${pols[i]} \
# --lr ${lrs[j]} --weight-decay ${decays[j]} \
# --lr-steps ${steps[j]} --lr-gamma ${gammas[j]} \
# --scale-coslr ${scales[j]} --exp-coslr ${exps[j]} \
# --gpu-idx $c \
# --config cifar10_vgg16_train &
# ((c=c+1))
# sleep 2
# done
# done



