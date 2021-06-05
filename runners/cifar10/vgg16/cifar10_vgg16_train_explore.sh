# VGG16
# TRAINING
# ./runners/cifar10/vgg16/cifar10_vgg16_train_explore.sh


# --logterminal \

c=2
for i in 0 1 2 
do   
python -m torch.distributed.launch --nproc_per_node=1 \
--master_port=$RANDOM main.py \
-a vgg16 --dataname cifar10 \
--optimizer SGD+M \
--lr-policy cosine_lr \
--lr 0.05 \
--weight-decay 0.0005 \
--batch-size 128 \
--stages 160 \
--scale-coslr 1.06 --exp-coslr 1 \
--gpu-idx $c \
--config cifar10_vgg16_train &
sleep 10
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



