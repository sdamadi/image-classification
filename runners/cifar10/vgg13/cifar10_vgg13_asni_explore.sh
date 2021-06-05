# VGG11
# ASNI
# ./runners/cifar10/vgg13/cifar10_vgg13_asni_explore.sh

# --logterminal \

for i in 0 1 2 
do
for j in 6
do  
python -m torch.distributed.launch --nproc_per_node=1 \
--master_port=$RANDOM main.py \
-a vgg13 --dataname cifar10 \
--lr-policy cosine_lr \
--label-smoothing 0 \
--lr 0.05 --weight-decay 0.0005 \
--scale-coslr 1.06 \
--batch-size 128 \
--stages 160 \
--asni-rest-stage 160 \
--gpu-idx $j \
--asni-sigmoid-scale-1 16 \
--asni-sigmoid-mag-1 98 \
--asni-sigmoid-trans-1 0.5 \
--asni-mode sigmoid \
--pruning-strategy asni \
--config cifar10_vgg13_asni &
sleep 2
done
done












# c=0
# ts=(2 4 6 8 10 12)
# mags=(99.5)
# for i in 0 1 2 3 4 5
# do
# for j in 0
# do  
# python -m torch.distributed.launch --nproc_per_node=1 \
# --master_port=$RANDOM main.py \
# -a vgg16 --dataname cifar10 \
# --lr-policy cosine_lr \
# --lr 0.05 --weight-decay 0.0005 \
# --lr-steps 80 120 --lr-gamma 0.1 \
# --scale-coslr 1.06 --exp-coslr 1 \
# --asni-sigmoid-scale ${ts[i]} \
# --asni-sigmoid-mag ${mags[j]} \
# --gpu-idx $c \
# --config cifar10_vgg16_asni &
# ((c=c+1))
# sleep 2
# done
# done

# sleep 2h

# c=2
# ts=(2 4 6 8 10 12)
# mags=(99.5)
# for i in 0 1 2 3 4 5
# do
# for j in 0
# do     
# python -m torch.distributed.launch --nproc_per_node=1 \
# --master_port=$RANDOM main.py \
# -a vgg16 --dataname cifar10 \
# --lr-policy cosine_lr \
# --lr 0.1 --weight-decay 0.0001 \
# --lr-steps 80 120 --lr-gamma 0.1 \
# --scale-coslr 1.02 --exp-coslr 1 \
# --asni-sigmoid-scale ${ts[i]} \
# --asni-sigmoid-mag ${mags[j]} \
# --gpu-idx $c \
# --config cifar10_vgg16_asni &
# ((c=c+1))
# sleep 2
# done
# done

# sleep 2h

# c=2
# ts=(2 4 6 8 10 12)
# mags=(99.5)
# for i in 0 1 2 3 4 5
# do
# for j in 0
# do   
# python -m torch.distributed.launch --nproc_per_node=1 \
# --master_port=$RANDOM main.py \
# -a vgg16 --dataname cifar10 \
# --lr-policy multistep_lr \
# --lr 0.1 --weight-decay 0.0001 \
# --lr-steps 80 120 --lr-gamma 0.1 \
# --scale-coslr 1 --exp-coslr 1 \
# --asni-sigmoid-scale ${ts[i]} \
# --asni-sigmoid-mag ${mags[j]} \
# --gpu-idx $c \
# --config cifar10_vgg16_asni &
# ((c=c+1))
# sleep 2
# done
# done



