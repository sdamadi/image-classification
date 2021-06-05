# CONV6
# ASNI
# ./runners/cifar10/conv6/cifar10_conv6_asni_explore.sh

c=1
for i in 0 1 2
do    
python -m torch.distributed.launch --nproc_per_node=1 \
--master_port=$RANDOM main.py \
-a conv6 --dataname cifar10 \
--optimizer Adam \
--lr-policy constant_lr \
--lr 0.0003 \
--batch-size 60 \
--stages 30 \
--asni-rest-stage 30 \
--asni-mode sigmoid \
--asni-sigmoid-mag-1 94 \
--asni-sigmoid-trans-1 0.5 \
--asni-sigmoid-scale-1 3 \
--asni-sigmoid-mag-2 0 \
--asni-sigmoid-trans-2 0.5 \
--asni-sigmoid-scale-2 2 \
--asni-perc-max 100 \
--gpu-idx $c \
--config cifar10_conv6_asni &
sleep 5
done




