# CONV2
# ASNI
# ./runners/mnist/conv2/mnist_conv2_asni_explore.sh

c=6
for i in 0 1 2
do    
python -m torch.distributed.launch --nproc_per_node=1 \
--master_port=$RANDOM main.py \
-a conv2 --dataname mnist \
--optimizer Adam \
--lr-policy constant_lr \
--lr 0.0002 \
--batch-size 60 \
--stages 20 \
--asni-rest-stage 20 \
--asni-mode sigmoid \
--asni-sigmoid-mag-1 99.2 \
--asni-sigmoid-trans-1 0.5 \
--asni-sigmoid-scale-1 2 \
--asni-sigmoid-mag-2 0 \
--asni-sigmoid-trans-2 0.5 \
--asni-sigmoid-scale-2 2 \
--asni-perc-max 100 \
--gpu-idx $c \
--config mnist_conv2_asni &
sleep 4
done






