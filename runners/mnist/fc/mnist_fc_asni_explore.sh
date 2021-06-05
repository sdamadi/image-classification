# FC
# ASNI
# ./runners/mnist/fc/mnist_fc_asni_explore.sh

c=5
for i in 0 1 2
do
python -m torch.distributed.launch --nproc_per_node=1 \
--master_port=$RANDOM main.py \
-a fc --dataname mnist \
--optimizer Adam \
--lr-policy constant_lr \
--lr 0.00012 \
--scale-coslr 1 \
--batch-size 60 \
--stages 50 \
--asni-mode sigmoid \
--asni-rest-stage 50 \
--asni-sigmoid-mag-1 98 \
--asni-sigmoid-trans-1 0.5 \
--asni-sigmoid-scale-1 5 \
--gpu-idx $c \
--config mnist_fc_asni &
sleep 5
done



# python -m torch.distributed.launch --nproc_per_node=1 \
# --master_port=$RANDOM main.py \
# -a fc --dataname mnist \
# --optimizer SGD+M \
# --lr-policy cosine_lr \
# --lr 0.01 \
# --scale-coslr 1 \
# --batch-size 60 \
# --stages 50 \
# --asni-mode sigmoid \
# --asni-rest-stage 50 \
# --asni-sigmoid-mag-1 99 \
# --asni-sigmoid-trans-1 0.5 \
# --asni-sigmoid-scale-1 1 \
# --gpu-idx $c \
# --logterminal \
# --config mnist_fc_asni