# resnet34
# ASNI
# ./runners/cifar10/resnet34/cifar10_resnet34_asni_explore.sh


for i in 0 1 2 
do
for j in 2
do  
python -m torch.distributed.launch --nproc_per_node=1 \
--master_port=$RANDOM main.py \
-a resnet34 --dataname cifar10 \
--lr-policy cosine_lr \
--label-smoothing 0 \
--lr 0.05 --weight-decay 0.0005 \
--scale-coslr 1.06 \
--batch-size 128 \
--stages 160 \
--asni-rest-stage 160 \
--gpu-idx $j \
--asni-sigmoid-scale-1 16 \
--asni-sigmoid-mag-1 97 \
--asni-sigmoid-trans-1 0.5 \
--asni-mode sigmoid \
--pruning-strategy asni \
--config cifar10_resnet34_asni &
sleep 5
done
done


