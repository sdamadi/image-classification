# CONV2
# TRAINING
# ./runners/mnist/conv2/mnist_conv2_train_explore.sh

for i in 4
do    
python -m torch.distributed.launch --nproc_per_node=1 \
--master_port=$RANDOM main.py \
-a conv2 --dataname mnist \
--optimizer Adam \
--lr-policy constant_lr \
--lr 0.0002 \
--batch-size 60 \
--stages 20 \
--gpu-idx $i \
--logterminal \
--config mnist_conv2_train
sleep 4
done



