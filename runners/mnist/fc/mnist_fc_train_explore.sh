# FC
# TRAINING
# ./runners/mnist/fc/mnist_fc_train_explore.sh

# $(seq 0 4)

for i in 0
do    
python -m torch.distributed.launch --nproc_per_node=1 \
--master_port=$RANDOM main.py \
-a fc --dataname mnist \
--optimizer SGD+M \
--lr-policy cosine_lr \
--lr 0.001 \
--batch-size 60 \
--stages 50 \
--gpu-idx 0 \
--logterminal \
--config mnist_fc_train
sleep 3
done



