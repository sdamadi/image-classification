# Training an FC network on MNIST
# run the following to run 5 different process
# ./runners/mnist/fc/mnist_fc_train_explore.sh

# for i in $(seq 0 2)

for i in 0
do    
python -m torch.distributed.launch --nproc_per_node=1 \
--master_port=$RANDOM main.py \
-a fc --dataname mnist \
--optimizer SGD+M \
--lr-policy cosine_lr \
--lr 0.0012 \
--batch-size 1024 \
--epochs 10 \
--gpu-idx $i \
--logterminal \
--config mnist_fc_train
sleep 5
done