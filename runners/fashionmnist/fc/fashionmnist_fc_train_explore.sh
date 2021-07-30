# Training an FC network on FashionMNIST
# run the following to run 5 different process
# ./runners/mnist/fc/mnist_fc_train_explore.sh

# $(seq 0 4)

for i in $(seq 0 2)
do    
python -m torch.distributed.launch --nproc_per_node=1 \
--master_port=$RANDOM main.py \
-a fc --dataname fashionmnist \
--optimizer SGD+M \
--lr-policy cosine_lr \
--lr 0.001 \
--batch-size 60 \
--gpu-idx $i \
--config fashionmnist_fc_train &
sleep 3
done



