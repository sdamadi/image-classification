# Training an CONV6A network on MNIST
# run the following to run 5 different process
# ./runners/mnist/conv6a/mnist_conv6a_train_explore.sh

c=7
for i in 0
do    
python -m torch.distributed.launch --nproc_per_node=1 \
--master_port=$RANDOM main.py \
-a conv6a --dataname mnist \
--optimizer Adam \
--lr-policy constant_lr \
--lr 0.0003 \
--batch-size 60 \
--gpu-idx $c \
--logterminal \
--config mnist_conv6a_train
sleep 4
done