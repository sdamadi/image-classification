# Training an CONV2 network on MNIST
# run the following to run 5 different process
# ./runners/mnist/conv2/mnist_conv2_train_explore.sh

for i in 0
do    
python -m torch.distributed.launch --nproc_per_node=1 \
--master_port=$RANDOM main.py \
-a conv2 --dataname mnist \
--optimizer Adam \
--lr-policy constant_lr \
--lr 0.0002 \
--batch-size 60 \
--gpu-idx $i \
--logterminal \
--config mnist_conv2_train
sleep 4
done



