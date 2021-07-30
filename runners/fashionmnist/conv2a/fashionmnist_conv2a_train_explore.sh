# Training an CONV2A network on FashionMNIST
# run the following to run 5 different process
# ./runners/mnist/conv2a/fashionmnist_conv2a_train_explore.sh

for i in 0
do    
python -m torch.distributed.launch --nproc_per_node=1 \
--master_port=$RANDOM main.py \
-a conv2a --dataname fashionmnist \
--optimizer Adam \
--lr-policy constant_lr \
--lr 0.0002 \
--batch-size 60 \
--gpu-idx $i \
--logterminal \
--config fashionmnist_conv2a_train
sleep 4
done



