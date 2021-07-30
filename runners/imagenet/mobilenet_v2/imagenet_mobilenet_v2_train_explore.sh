# Training an MOBILENETV2 network on IMAGENET1K
# run the following to run 5 different process
# ./runners/imagenet/mobilenet_v2/imagenet_mobilenet_v2_train_explore.sh


python -m torch.distributed.launch --nproc_per_node=4 \
--master_port=$RANDOM main.py -a mobilenet_v2 --dataname imagenet \
--optimizer SGD+M \
--lr-policy cosine_lr \
--lr 0.4 \
--warmup-length 5 \
--scale-coslr 1.02 \
--weight-decay 0.00004 \
--batch-size 205 \
--gpu-idx 0 1 2 3 \
--logterminal \
--config imagenet_mobilenet_v2_train 





