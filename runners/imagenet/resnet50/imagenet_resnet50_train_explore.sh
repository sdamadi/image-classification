# Training an RESNET50 network on IMAGENET1K
# run the following to run 5 different process
# ./runners/imagenet/resnet50/imagenet_resnet50_train_explore.sh

for i in 0
do
python -m torch.distributed.launch --nproc_per_node=4 \
--master_port=$RANDOM main.py -a resnet50 --dataname imagenet \
--optimizer SGD+M \
--lr-policy constant_lr \
--lr 0.35 \
--label-smoothing 0 \
--scale-coslr 1.02 \
--weight-decay 0.0001 \
--batch-size 205 \
--epochs 90 \
--gpu-idx 0 1 2 3 \
--logterminal \
--config imagenet_resnet50_train 
done


# for i in 0
# do
# python -m torch.distributed.launch --nproc_per_node=4 \
# --master_port=$RANDOM main.py -a resnet50 --dataname imagenet \
# --optimizer SGD+M \
# --lr-policy cosine_lr \
# --lr 0.35 \
# --label-smoothing 0 \
# --scale-coslr 1.02 \
# --weight-decay 0.0001 \
# --batch-size 205 \
# --epochs 90 \
# --gpu-idx 0 1 2 3 \
# --logterminal \
# --config imagenet_resnet50_train 
# done