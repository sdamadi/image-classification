# VGG16
# QUANTIZE
# ./runners/cifar10/vgg16_quantize_explore.sh

# times=("2021_5_3_3_29_45" "2021_5_3_3_29_49" "2021_5_3_3_29_53" "2021_5_3_3_29_57")

times=("2021_5_4_4_4_15")
c=6

for i in $(seq 0 $((${#times[@]}-1)))
do
python -m torch.distributed.launch --nproc_per_node=1 \
--master_port=$RANDOM main.py \
-a vgg16 --dataname cifar10 \
--lr-policy cosine_lr \
--lr 0.05 --weight-decay 0.0005 \
--scale-coslr 1.02 --exp-coslr 1 \
--prepruned-scen ${times[i]} \
--nonpruned-percent 0.7 \
--quantize-bias \
--quantize-prepruned \
--gpu-idx $c \
--logterminal \
--config cifar10_vgg16_quantized 
((c=c+1))
sleep 2
done

# --quantize-prepruned \
# --logterminal \

