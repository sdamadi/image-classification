# CONV2
# ASNI
# ./runners/mnist/conv2/mnist_conv2_trainsparse_explore.sh

# --quantize-prepruned \
# --quantize-bias \

scenarios=("2021_5_28_7_15_32" "2021_5_28_7_15_34" "2021_5_28_7_15_40")

for i in ${scenarios[@]}
do 
python -m torch.distributed.launch --nproc_per_node=1 \
--master_port=$RANDOM main.py \
-a conv2 --dataname mnist \
--optimizer Adam \
--lr-policy constant_lr \
--lr 0.0002 \
--prepruned-model \
--prepruned-scen $i \
--mask-stage 19 \
--nonpruned-percent 2.6 \
--batch-size 60 \
--stages 20 \
--gpu-idx 1 \
--config mnist_conv2_quantized &
sleep 5
done


# python -m torch.distributed.launch --nproc_per_node=1 \
# --master_port=$RANDOM main.py \
# -a conv2 --dataname mnist \
# --optimizer Adam \
# --lr-policy constant_lr \
# --lr 0.0002 \
# --prepruned-model \
# --prepruned-scen 2021_5_28_7_15_32 \
# --quantize-prepruned \
# --quantize-bias \
# --mask-stage 19 \
# --nonpruned-percent 2.6 \
# --batch-size 60 \
# --stages 20 \
# --gpu-idx 6 \
# --logterminal \
# --config mnist_conv2_quantized
