# CONV6
# ASNI
# ./runners/cifar10/conv6/cifar10_conv6_trainsparse_explore.sh

# --quantize-prepruned \
# --quantize-bias \

scenarios=("2021_5_28_8_4_55" "2021_5_28_8_4_58" "2021_5_28_8_5_3")

for i in ${scenarios[@]}
do 
python -m torch.distributed.launch --nproc_per_node=1 \
--master_port=$RANDOM main.py \
-a conv6 --dataname cifar10 \
--optimizer Adam \
--lr-policy constant_lr \
--lr 0.0003 \
--prepruned-model \
--prepruned-scen $i \
--mask-stage 29 \
--nonpruned-percent 7.3 \
--batch-size 60 \
--stages 30 \
--gpu-idx 6 \
--config cifar10_conv6_quantized &
sleep 5
done