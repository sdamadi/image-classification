# CONV4
# ASNI
# ./runners/cifar10/conv4/cifar10_conv4_trainsparse_explore.sh


# --quantize-prepruned \
# --quantize-bias \


scenarios=("2021_5_28_8_0_8" "2021_5_28_8_0_13" "2021_5_28_8_0_18")
for i in ${scenarios[@]}
do 
python -m torch.distributed.launch --nproc_per_node=1 \
--master_port=$RANDOM main.py \
-a conv4 --dataname cifar10 \
--optimizer Adam \
--lr-policy constant_lr \
--lr 0.0003 \
--prepruned-model \
--prepruned-scen $i \
--mask-stage 24 \
--nonpruned-percent 5.5 \
--batch-size 60 \
--stages 25 \
--gpu-idx 1 \
--config cifar10_conv4_quantized &
sleep 5
done

