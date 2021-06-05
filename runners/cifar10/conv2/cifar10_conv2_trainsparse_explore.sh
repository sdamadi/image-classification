# CONV2
# ASNI
# ./runners/cifar10/conv2/cifar10_conv2_trainsparse_explore.sh

# --quantize-prepruned \
# --quantize-bias \

scenarios=('2021_5_28_4_0_37' '2021_5_28_4_0_38' '2021_5_28_4_0_42')
for i in ${scenarios[@]}
do 
python -m torch.distributed.launch --nproc_per_node=1 \
--master_port=$RANDOM main.py \
-a conv2 --dataname cifar10 \
--optimizer Adam \
--lr-policy constant_lr \
--lr 0.0002 \
--prepruned-model \
--prepruned-scen $i \
--mask-stage 19 \
--nonpruned-percent 3.3 \
--batch-size 60 \
--stages 20 \
--gpu-idx 5 \
--config cifar10_conv2_quantized &
sleep 5
done



