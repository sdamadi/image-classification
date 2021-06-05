# CONV6
# ASNI
# ./runners/mnist/conv6/mnist_conv6_trainsparse_explore.sh

# --quantize-prepruned \
# --quantize-bias \


scenarios=("2021_5_27_23_55_38" "2021_5_27_23_55_40" "2021_5_27_23_55_45")

for i in ${scenarios[@]}
do 
python -m torch.distributed.launch --nproc_per_node=1 \
--master_port=$RANDOM main.py \
-a conv6 --dataname mnist \
--optimizer Adam \
--lr-policy constant_lr \
--lr 0.0003 \
--prepruned-model \
--prepruned-scen $i \
--mask-stage 29 \
--nonpruned-percent 1.7 \
--batch-size 60 \
--stages 30 \
--gpu-idx 0 \
--config mnist_conv6_quantized &
sleep 5
done



