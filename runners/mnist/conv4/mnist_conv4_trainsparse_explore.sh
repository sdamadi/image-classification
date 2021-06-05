# CONV4
# ASNI
# ./runners/mnist/conv4/mnist_conv4_trainsparse_explore.sh

# --quantize-prepruned \
# --quantize-bias \


scenarios=("2021_5_27_23_44_55" "2021_5_27_23_44_59" "2021_5_27_23_45_3")
for i in ${scenarios[@]}
do 
python -m torch.distributed.launch --nproc_per_node=1 \
--master_port=$RANDOM main.py \
-a conv4 --dataname mnist \
--optimizer Adam \
--lr-policy constant_lr \
--lr 0.0003 \
--prepruned-model \
--prepruned-scen $i \
--mask-stage 24 \
--nonpruned-percent 2.1 \
--batch-size 60 \
--stages 25 \
--gpu-idx 3 \
--config mnist_conv4_quantized &
sleep 5
done


