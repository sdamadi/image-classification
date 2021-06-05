# FC
# TRAINSPARSE
# ./runners/mnist/fc/mnist_fc_trainsparse_explore.sh


# --quantize-prepruned \
# --quantize-bias \

scenarios=('2021_5_28_0_28_18' '2021_5_28_0_28_22' '2021_5_28_0_28_27')
for i in ${scenarios[@]}
do 
python -m torch.distributed.launch --nproc_per_node=1 \
--master_port=$RANDOM main.py \
-a fc --dataname mnist \
--optimizer Adam \
--lr-policy constant_lr \
--lr 0.00012 \
--prepruned-model \
--prepruned-scen $i \
--mask-stage 49 \
--nonpruned-percent 3.1 \
--batch-size 60 \
--stages 50 \
--gpu-idx 0 \
--config mnist_fc_quantized &
sleep 5
done


 
