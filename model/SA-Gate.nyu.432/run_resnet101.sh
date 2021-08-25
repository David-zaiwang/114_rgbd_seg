export NGPUS=2
export CUDA_VISIBLE_DEVICES='5,6'
/opt/conda/envs/rgbd/bin/python3 -m torch.distributed.launch --nproc_per_node=$NGPUS train.py >> run_resnet101.log 2>&1
