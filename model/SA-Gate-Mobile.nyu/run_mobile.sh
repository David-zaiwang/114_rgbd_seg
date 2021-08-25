export NGPUS=1
/opt/conda/envs/rgbd/bin/python3 -m torch.distributed.launch --nproc_per_node=$NGPUS train.py >> run_mobile.log 2>&1
