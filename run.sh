nvidia-smi
pwd
log_date=$(date +'%Y%m%d')
CUDA_VISIBLE_DEVICES=6,7
nohup ../../softwares/anaconda3/envs/dsd/bin/python -u  trainer.py  >/dev/null 2 > logs/log_$log_date.txt 2>&1 &
