nvidia-smi
pwd
log_date=$(date +'%Y%m%d')
nohup ../../softwares/anaconda3/envs/dsd/bin/python -u  train.py  >/dev/null 2 > logs/service_$log_date.txt 2>&1 &
