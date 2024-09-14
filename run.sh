nvidia-smi
pwd
ls
#yum install -y espeak
#yum install -y portaudio
#export PATH=../../softwares/anaconda3/envs/dsd/bin:../../softwares/ffmpeg-6.1-tione1/:$PATH
#export LD_LIBRARY_PATH=../../softwares/lib/lib.real:../../softwares/anaconda3/envs/dsd/lib:/usr/local/openmpi/lib:/usr/local/nccl/lib:/usr/local/openmpi/lib:/usr/local/gcc/lib:/usr/local/gcc/lib64:
#export LD_LIBRARY_PATH=../../softwares/lib/lib.real:/usr/local/python3/lib/python3.8/site-packages/torch/lib:/usr/local/nccl/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/python3/lib:/usr/local/python3/lib64:/usr/local/openmpi/lib:/usr/local/gcc/lib:/usr/local/gcc/lib64:/usr/local/python3/lib/python3.8/site-packages/torch/lib:/usr/local/openmpi/lib:/usr/local/nccl/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/nccl/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/python3/lib:/usr/local/python3/lib64:/usr/local/openmpi/lib:/usr/local/gcc/lib:/usr/local/gcc/lib64:
#env
log_date=$(date +'%Y%m%d')
nohup ../../softwares/anaconda3/envs/dsd/bin/python -u  train.py  >/dev/null 2 > logs/service_$log_date.txt 2>&1 &
