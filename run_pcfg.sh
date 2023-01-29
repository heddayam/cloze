#CUDA_VISIBLE_DEVICES=0  python pcfg.py --start 0 --n_cpus 8 & 
#CUDA_VISIBLE_DEVICES=1  python pcfg.py --start 10 --n_cpus 8

CUDA_VISIBLE_DEVICES=1  python pcfg.py --start 0 --step 1000 --n_cpus 7 & 
CUDA_VISIBLE_DEVICES=0  python pcfg.py --start 1000 --step 1000 --n_cpus 7

