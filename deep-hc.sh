#!/bin/sh
#SBATCH --job-name deepFCHC-GNN          # this is a parameter to help you sort your job when listing it
#SBATCH --error run.e%j     # optional. By default a file slurm-{jobid}.out will be created
#SBATCH --output run.o%j      # optional. By default the error and output files are merged
#SBATCH --ntasks 1                    # number of tasks in your job. One by default
#SBATCH --cpus-per-task 4             # number of cpus for each task. One by default 
#SBATCH --mem 32GB                   #####CPU is mutually exclusive with the one for GPUs.#sbatch --mem-per-cpu=16000 # in MB
#SBATCH --partition shared-gpu         # the partition to use. By default debug-cpu. (shared-gpu)
#SBATCH --gres=gpu:1 #,VramPerGpu:12G
#SBATCH --time 0-11:59:59                  # maximum RUN time of the partition.

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate fchc-gnn

###################################
export CUDA_LAUNCH_BLOCKING=1
nvidia-smi

python -u hc_main.py --model FCHCGAT --num_layers 2 --hidden_features 32 --num_heads 2 --out_heads 2 --dropout 0.3 --start_lr 0.01 --max_num_epochs 200 --num_repetitions 2 --graph_path 'hc_graphs/HC7_graph.pt' 
#python -u hc_main.py --model FCHCSAGE --num_layers 2 --hidden_features 64 --dropout 0.3 --start_lr 0.01 --max_num_epochs 200 --num_repetitions 2 --graph_path 'hc_graphs/HC7_graph.pt' 
#python -u hc_main.py --model FCHCGCN --num_layers 2 --hidden_features 64  --dropout 0.2 --start_lr 0.01 --max_num_epochs 200 --num_repetitions 2 --graph_path 'hc_graphs/HC7_graph.pt' 
#python -u hc_main.py --model FCHCGNN --num_layers 2 --hidden_features 64 --dropout 0.2 --start_lr 0.01 --max_num_epochs 200 --num_repetitions 2 --graph_path 'hc_graphs/HC7_graph.pt' 
#python -u hc_main.py --model FCHCDNN --num_layers 2 --hidden_features 256 --dropout 0.3 --start_lr 0.01 --max_num_epochs 200 --num_repetitions 2 --graph_path 'hc_graphs/HC7_graph.pt' 
