#!/bin/bash
#SBATCH --job-name deepFCHC-GNN
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem=32GB
#SBATCH --partition=shared-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:14:59
#SBATCH --error=run%A.e
#SBATCH --output=run%A.o

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate test_gpu
nvidia-smi

export CUDA_LAUNCH_BLOCKING=1

#python -u hc_main_gat.py --graph_path '/home/users/b/bini/gnn/imagenet_gnn/hc_graphs/HC7_graph.pt'
python -u hc_main.py --model HCFCGAT --num_layers 2 --hidden_features 64 --num_heads 2 --out_heads 2 --dropout 0.3 --start_lr 0.01 --max_num_epochs 200 --num_repetitions 2 --graph_path 'hc_graphs/HC7_graph.pt' 
