#!/bin/sh
#SBATCH --job-name shallowFCHC-GNN         
#SBATCH --error run.e%j     
#SBATCH --output run.o%j      
#SBATCH --ntasks 1                    
#SBATCH --cpus-per-task 4             
#SBATCH --mem 32GB                   
#SBATCH --partition shared-gpu        
#SBATCH --gres=gpu:1,VramPerGpu:15G
#SBATCH --time 0-04:14:59                  
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate fchc-gnn

nvidia-smi
python -u -m shallow-hierarchy.shallow_main.py --model SAGE --num_layers 2 --hidden_features 64  --dropout 0.2  --start_lr 0.001 --max_num_epochs 200 --num_repetitions 4 --graph_path 'hc_graphs/HC7_shallowgraph.pt' 
