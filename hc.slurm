#!/bin/sh
#SBATCH --job-name deep-HCFCGNN         # this is a parameter to help you sort your job when listing it
#SBATCH --error run.e%j     # optional. By default a file slurm-{jobid}.out will be created
#SBATCH --output run.o%j      # optional. By default the error and output files are merged
#SBATCH --ntasks 1                    # number of tasks in your job. One by default
#SBATCH --cpus-per-task 4             # number of cpus for each task. One by default 
#SBATCH --mem 32GB                   #####CPU is mutually exclusive with the one for GPUs.#sbatch --mem-per-cpu=16000 # in MB
#SBATCH --partition shared-gpu         # the partition to use. By default debug-cpu. (shared-gpu)
#SBATCH --gres=gpu:1 #,VramPerGpu:12G
#SBATCH --time 0-11:59:59                  # maximum RUN time.

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate fchc-gnn
###################################
nvidia-smi
python -u deep-hierarchy.hc_main.py --graph_path '/hc_graphs/HC7_graph.pt'
