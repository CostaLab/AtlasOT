#!/bin/bash
#SBATCH -c 20
#SBATCH --mem=200G
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:1
#SBATCH --output=/beegfs/data/users/kpeng/logs/output.%J.%x.txt
#SBATCH --error=/beegfs/data/users/kpeng/logs/error.%J.%x.txt
#SBATCH --job-name=P001-smFISH-gimvi
#SBATCH --mail-type=END
#SBATCH --mail-user=tudoupengkai@gmail.com
#SBATCH --time=30:00:00

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate scvi

# 运行Python脚本
python -u P001-smFISH.py --method gimvi --results_dir ./P001-smFISH
