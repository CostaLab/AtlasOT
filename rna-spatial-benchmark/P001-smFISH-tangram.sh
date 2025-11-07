#!/bin/bash
#SBATCH --mem=200G
#SBATCH --partition=gpu_node
#SBATCH --output=/beegfs/data/users/kpeng/logs/output.%J.%x.txt
#SBATCH --error=/beegfs/data/users/kpeng/logs/error.%J.%x.txt
#SBATCH --job-name=P001-smFISH-tangram
#SBATCH --mail-type=END
#SBATCH --mail-user=tudoupengkai@gmail.com
#SBATCH --time=30:00:00

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tangram

# 运行Python脚本
python -u P001-smFISH.py --method tangram --results_dir ./P001-smFISH
