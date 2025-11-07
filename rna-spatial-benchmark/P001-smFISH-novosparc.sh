#!/bin/bash
#SBATCH -c 20
#SBATCH --mem=200G
#SBATCH --output=/beegfs/data/users/kpeng/logs/output.%J.%x.txt
#SBATCH --error=/beegfs/data/users/kpeng/logs/error.%J.%x.txt
#SBATCH --job-name=P001-smFISH-novosparc
#SBATCH --mail-type=END
#SBATCH --mail-user=tudoupengkai@gmail.com
#SBATCH --time=30:00:00

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate novosparc

# 运行Python脚本
python -u P001-smFISH.py --method novosparc --results_dir ./P001-smFISH
