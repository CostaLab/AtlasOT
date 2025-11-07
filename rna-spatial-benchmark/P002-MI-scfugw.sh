#!/bin/bash
#SBATCH -c 20
#SBATCH --mem=200G
#SBATCH --partition=gpu_node
#SBATCH --output=/beegfs/data/users/kpeng/logs/output.%J.%x.txt
#SBATCH --error=/beegfs/data/users/kpeng/logs/error.%J.%x.txt
#SBATCH --job-name=P002-MI-scfugw-common
#SBATCH --mail-type=END
#SBATCH --mail-user=tudoupengkai@gmail.com
#SBATCH --time=30:00:00

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate scfugw

# 运行Python脚本 - 自动处理所有样本
python -u P002-MI.py --method scfugw --results_dir ./P002-MI
