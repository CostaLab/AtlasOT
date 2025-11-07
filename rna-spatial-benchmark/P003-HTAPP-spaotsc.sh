#!/bin/bash
#SBATCH -c 20
#SBATCH --mem=200G
#SBATCH --output=/beegfs/data/users/kpeng/logs/output.%J.%x.txt
#SBATCH --error=/beegfs/data/users/kpeng/logs/error.%J.%x.txt
#SBATCH --job-name=P003-HTAPP-spaotsc
#SBATCH --mail-type=END
#SBATCH --mail-user=tudoupengkai@gmail.com
#SBATCH --time=7-30:00:00

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate spaotsc

# 运行Python脚本 - 自动处理所有样本
python -u P003-HTAPP.py --method spaotsc --results_dir ./P003-HTAPP
