#!/bin/bash
#SBATCH -c 20
#SBATCH --mem=150G
#SBATCH --output=/data/gr313514/logs/output.%J.%x.txt
#SBATCH --error=/data/gr313514/logs/error.%J.%x.txt
#SBATCH --job-name=Heart_main
#SBATCH --mail-type=END
#SBATCH --mail-user=tudoupengkai@gmail.com
#SBATCH --time=300:00:00

python -u main.py
