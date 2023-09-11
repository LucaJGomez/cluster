#!/bin/bash
#SBATCH --job-name=Train_LCDMpert_N.py
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=2:00:00

# Cargar el módulo necesario (si es necesario)
# module load python/3.8

# Ejecutar el script de Python
python Train_LCDMpert_N.py
