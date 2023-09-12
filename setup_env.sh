#!/bin/bash
#SBATCH -p shared
#SBATCH -c 2
#SBATCH -t 0-02:00
#SBATCH --mem=10000
#SBATCH -o myoutput_%j.out
#SBATCH -e myerrors_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nombre@mail.com
module load python/3.10.9-fasrc01
mamba create -n venv_cpu
source activate venv_cpu
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install matplotlib
conda install scipy
conda install seaborn
conda install tqdm
conda install numba
conda install -c conda-forge emcee
conda install arviz
pip install corner
pip install getdist
pip install neurodiffeq
conda deactivate
 
