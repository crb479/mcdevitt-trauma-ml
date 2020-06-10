#!/bin/sh

#SBATCH --job-name=pubmedPDFdownload

#SBATCH --time=144:00:00
#SBATCH --nodes=1
#SBATCH --mem=10GB
#SBATCH --output=output.out
#SBATCH --cpus-per-task=4

#SBATCH --mail-type=END
#SBATCH --mail-user=david.shaw@nyu.edu

echo "***sbatch complete"

module purge
module load gcc/6.3.0
module load anaconda3/5.3.1

conda create --prefix. /env python=3.7
source activate ./env

conda install bio
conda install Bio
conda install lxml
conda install requests
conda install requests3
conda install beautifulsoup4
conda install pandas
conda install biopython
conda install -c mlgill fake-useragent

# python download.py

python pubmed_central.py -o pubmed/PDF/ pubmed/articles.csv