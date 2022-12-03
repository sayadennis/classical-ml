#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mail-user=sayarenedennis@northwestern.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name="clsclml"
#SBATCH --output=test_classical_ml.out

module purge all
module load python-miniconda3/4.12.0
source activate classical-ml

python classical-ml/ClassicalML/run_classical_ml.py \
    --input /projects/b1131/saya/bbcar/data/02b_cnv/06_modified_data/all/reg_thres_conf90_all_studyindex.csv \
    --label /projects/b1131/saya/bbcar/data/02b_cnv/bbcar_label_studyid.csv \
    --outfn /home/srd6051/classical_ml_test.csv \
    --indexdir /projects/b1131/saya/bbcar/data/02b_cnv/indices/ \
    --scoring accuracy \
    --savemodel ~/testmodels/
