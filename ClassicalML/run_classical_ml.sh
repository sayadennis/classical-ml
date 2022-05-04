#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -t 48:00:00
#SBATCH -n 12
#SBATCH --mail-user=sayarenedennis@northwestern.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name="clasml"
#SBATCH --output=/home/srd6051/run_classical_ml.out

. ~/anaconda3/etc/profile.d/conda.sh
conda activate bbcarenv

python classical_ml/ClassicalML/run_classical_ml.py \
    --input /projects/b1122/saya/06_modified_data/reg_thres_conf90_studyindex.csv \
    --label /projects/b1122/saya/bbcar_label_studyindex.csv \
    --outfn /home/srd6051/classical_ml_test.csv \
    --indexdir /projects/b1122/saya/indices/ \
    --scoring accuracy \
    --nmf 500 \
    --savemodel "true"

# --mem=0
