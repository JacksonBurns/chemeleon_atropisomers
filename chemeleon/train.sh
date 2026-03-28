REPETITIONS=(0 1 2 3 4)
FOLDS=(0 1 2 3 4)
mkdir -p output

for repetition_number in "${REPETITIONS[@]}"; do
    mkdir -p "output/repetition_${repetition_number}"
    for fold_number in "${FOLDS[@]}"; do

        OUTDIR="output/repetition_${repetition_number}/fold_${fold_number}"
        LOGFILE="${OUTDIR}/log.txt"

        chemprop train \
            --output-dir "${OUTDIR}" \
            --logfile "${LOGFILE}" \
            --data-path \
                "../cv_splits/repetition_${repetition_number}/fold_${fold_number}/train.csv" \
                "../cv_splits/repetition_${repetition_number}/fold_${fold_number}/val.csv" \
                "../cv_splits/repetition_${repetition_number}/fold_${fold_number}/test.csv" \
            --descriptors-columns T \
            --from-foundation CheMeleon \
            --pytorch-seed 42 \
            --smiles-columns SMILES \
            --target-columns "ΔG(kcal/mol)" \
            --task-type regression \
            --patience 5 \
            --init-lr 0.0001 \
            --warmup-epochs 5 \
            --loss mse \
            --metrics rmse r2 mse mae \
            --show-individual-scores \
            --ffn-num-layers 2 \
            --ffn-hidden-dim 512 \
            --batch-size 32 \
            --epochs 50
        
        mv ${OUTDIR}/model_0/test_predictions.csv ../cv_splits/repetition_${repetition_number}/fold_${fold_number}/chemeleon_pred.csv
    done
done
