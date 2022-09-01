#!/bin/bash
set -e
EXEC=./gbcls.py
NJOBS=14
EXP_DIR=./output_tab
mkdir -p $EXP_DIR


# Classification
echo 'Classification'
LABELS=( 'material_group' 'place_country_code' 'technique_group' 'time_label' )

for LABEL in "${LABELS[@]}"
do
    echo $LABEL
    # set DATACOLS
    DATACOLS=("${LABELS[@]}")
    for i in "${!DATACOLS[@]}"; do
        if [[ ${DATACOLS[i]} = $LABEL ]]; then
            unset 'DATACOLS[i]'
        fi
    done
    DATACOLS+=("museum")

    # Run TRAIN/TUNE and DEV EVAL/PREDICT
    echo "Running Train/Tune for $LABEL"
    $EXEC \
        --output_dir "$EXP_DIR" \
        --train "trn" \
        --tune "dev" \
        --data ../data/dataset/dataset.tsv \
        --cols "${DATACOLS[@]}" \
        --target $LABEL \
		--n-jobs $NJOBS \
        --eval "dev" \
        --classify "dev"

    # Run TEST EVAL/PREDICT
    echo "Running EVAL/PREDICT for test $LABEL"
    $EXEC \
        --output_dir "$EXP_DIR" \
        --model-load "$EXP_DIR/$LABEL/model.pickle" \
        --data ../data/dataset/dataset.tsv \
        --cols "${DATACOLS[@]}" \
        --target $LABEL \
        --eval "tst" \
        --classify "tst"
    echo " "
    echo " "
done

