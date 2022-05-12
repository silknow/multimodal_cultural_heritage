#!/bin/bash
# all
#material_group
./gbcls.py --output_dir ./output_multimodal --train dev --tune "dev" --data ../data/multimodal/dataset_multimodal.tsv --cols pred_img_material_group pred_txt_material_group pred_tab_material_group museum place_country_code technique_group time_label --target material_group --eval tst --n-jobs 14

#place_country_code
./gbcls.py --output_dir ./output_multimodal --train dev --tune "dev" --data ../data/multimodal/dataset_multimodal.tsv --cols pred_img_place_country_code pred_txt_place_country_code pred_tab_place_country_code museum material_group technique_group time_label --target place_country_code --eval tst --n-jobs 14

#technique_group
./gbcls.py --output_dir ./output_multimodal --train dev --tune "dev" --data ../data/multimodal/dataset_multimodal.tsv --cols pred_img_technique_group pred_txt_technique_group pred_tab_technique_group museum material_group place_country_code time_label --target technique_group --eval tst --n-jobs 14

#time_label
./gbcls.py --output_dir ./output_multimodal --train dev --tune "dev" --data ../data/multimodal/dataset_multimodal.tsv --cols pred_img_time_label pred_txt_time_label pred_tab_time_label museum material_group place_country_code technique_group --target time_label --eval tst --n-jobs 14
