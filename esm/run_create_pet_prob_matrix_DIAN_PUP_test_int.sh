#!/bin/bash 

python3.6 create_pet_probability_matrix.py --parametric_files_dir "/home/users/llevitis/ace_mount/ace_home/DIAN_PUP_output/sub-*/ses-v00/*pib*.nii.gz" \
  --dkt_atlas_dir "/home/users/llevitis/ace_mount/ace_home/DIAN_PUP_output/sub-*/ses-v00/*esm-regions_space-T1w.nii.gz" \
  --ref_regions 79 80 \
  --output_name "DIAN_PUP_coregistered_voxelwise_ecdf_orig_method_test_int" \
  --dataset "DIAN" \
  --visit "v00"
