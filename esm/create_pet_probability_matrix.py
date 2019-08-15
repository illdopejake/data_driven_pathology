#!/usr/bin/env python

import sys
import glob
import re
sys.path.insert(0,'..')
import ESM_utils as esm
import pandas as pd
from argparse import ArgumentParser

def get_sub_ids(parametric_images):
    subs = []
    for path in parametric_images:
        sub=path.split("/")[-1].split("_")[0].split("-")[1]
        subs.append(sub)
    return subs

def get_dkt_labels():
    df = pd.read_csv("../../data/atlases/dst_labels.csv", header=None)
    cols = list(df[1][0:78])
    left_dkt_roi_cols = ['Left%s' % x for x in cols[0:39]]
    right_dkt_roi_cols = ['Right%s' % x for x in cols[39:78]]
    dkt_roi_cols = left_dkt_roi_cols + right_dkt_roi_cols
    return dkt_roi_cols


def create_prob_matrix(parametric_images,
                       atlas_images,
                       ref,
                       output_name,
                       dataset,
                       visit
                       ):
     output = esm.Prepare_PET_Data(files_in=parametric_images,
                                   atlases=atlas_images,
                                   ref=ref,
                                   orig_prob_method_matlab=True,
                                   save_matrix="return",
                                   );
     # remove the L & R cerebellum
     output = pd.read_csv("PET_data_roi_data.csv")
     output = output.drop(["roi_79", "roi_80"], axis=1)
     subs = get_sub_ids(parametric_images)
     dkt_roi_cols = get_dkt_labels()
     output.index = subs
     output.columns = dkt_roi_cols
     output.to_csv("../../data/" + dataset + "/pet_probability_matrices/" + output_name + "_" + visit + ".csv")

def main():
    parser = ArgumentParser()
    parser.add_argument("--parametric_files_dir",
                        help="Please pass the files directory containing the parametric PET images")
    parser.add_argument("--visit",
			help="")
    parser.add_argument("--dkt_atlas_dir",
                        help="")
    parser.add_argument("--ref_region_dir",
                        default=None,
                        help="Path to dir containing reference region images")
    parser.add_argument("--ref_regions",
                        nargs="+",
                        type=int,
                        default=None, 
                        help="Integer values for reference region labels")
    parser.add_argument("--output_name",
                        help="Please provide filename for PET probability matrix.")
    parser.add_argument("--dataset",
                        help="ADNI or DIAN")
    results = parser.parse_args()

    parametric_files_dir = results.parametric_files_dir
    visit = results.visit
    dkt_atlas_dir = results.dkt_atlas_dir
    ref_region_dir = results.ref_region_dir
    ref_regions = results.ref_regions
    dataset = results.dataset
    output_name = results.output_name

    if ref_region_dir is None and ref_regions is None: 
        raise IOError("Please provide either a ref region files dir or list of ref region labels")
    else: 
        if ref_region_dir is not None: 
            ref = sorted(glob.glob(ref_region_dir))
        elif ref_regions is not None: 
            ref = ref_regions
        
    parametric_images = sorted(glob.glob(parametric_files_dir))
    dkt_atlas_images = sorted(glob.glob(dkt_atlas_dir))

    #print(parametric_files_dir)

    create_prob_matrix(parametric_images,
                       dkt_atlas_images,
                       ref,
                       output_name,
                       dataset,
                       visit)

if __name__ == "__main__":
    main()










