import os
import pandas
import numpy as np
import nibabel as ni
from glob import glob

# IF RUNNING THIS FROM THE COMMAND-LINE (i.e. not directly from an interpreter), 
# UN-COMMENT THE FOLLOWING LINES AND FILL IN THE VARIABLES. (SEE DOCUMENTATION
# BELOW)

# # THESE ARGUMENTS ARE REQUIRED
# image_dir = 
# LDA_dir =
# # THESE ARGUMENTS ARE OPTIONAL AND SHOULD BE CHANGED ACCORDING TO USE CASE
# LDA_str = 'factor'
# mask_image = 'auto'
# outdir = './'
# outname = 'LDA_scores'
# save_results = True
# return_results = False

def transform_by_LDA_factor(image_dir, LDA_dir, LDA_str = 'factor',
                            mask_image = 'auto', outdir = './', 
                            outname = 'LDA_scores',
                           save_results = True, 
                           return_results = False):

    '''This function will apply LDA factors (expressed as .nii files) to
    existing images in order to derive LDA factor scores. The funtion
    will output a .csv with one score for each factor, for each subject.

    image_dir = a directory containing nifti images to be transformed.
        This directory should not contain any other nifti images except those
        that the user wants factor scores extracted from
    LDA_dir = a directory containing the LDA factor images
    LDA_str = text in front of the LDA factor images indicating they are the
        correct images. For example, if you did ls LDA_str*, it would only
        find the LDA factor images.
    mask_image = An image to mask the factor scoring.
        If None, assumes no mask was used during LDA procedure (UNLIKELY)
        If a path to a binary mask image, will mask analysis with the image
        If 'auto', script will try to figure out where the LDA was masked
        *WARNING* inconcistency with the original LDA input can lead to
        incorrect scores
    outdir = a directory to write the results to
    outname = a filename for the results. Note: don't add an extension (e.g. .csv)
    save_results = If True, results will be written to file based on outdir and 
        outname
    return_results = If True, results will be returned to python environment
        (not recommended unless using an interpreter like iPython)

    '''
    
    scans = glob(os.path.join(image_dir, '*.ni*'))
    print('found %s scans to transform'%len(scans))
    factor_paths = glob(os.path.join(LDA_dir,'%s*.ni*'%LDA_str))
    print('found %s LDA factors'%len(factor_paths))
    
    ids = [os.path.split(x)[-1].split('.')[0] for x in scans]
    fac_ids = [os.path.split(x)[-1].split('.')[0] for x in factor_paths]
    results = pandas.DataFrame(index = ids, columns = fac_ids)
    
    factors = ni.concat_images(factor_paths).get_data()
    if mask_image and mask_image != 'auto':
        mask = ni.load(mask_image).get_data()
        for j in range(factors.shape[-1]):
            factors[:,:,:,j][mask==0] = 0
    elif mask_image == 'auto':
        mcoords = np.where(factors[:,:,:,0]==0)
        mask = np.ones_like(factors[:,:,:,0])
        mask[mcoords] = 0
    
    for i,scan in enumerate(scans):
        sid = ids[i]
        print('transforming %s, %s of %s'%(sid, (i+1), len(scans)))
        dat = ni.load(scan).get_data()
        if mask_image:
            dat[mask==0] = 0
        for j in range(factors.shape[-1]):
            score = np.dot(dat.flat, factors[:,:,:,j].flat)
            results.loc[sid, fac_ids[j]] = score
    
    if save_results:
        flnm = os.path.join(outdir,'%.csv'%outname)
        results.to_csv(flnm)
        print('results written to %s'%flnm)
    
    if return_results:
        return results

if __name__ == '__main__':

    transform_by_LDA_factor(image_dir, LDA_dir, LDA_str,
                            mask_image, outdir, outname,
                           save_results, return_results)
    