import os
import sys
import pandas
import numpy as np
import nibabel as nib
from nilearn import image
from sklearn.mixture import GaussianMixture
from esm.ESM_utils import model_tfm, W_Transform

#### USER INPUT GOES HERE

# a list of paths to tau-PET scans 
scans = sorted(glob('/data1/users/jvogel/ADNI_tau/template_space/tau_images/*/TAU_PET/template_space/TAU_SUVr_*.nii'))
# path to a binary image mask
mask = '/home/users/jvogel/Science/templates/masks/ADNI_GM_mask_1p5mm_nocereb_thr0p9.nii'
# directory to save images
oudir = '/data1/users/jvogel/ADNI_tau/wtest/'

#### SCRIPT STARTS HERE.
#### DONT EDIT BELOW THIS LINE UNLESS YOU KNOW WHAT YOURE DOING
##################################################################
def img_wscore_wrapper(scans,mask=None,
                       outdir = None,outnm='TFMd',
                       save_style='3d'):
    ''' Will transform images by using mixture modeling to determine a "normal"
    distribution, and then normalizing ALL values to values along the "normal" 
    distribution. This procedure is done voxelwise, and voxels that do not have
    a binomial distribution are simply z-scored.

    NOTE: Input scans should be in the same space (i.e. a standard space) and
    should be smoothed for best results. 

    NOTE: Providing an image mask (see arguments below) will speed up the 
    procedure quite a bit.

    Parameters
    ----------
    scans : list
        A list of (str) paths to input images.

    mask: str, optional
        If a path to a mask image is passed, all images will be masked using
        the mask image. Note that the mask image must be a binary image in the 
        same space as images in "scans". Passing a mask image will greatly speed 
        up computation. Passing None will perform transformation across all 
        voxels. 

    outdir: str, optional
        Where to save the images to after transformation. If None is passed,
        images will output to current working directory

    outnm: str, optional, default 'TFMd'
        String to append to the front of all images created post-transformation.

    save_style: str, optional, default '3d'
        String describing in what format to save files.
        Must be one of::

            '3d': Save each new scan individually.
            '4d': Save all new scans into a single 4D image.

        Note that in the '3d' case, image filenames will contain integers
        reflecting the input order in scans. For example, the image pointed to
        by the third path in scans will be labeled as "2".


    '''
    
    if save_style not in ['3d','4d']:
        raise IOError("save style must be set to '3d' or '4d'")
        
    if not outdir:
        print('WARNING: No output directory passed. Saving images to currend working directory')
        outdir = os.get_cwd()
        
    print('loading images')
#     if input_style=='nilearn':
    i4d,aff = load_images_nil(scans)
#     elif input_style=='nib':
#         i4d,aff = load_images_nib(scans,verbose)

    masked = False
    if mask:
        if type(mask) == str:
            if os.path.isfile(mask):
                print('masking')
                i2d,mskr = mask_nil(i4d,aff,mask)
                masked = True
            else:
                raise IOError('could not find any image at path: %s'%mask)
    else:
        print('setting to 2D')
        i2d, old_shape = to_2d(i4d)

    print('beginning transformation')
    transformed = img_wscore(i2d)

    print('reverting back to image format')
    if masked:
        tfm_4d = mskr.inverse_transform(transformed).values
    else:
        tfm_4d = back_to_4d(transformed,old_shape)
    
    if save_style == '4d':
        tfm_img = nib.Nifti1Image(tfm_4d,aff)
        flnm = os.path.join(outdir,outnm)
        tfm_img.to_filename(flnm)
    elif save_style == '3d':
        for i in range(tfm_4d.shape[-1]):
            tfm_img = nib.Nifti1Image(tfm_4d[:,:,:,i],aff)
            flnm = os.path.join(outdir,outnm+'_%s'%i)
            tfm_img.to_filename(flnm)
    
def load_images_nib(scans,verbose):
    imgs = []
    shape_getter = nib.load(scans[0])
    aff = shape_getter.affine
    x,y,z = shape_getter.shape
    if not verbose:
        print('loading scans')
    for scan in scans:
        if verbose:
            print('loading',scan)
        img = nib.load(scan)
        data = img.get_data().reshape(x,y,z,1)
        imgs.append(data)
    print('concatenating')
    i4d = np.concatenate(imgs,axis=3)
    return i4d,aff

def load_images_nil(scans):
    i4d = image.concat_imgs(scans)
    aff = i4d.affine
    i4d = i4d.get_data()
    return i4d,aff

def img_wscore(mskd):
    ndata = []
    models = {'one_comp': GaussianMixture(n_components=1,random_state=123),
              'two_comp': GaussianMixture(n_components=2,random_state=123, 
                                          tol=0.00001, max_iter=1000)}
    mskd = pandas.DataFrame(mskd)
    mskd.columns = [str(x) for x in mskd.columns]
    n_cols = len(mskd.columns)
    for i,col in enumerate(mskd.columns):
        if i%1000 == 0:
            print('working on voxel %s of %s'%((i+1),n_cols))
        if all(mskd[col]==0):
            jnk = mskd[col].values
            ndata.append(jnk.reshape(len(jnk),1))
        else:
            tfm, report = model_tfm(mskd[col],mskd[col],models,'left','nan')
            w_input = pandas.DataFrame(mskd[col])
            if report['model'] != 'two_comp':
                wtfm = (mskd[col].values - mskd[col].mean()) / mskd[col].std()
                ndata.append(wtfm.reshape(len(wtfm),1))
            else:
                mod = models['two_comp']
                mod.fit(mskd[col].values.reshape(-1,1))
                labels = mod.predict(mskd[col].values.reshape(-1,1))
                norm_idx = w_input.index[np.where(labels==0)]
                wtfm = W_Transform(w_input,norm_index=norm_idx)
                ndata.append(wtfm.values)
    WTfm = np.concatenate(ndata, axis=1)
    
    return WTfm

def mask_nil(img_data,affine,mask):
    from nilearn import input_data
    nimg = nib.Nifti1Image(img_data, affine)
    mskr = input_data.NiftiMasker(mask)
    mskd = mskr.fit_transform(nimg)
    return mskd, mskr

def to_2d(i4d):
    holder = []
    x,y,z,t = i4d.shape
    for i in range(t):
        arr = i4d[:,:,:,i].flatten()
        arr = arr.reshape(1,len(arr))
        holder.append(arr)
    i2d = np.concatenate(holder,axis=0)
    old_shape = [x,y,z]
    
    return i2d,old_shape

def back_to_4d(i2d, old_shape):
    holder = []
    x,y,z = tuple(old_shape)
    for i in range(len(i2d)):
        mtx = i2d[i,:].reshape(x,y,z,1)
        holder.append(mtx)
    i4d = np.concatenate(holder,axis=3)
    return i4d

if __name__ == '__main__':

    img_wscore_wrapper(scans,mask,outdir,outnm='TFMd', save_style='3d')