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
# beta_dir = 
# spreadsheet = 
# outdir = 
# # THESE ARGUMENTS ARE DEFAULT BUT CAN BE CHANGED TO FIT USE CASES
# sdres_img =  ''
# beta_str = 'beta'
# intercept = 'first'
# res_str = 'Res_'
# cols_to_use = []
# img_mask = None
# subject_col = ''
# memory_load = 'low'

def main(image_dir, beta_dir, spreadsheet, outdir, sdres_img = '', beta_str = 'beta',
                           intercept='first', res_str = 'Res_', 
                          cols_to_use = [], img_mask = None, subject_col = ''
                          ):
    '''This script will create W-SCORE images from a set of input images, and a
    spreadsheet. This script makes several assumptions about the inputs:
    
    1) The "image_dir" path must ONLY contain input images, and no other files or 
        directories
    2) The rows of your spreadsheet MUST be in the exact same order as the images 
        in the "image_dir directory. So subject 1 should be row 1 of the spreadsheet 
        (not including headers)
    3) The values in the dataframe should all be of a numeric type (int or float)
    4) The script assumes you have run an SPM model for the variables included in 
        the W-SCORE, and that you thus have BETA images for each of those variables
    5) The number of BETA images (not including the intercept) should be in the exact 
        same order as either the columns of your spreadsheet, or the columns in the
        cols_to_use argument.
    6) If cols_to_use is not passed, script assumes that spreadsheet only includes 
        columns for which BETA images exist, as well as a subject column (but only
        if the subject_col argument is passed)
    7) The script assumes you either have an image representing the Standard
        Deviation of the Residuals, or you have asked SPM to create the residuals in 
        the model described in 3)
    
    If all of these assumptions have been met, you may proceed.
    
    image_dir = an directory containing ONLY raw images to be W-transformed (REQUIRED)
    beta_dir = a directory containing beta images generated from an SPM model (REQUIRED)
    spreadsheet = a spreadsheet with one row for each subject in image_dir (same order),
        and columns indicating values for variables to be accounted for (e.g. age, sex),
        which correspond to, and are in the same order as, beta maps in beta_dir. (REQUIRED)
    outdir = a directory to store the transformed output files (does not have to exist) (REQUIRED)
    sdres_img = a path to an image representing the Standard Deviation of the Residuals (of the 
        normative model). Or, leave blank and script will create one for you using residual
        images created by the SPM model (Res_str must be passed)
    beta_str = string labeling beta images from the SPM model (the text in front of the image 
        names)
    intercept = SPM puts the intercept as the first or last image, depending on the version.
        Enter "first" if its first, or "last" if its last. If you did not include the
        intercept in the model, pass "none". (WARNING, "none" is in beta and will crash)
    res_str = if you did not pass an option for sdres_img, you must an argument here. This
        is the string labels of Residual images created by the SPM model.
        **WARNING** This may crash your computer if you don't have sufficient memory
    cols_to_use = a list of columns from df to use can be specified here. If your df has 
        columns you do not wish to use in the w-scoring, you can specify the names of the
        columns you do wish to use (be sure they are in the exact same order as the beta
        images in the beta_dir)
    img_mask = path to a binarized mask image in the same dimensions as your input images. If
        passed, the w-score images will be masked (RECOMMENDED)
    subject_col = name of a column in df with the subject IDs. This will automatically label
        the w-score images with the subject IDs from this columns
    
    '''
    
    # check inputs
    print('initating and checking inputs')
    for cdir in [image_dir, beta_dir]:
        if type(cdir) != str:
            raise TypeError('%s must be a path pointing to a valid directory, you passed a %s object'%(cdir,
                                                                                                    type(cdir)))
        if not os.path.isdir(image_dir):
            raise IOError('could not find the directory specified in argument "image_dir"')
    if type(outdir) != str:
        raise TypeError('%s must be a path pointing to a valid directory, you passed a %s object'%(outdir,
                                                                                                type(cdir)))
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    
    if intercept not in ['first','last', 'none']:
        raise IOError('intercept must be set to "first", "last" or "none". See docstring for more info')
    
    # load inputs
    raw_paths = glob(os.path.join(image_dir,'*.ni*'))
    print('found %s images to transform'%len(raw_paths))
    
    beta_paths = glob(os.path.join(beta_dir,'%s*'%beta_str))
    if len(beta_paths) == 0:
        raise IOError('No beta images found in specified directory. Please revise beta_dir or beta_str arguments')
    if intercept == 'none':
        n_betas = len(beta_paths)
    else:
        n_betas = len(beta_paths) - 1
    if len(cols_to_use) > 0:
        if len(cols_to_use) != n_betas:
            raise IOError('# of columns in cols_to_use (%s) must match number of non-intercept beta maps (%s)'%(
                                                                                            len(cols_to_use),
                                                                                            n_betas))

    print('preparing spreadsheet')
    if type(spreadsheet) == str:
        if '.csv' in spreadsheet:
            df = pandas.read_csv(spreadsheet)
        elif 'xl' in spreadsheet[-4:]:
            try:
                jnk = pandas.ExcelFile(spreadsheet)
                df = pandas.ExcelFile(jnk).parse(jnk.sheet_names[0])
            except:
                raise IOError('could not load excel file. Please convert to csv')
        else:
            try:
                df = pandas.read_table(spreadsheet)
            except:
                raise IOError('could not read spreadsheet. Try loading the df yourself with pandas and inputting that')
    else:
        df = pandas.DataFrame(spreadsheet,copy=True)
    if subject_col:
        if type(subject_col) == int:
            subject_col = df.columns[subject_col]
            subject_IDs = df[subject_col].values
            print('using %s for subject IDs'%subject_col)
        else:
            if subject_col not in df.columns:
                raise IOError('could not find %s in any of the columns of your dataframe: %s'%(subject_col,
                                                                                   df.columns))
            else:
                subject_IDs = df[subject_col].values
        df.drop(subject_col, axis=1, inplace=True)
    else:
        subject_IDs = []
    if len(cols_to_use) > 0:
        df = df[cols_to_use]
    if df.shape[0] != len(raw_paths):
        raise IOError('number of scans (n=%s) does not match number of rows in spreadsheet (n=%s)'%(len(raw_paths),
                                                                                               df.shape[0]))
    
    print('loading beta images')
    if intercept == 'first':
        int_img = ni.load(beta_paths[0]).get_data()
        beta_paths.remove(beta_paths[0])
    elif intercept == 'last':
        int_img = ni.load(beta_paths[-1]).get_data()
        beta_paths.remove(beta_paths[-1])
    else:
        int_img = None
    jnk = ni.concat_images(beta_paths)
    beta_imgs = jnk.get_data()
    aff = jnk.affine
    if len(beta_imgs.shape) > 4:
        try:
            x,y,z = beta_imgs.shape[:3]
            s = beta_imgs.shape[-1]
            beta_imgs.reshape(x,y,z,s)
        except:
            raise IOError('shape of beta images is %s, expecting a set of 3D images (so 4d)'%beta_imags.shape)
    
    if beta_imgs[:,:,:,0].shape != ni.load(raw_paths[0]).shape:
        raise IOError('inconsistent dimensions between betas and raw images')
    
    if sdres_img:
        if os.path.isfile(sdres_img):
            sdres_img = ni.load(sdres_img).get_data()
        else:
            raise IOError('could not find any SD of residual images at path %s'%sdres)
    else:
        sdres_img = create_sdres_img(res_str, beta_dir)
    
    if type(img_mask) != type(None):
        try:
            mask = ni.load(img_mask).get_data()
        except:
            raise IOError('could not load mask. Please ensure the path points to an existing nifti image')
        if mask.shape != ni.load(raw_paths[0]).shape:
            raise IOError('dimensions of mask (%s) do not match the dimensions of input images (%s)'%(mask.shape,
                                                                                                   ni.load(raw_paths[0]).shape))
    else:
        mask = None
    
    w_transform(beta_imgs, int_img, raw_paths,  sdres_img, 
                df, subject_IDs, aff, mask, outdir)
    
    print('FINISHED! W-SCORE images written to %s'%outdir)
    
def create_sdres_img(res_str, beta_dir):
    
    res_paths = glob(os.path.join(beta_dir,'%s*'%res_str))
    print('calculating standard deviation of the residuals')
    print('loading res images...')
    res_imgs = ni.concat_images(res_paths).get_data()
    print('calculating...')
    sdres_img = res_imgs.std(ddof=1,axis=3)
    
    return sdres_img
    
def w_transform(beta_imgs, int_img, raw_paths, sdres_img, 
                df, subject_IDs, aff, mask, outdir):
        
    print('performing w-score transformations...')
    x,y,z = ni.load(raw_paths[0]).shape
    for i,scan in enumerate(raw_paths):
        if len(subject_IDs) > 0:
            sid = subject_IDs[i]
        else:
            sid = os.path.split(scan).split('.')[0]
        coefs = []
        if type(int_img) != type(None):
            coefs.append(int_img.reshape(x,y,z,1))
        for j,col in enumerate(df.columns):
            val = df.loc[df.index[i],col]
            val_img = np.full_like(beta_imgs[:,:,:,j],val) 
            coef = beta_imgs[:,:,:,j] * val_img
            #coef = np.multiply(beta_imgs[:,:,:,j],val)
            coefs.append(coef.reshape(x,y,z,1))
        jnk = np.concatenate(coefs,axis=3)
        predicted = jnk.sum(axis=3)
        observed = ni.load(scan).get_data()
        wscr_mat = (observed - predicted) / sdres_img
        if type(mask) != type(None):
            wscr_mat[mask==0] = 0
        wscr_img = ni.Nifti1Image(wscr_mat,aff)
        flnm = os.path.join(outdir,'WSCORE_%s'%(sid))
        wscr_img.to_filename(flnm)
        print('finished %s, %s of %s'%(sid,i+1,len(raw_paths)))


if __name__ == '__main__':

    main(image_dir, beta_dir, spreadsheet, outdir, sdres_img, beta_str,
                           intercept, res_str, 
                          cols_to_use , img_mask, subject_col,
                          memory_load)
