import os
import pandas
import numpy as np
import nibabel as ni
import itertools
from glob import glob
import statsmodels.distributions.empirical_distribution as ed
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.io import savemat,loadmat
from matplotlib import mlab
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from statsmodels.sandbox.stats.multicomp import multipletests

def Extract_Values_from_Atlas(files_in, atlas, 
                              mask = None, mask_threshold = 0,
                              blocking = 'one_at_a_time', 
                              labels = [], sids = [], 
                              output = None,):
    '''
    This function will extract mean values from a set of images for 
    each ROI from a given atlas. Returns a Subject x ROI pandas
    DataFrame (and csv file if output argument is set to a path).
    
    Use blocking argument according to memory capacity of your
    computer vis-a-vis memory requirements of loading all images.
    
    files_in: determines which images to extract values from. Input 
    can be any of the following:
        -- a list of paths
        -- a path to a directory containing ONLY files to extract from
        -- a search string (with wild card) that would return all
        desired images. For example, doing ls [files_in] in a terminal 
        would list all desired subjects
        -- a 4D Nifti image
        **NOTE** be aware of the order of file input, which relates to 
        other arguments
        
    atlas: Path to an atlas, or a Nifti image or np.ndarry of desired 
    atlas. Or, if doing native space analysis, instead, supply a list 
    of paths to atlases that match each subject. 
        NOTE: In this case, The order of this list should be the same
        order as subjects in files_in
    
    mask: Path to a binary inclusive mask image. Script will set all 
    values to 0 for every image where mask voxels = 0. This process
    is done before extraction. If doing a native space analysis,
    instead, supply a list of paths to masks that match each subject
    and each atlas.
    
    mask_threshold: An integer that denotes the minimum acceptable
    size (in voxels) of an ROI after masking. This is to prevent
    tiny ROIs resulting from conservative masks that might have
    spuriously high or low mean values due to the low amount of
    information within.
    
    blocking: loading all images to memory at once may not be possible
    depending on your computer. Acceptable arguments are:
        -- 'one_at_a_time': will extract values from each image
        independently. Recommended for memories with poor memory 
        capacity. Required for native space extraction.
        -- 'all_at_once': loads all images into memory at once.
        Provides a slight speed up for faster machines overe
        one_at_a_time, but is probably not faster than batching (see
        below). Only recommended for smaller datasets.
        ** WARNING ** Not recommended on very large datasets. Will
        crash computers with poor memory capacity.
        -- any integer: determines the number of images to be read to
        memory at once. Recommended for large datasets. 
    
    labels: a list of string labels that represent the names of the
    ROIs from atlas. 
        NOTE: ROIs are read consecutively from lowest to highest, and
        labels *must* match that order
    Default argument [] will use "ROI_x" for each ROI, where X
    corresponds to the actual ROI integer lael
    
    sids: a list of subject IDs in the same order as files_in. Default 
    argument [] will list subjects with consecutive integers.
    
    output: if you wish the resulting ROI values to be written to file,
    provide a FULL path. Otherwise, leave as None (matrix will be 
    returned)
    
    '''
    
    if type(blocking) == str and blocking not in ['all_at_once','one_at_a_time']:
        raise IOError('blocking only accepts integers or argumennts of "all_at_once" or "one_at_a_time"')
    
    if type(atlas) == list: 
        if blocking != 'one_at_a_time':
            print('WARNING: you have passed a list of atlases but blocking is not set to one_at_a_time')
            print('Lists of atlases are for native space situations where each subject has their own atlas')
            print('If you want to test multiple atlases, run the script multiple times with different atlases')
            raise IOError('you have passed a list of atlases but blocking is not set to one_at_a_time')
    
    if type(mask) != type(None):
        if type(atlas) != type(mask):
            raise IOError('for masking, list of masks must be passed that equals length of atlas list')
        elif type(mask) == list:
            if len(atlas) != len(mask):
                raise IOError('list of atlases (n=%s) and masks (n=%s) are unequal'%(len(atlases),
                                                                                    len(masks)))
        
    if type(atlas) != list:
        if type(atlas) == str:
            try:
                atl = ni.load(atlas).get_data()
            except:
                raise IOError('could not find an atlas at the specified location: %s'%atlas)
        elif type(atlas) == ni.nifti1.Nifti1Image:
            atl = atlas.get_data()
        elif type(atlas) == np.ndarray:
            atl = atlas
        else:
            print('could not recognize atlas filetype. Please provide a path, a NiftiImage object, or an numpy ndarray')
            raise IOError('atlas type not recognized')
        
        if blocking == 'all_at_once':
            i4d = load_data(files_in, return_images=True).get_data()
            if i4d.shape[:-1] != atl.shape:
                raise IOError('image dimensions do not match atlas dimensions')
            if type(mask) != type(None):
                print('masking...')
                mask_data = ni.load(mask).get_data()
                mask_data = mask_atlas(mask_data, atl, mask_threshold)
                i4d = mask_image_data(i4d, mask_data)
            if len(sids) == 0:
                sids = range(i4d.shape[-1])
            print('extracting values from atlas')
            roi_vals = generate_matrix_from_atlas(i4d, atl, labels, sids)
        else:
            image_paths = load_data(files_in, return_images = False)
            if blocking == 'one_at_a_time':
                catch = []
                for i,image_path in enumerate(image_paths):
                    if len(sids) > 0: 
                        sid = [sids[i]]
                    else:
                        sid = [i]
                    print('working on subject %s'%sid[0])
                    img = ni.load(image_path).get_data()
                    try:
                        assert img.shape == atl.shape, 'fail'
                    except:
                        print('dimensions for subject %s (%s) image did not match atlas dimensions (%s)'%(sid,
                                                                                 img.shape,
                                                                                 atl.shape))
                        print('skipping subject %s'%sid[0])
                        continue
                    if type(mask) != type(None):
                        mask_data = ni.load(mask).get_data()
                        mask_data = mask_atlas(mask_data, atl, mask_threshold)
                        img = mask_image_data(img, mask_data)
                    f_mat = generate_matrix_from_atlas(img, atl, labels, sid)
                    catch.append(f_mat)
                roi_vals = pandas.concat(catch)
            elif type(blocking) == int:
                block_size = blocking
                if len(image_paths)%block_size == 0:
                    blocks = int(len(image_paths)/block_size)
                    remainder = False
                else:
                    blocks = int((len(image_paths)/blocking) + 1)
                    remainder = True
                catch = []
                count = 0
                if type(mask) != type(None):
                    mask_data = ni.load(mask).get_data()
                    mask_data = mask_atlas(mask_data, atl, mask_threshold)
                for block in range(blocks):
                    if block == (blocks - 1) and remainder:
                        print('working on final batch of subjects')
                        sub_block = image_paths[count:]
                    else:
                        print('working on batch %s of %s subjects'%((block+1),block_size))
                        sub_block = image_paths[count:(count+block_size)]
                    i4d = load_data(sub_block, return_images = True).get_data()
                    if i4d.shape[:-1] != atl.shape:
                        raise IOError('image dimensions (%s) do not match atlas dimensions (%)'%(atl.shape,
                                                                                                i4d.shape[:-1]
                                                                                                ))
                    if type(mask) != type(None):
                        if len(mask_data.shape) == 4:
                            tmp_mask = mask_data[:,:,:,:block_size]
                        else:
                            tmp_mask = mask_data
                        i4d = mask_image_data(i4d, tmp_mask)
                    if block == (blocks - 1) and remainder:
                        if len(sids) == 0:
                            sids_in = range(count,i4d.shape[-1])
                        else:
                            sids_in = sids[count:]
                    else:
                        if len(sids) == 0:
                            sids_in = range(count,(count+block_size))
                        else:
                            sids_in = sids[count:(count+block_size)]
                    f_mat = generate_matrix_from_atlas(i4d, atl, labels, sids_in)
                    catch.append(f_mat)
                    count += block_size
                roi_vals = pandas.concat(catch)
    else:
        image_paths = load_data(files_in, return_images = False)
        if len(atlas) != len(image_paths):
            raise IOError('number of images (%s) does not match number of atlases (%s)'%(len(image_paths),
                                                                                      len(atlas)))
        catch = []
        for i,image_path in enumerate(image_paths):
            if len(sids) > 0: 
                sid = [i]
            else:
                sid = [sids[i]]
            print('working on subject'%sid)
            img = ni.load(image_path).get_data()
            atl = ni.load(atlas[i]).get_data()
            if type(mask) != type(None):
                mask_data = ni.load(mask[i]).get_data()
                mask_data = mask_atlas(mask_data, atl, mask_threshold)
                img = mask_image_data(img,mask_data)
            try:
                assert img.shape == atl.shape, 'fail'
            except:
                print('dimensions for subject %s (%s) image did not match atlas dimensions (%s)'%(sid,
                                                                                                 img.shape,
                                                                                                 atl.shape
                                                                                                 ))
                print('skipping subject %s'%sid)
                continue
            f_mat = generate_matrix_from_atlas(img, atl, labels, sid)
            catch.append(f_mat)
        roi_vals = pandas.concat(catch)    

    if output:
        roi_vals.to_csv(output)
    return roi_vals
    
def generate_matrix_from_atlas(files_in, atl, labels, sids):
    
    if len(files_in.shape) == 3:
        x,y,z = files_in.shape
        files_in = files_in.reshape(x,y,z,1)
    atl = atl.astype(int)
    if max(np.unique(atl)) != (len(np.unique(atl)) -1):
        atl = fix_atlas(atl)
    if len(labels) > 0:
        cols = labels
    else:
        cols = ['roi_%s'%x for x in np.unique(atl) if x != 0]
    f_mat = pandas.DataFrame(index = sids,
                             columns = cols)
    tot = np.bincount(atl.flat)
    for sub in range(files_in.shape[-1]):
        mtx = files_in[:,:,:,sub]
        sums = np.bincount(atl.flat, weights = mtx.flat)
        rois = (sums/tot)[1:]
        f_mat.loc[f_mat.index[sub]] = rois
    
    return f_mat
    

def load_data(files_in, return_images):
    
    fail = False
    
    if type(files_in) == str:
        if os.path.isdir(files_in):
            print('It seems you passed a directory')
            search = os.path.join(files_in,'*')
            flz = glob(search)
            num_f = len(flz)
            if num_f == 0:
                raise IOError('specified directory did not contain any files')
            else:
                print('found %s images!'%num_f)
            if return_images:
                i4d = ni.concat_images(flz)
        elif '*' in files_in:
            print('It seems you passed a search string')
            flz = glob(files_in)
            num_f = len(flz)
            if num_f == 0:
                raise IOError('specified search string did not result in any files')
            else:
                print('found %s images'%num_f)
            if return_images:
                i4d = ni.concat_images(flz)
        else:
            fail = True
    elif type(files_in) == list:
        flz = files_in
        print('processing %s subjects'%len(files_in))
        if return_images:
            i4d = ni.concat_images(files_in)
    elif type(files_in) == ni.nifti1.Nifti1Image:
        print('processing %s subjects'%files_in.shape[-1])
        i4d = files_in
    else:
        fail = True
        
    if fail:
        print('files_in not recognized.', 
                    'Please enter a search string, valid directory, list of paths, or a Nifti object')
        raise ValueError('I do not recognize the files_in input.')
    
    if return_images:
        return i4d
    else:
        return flz
    
def mask_image_data(image_data, mask_data):
    
    if len(image_data.shape) == 3:
        if mask_data.shape != image_data.shape:
            raise ValueError('dimensions of mask (%s) and image (%s) do not match!'%(mask_data.shape,
                                                                                    image_data.shape))
        image_data[mask_data==0] = 0
    
    elif len(image_data.shape) == 4:
        if len(mask_data.shape) == 4:
            if mask_data.shape != image_data.shape:
                raise ValueError('dimensions of mask (%s) and image (%s) do not match!'%(mask_data.shape,
                                                                                        image_data.shape))
            else:
                masker = mask_data
        else:
            if mask_data.shape != image_data.shape[:3]:
                raise ValueError('dimensions of mask (%s) and image (%s) do not match!'%(mask_data.shape,
                                                                                        image_data.shape[:3]))
            masker = np.repeat(mask_data[:, :, :, np.newaxis], image_data.shape[-1], axis=3)
        image_data[masker==0] = 0
    
    return image_data

def mask_atlas(mask_data, atlas_data, mask_threshold):
    
    if len(mask_data.shape) == 4:
        dim4 = mask_data.shape[-1]
        mask_data = mask_data[:,:,:,0]
        tfm_4d = True
    else:
        tfm_4d = False
        
    if max(np.unique(atlas_data)) != (len(np.unique(atlas_data)) -1):
        atlas_data = fix_atlas(atlas_data)
    mask_atlas = np.array(atlas_data, copy=True)
    new_mask = np.array(mask_data, copy=True)
    mask_atlas[mask_data == 0] = 0
    counts = np.bincount(mask_atlas.astype(int).flat)
    labs_to_mask = [x for x in range(len(counts)) if counts[x] < mask_threshold]
    for label in labs_to_mask:
        new_mask[atlas_data==label] = 0
    
    if tfm_4d:
        new_mask = np.repeat(new_mask[:, :, :, np.newaxis], dim4, axis=3)
    
    return new_mask
    
def fix_atlas(atl):
    new_atl = np.zeros_like(atl)
    atl_map = dict(zip(np.unique(atl),
                       range(len(np.unique(atl)))
                      ))
    for u in np.unique(atl):
        new_atl[atl == u] = atl_map[u]
    
    return new_atl


def Convert_ROI_values_to_Probabilities(roi_matrix, norm_matrix = None,
                                        models = None,
                                        target_distribution = 'right',
                                        outdir = False, fail_behavior = 'nan',
                                        mixed_probability = False, mp_thresh = 0.05):
    '''
    Will take a Subject x ROI array of values and convert them to probabilities,
    using ECDF (monomial distribution) or Gaussian Mixture models (binomial
    distribution), with or without a reference sample with the same ROIs.
    
    Returns a Subject x ROI matrix the same size as the input with probability
    values. A report is also generated if an argument is passed for models. The
    report details which model was selected for each ROI and notes any problems.
    
    roi_matrix -- A subject x ROI matrix. can be pandas Dataframe or numpy array
    
    norm_matrix -- A matrix with the same ROIs as roi_matrix. This sample will
    be used to fit the distributions used to calculate the probabilities of
    subject in roi_matrix. Norm_matrix and roi_matrix can have overlapping
    subjects
        if None (default), will use roi_matrix as norm_matrix
    
    models -- a dict object  pairing sklearn.gaussian models (values) with
    labels describing the models (keys). If more than one model is passed,
    for each ROI, model fit between all models will be evaluated and best model
    (lowest BIC) will be selected for that ROI.
        if None (default), probabilities will be calculated using ECDF.
        NOTE: Models with n_components=1 will be calculate probabilities using 
        ECDF. 
        NOTE: This script does not currently support models with
        n_distributions > 2
    
    target_distribution -- Informs the script whether the target distribution is
    expected to have lower values ('left', e.g. gray matter volume) or higher values
    ('right', e.g. tau-PET). The target distribution is the one for which
    probabilities are generated. For example, passing a value of 'right' will give
    the probability that a subject falls on the rightmost distribution of values for
    a particular ROI.
    
    outdir -- If the resulting probability matrix (and report) should be save to disk,
    provide the path to an existing directory.
        WARNING: Will overwrite already-existing outcome of this script one already
        exists in the passed directory
    
    fail_behavior -- Occasionally, two-component models will find distributions that
    are not consistent with the hypothesis presented in target_distribution.
    This argument tells the script what to do in such situations:
        'nan' will return NaNs for all ROIs that fail
        'values' will return probability values from one the distributions (selected
        arbitrarily)
        
    mixed_probability -- Experimental setting. If set to True, after calculating
    probabilities, for rois with n_components > 1 only, will set all values < 
    mp_thresh to 0. Remaining values will be put through ECDF. This will create less
    of a binarized distribution for n_components > 1 ROIs.
    
    mp_thresh -- Threshold setting for mixed_probability. Must be a float between 0
    and 1. Decides the arbitrary probability of "tau positivity". Default is 0.05.
    
    '''

    if target_distribution not in ['left','right']:
        raise IOError('target_distribution must be set to "left", "right" or None')
    
    if fail_behavior not in ['nan', 'values']:
        raise IOError('fail_behavior must be set to "nan" or "values"')
    
    if type(roi_matrix) == pandas.core.frame.DataFrame:
        roi_matrix = pandas.DataFrame(roi_matrix,copy=True)
    if type(roi_matrix) != pandas.core.frame.DataFrame:
        if type(roi_matrix) == np.ndarray:
            roi_matrix = np.array(roi_matrix,copy=True)
            roi_matrix = pandas.DataFrame(roi_matrix)
        else:
            raise IOError('roi_matrix type not recognized. Pass pandas DataFrame or np.ndarray')
    
    if mixed_probability:
        holdout_mtx = pandas.DataFrame(roi_matrix, copy=True)
    
    if type(norm_matrix) != type(None):
        if type(norm_matrix) == pandas.core.frame.DataFrame:
            norm_matrix = pandas.DataFrame(norm_matrix,copy=True)
        if type(norm_matrix) != pandas.core.frame.DataFrame:
            if type(norm_matrix) == np.ndarray:
                norm_matrix = np.array(norm_matrix,copy=True)
                norm_matrix = pandas.DataFrame(norm_matrix)
            else:
                raise IOError('roi_matrix type not recognized. Pass pandas DataFrame or np.ndarray')
        if norm_matrix.shape[-1] != roi_matrix.shape[-1]:
            raise IOError('norm_matrix must have the same number of columns as roi_matrix')
        elif all(norm_matrix.columns != roi_matrix.columns):
            raise IOError('norm_matrix must have the same column labels as roi_matrix')
    else:
        norm_matrix = pandas.DataFrame(roi_matrix, copy=True)
    
    results = pandas.DataFrame(index = roi_matrix.index, columns = roi_matrix.columns)
    if type(models) == type(None):
        for col in roi_matrix.columns:
            if not all([x==0 for x in roi_matrix[col]]):
                results.loc[:,col] = ecdf_tfm(roi_matrix[col], norm_matrix[col])
                if target_distribution == 'left':
                    results.loc[:,col] = (1 - results.loc[:,col].values)
                final_report = None
            else:
                results.loc[:,col] = [0 for x in range(len(roi_matrix[col]))]
        
    elif type(models) == dict:
        for label, model in models.items():
            if not hasattr(model, 'predict_proba'):
                raise AttributeError('Passed model %s requires the predict_proba attribute'%label)
            if not hasattr(model, 'n_components'):
                raise AttributeError('Passed model %s requires the n_components attribute'%label)
            elif model.n_components > 2:
                raise ValueError('Models with > 2 components currently not supported (%s, n=%s)'%(label,
                                                                                                 model.n_components))
        final_report = pandas.DataFrame(index = roi_matrix.columns,
                                       columns = ['model','n_components','reversed',
                                                 'perc. positive','problem'])
        for col in roi_matrix.columns:
            if not all([x==0 for x in roi_matrix[col]]):
                tfm, report_out = model_tfm(roi_matrix[col], norm_matrix[col], models, 
                                            target_distribution, fail_behavior)
                results.loc[:,col] = tfm
                final_report.loc[col,:] = pandas.DataFrame.from_dict(report_out,'index'
                                                                ).T[final_report.columns].values
                fails = len(final_report[final_report.problem!='False']['problem'].dropna())
            else:
                results.loc[:,col] = [0 for x in range(len(roi_matrix[col]))]
                final_report.loc[col,:] = [np.nan for x in range(len(final_report.columns))]
        if fails > 0:
            print('%s ROIs showed unexpected fitting behavior. See report...'%fails)
    else:
        raise ValueError('models must be a dict object or must be set to "ecdf". You passed a %s'%(type(models)))
    
    if mixed_probability:
        results = mixed_probability_transform(results, holdout_mtx, mp_thresh, final_report)
    
    if type(final_report) == type(None):
        if outdir:
            results.to_csv(os.path.join(outdir, 'results.csv'))
        return results
    else:
        if outdir:
            results.to_csv(os.path.join(outdir, 'results.csv'))
            final_report.to_csv(os.path.join(outdir, 'model_choice_report.csv'))
        return results, final_report
    
def ecdf_tfm(target_col, norm_col):
    return ed.ECDF(norm_col.values)(target_col.values)

def model_tfm(target_col, norm_col, models, target_distribution, fail_behavior):
    
    report = {}
    if len(models.keys()) > 1:
        model, label = compare_models(models,norm_col)
    else:
        model = models[list(models.keys())[0]]
        label = list(models.keys())[0]
    report.update({'model': label})
    report.update({'n_components': model.n_components})
    
    if model.n_components == 1:
        tfm = ecdf_tfm(target_col, norm_col)
        report.update({'reversed': 'False'})
        report.update({'perc. positive': np.nan})
        report.update({'problem': 'False'})
        
    else:
        fitted = model.fit(norm_col.values.reshape(-1,1))
        labs = fitted.predict(target_col.values.reshape(-1,1))
        d0_mean = target_col.values[labs==0].mean()
        d1_mean = target_col.values[labs==1].mean()
        numb = len([x for x in labs if x == 1])/len(target_col)
        if target_distribution == 'right':
            if d0_mean > d1_mean and numb > 0.5:
                report.update({'reversed': 'True'})
                report.update({'perc. positive': 1-numb})
                report.update({'problem': 'False'})
                tfm = fitted.predict_proba(target_col.values.reshape(-1,1))[:,0]
            elif d0_mean < d1_mean and numb < 0.5:
                report.update({'reversed': 'False'})
                report.update({'perc. positive': numb})
                report.update({'problem': 'False'})
                tfm = fitted.predict_proba(target_col.values.reshape(-1,1))[:,1]
            else:
                report.update({'reversed': np.nan})
                report.update({'perc. positive': np.nan})
                report.update({'problem': 'mean of 0s = %s, mean of 1s = %s, perc of 1s = %s'%(
                                                                       d0_mean, d1_mean, numb)})
                if fail_behavior == 'nan':
                    tfm = [np.nan for x in range(len(target_col))]
                elif fail_behavior == 'values':
                    tfm = fitted.predict_proba(target_col.values.reshape(-1,1))[:,1]
                
        else:
            if d0_mean < d1_mean and numb < 0.5:
                report.update({'reversed': 'False'})
                report.update({'perc. positive': numb})
                report.update({'problem': 'False'})
                tfm = fitted.predict_proba(target_col.values.reshape(-1,1))[:,0]
            elif d0_mean > d1_mean and numb > 0.5:
                report.update({'reversed': 'True'})
                report.update({'perc. positive': 1-numb})
                report.update({'problem': 'False'})
                tfm = fitted.predict_proba(target_col.values.reshape(-1,1))[:,1]
            else:
                report.update({'problem': 'mean of 0s = %s, mean of 1s = %s, perc of 1s = %s'%(
                                                                       d0_mean, d1_mean, numb)})
                if fail_behavior == 'nan':
                    tfm = [np.nan for x in range(len(target_col))]
                elif fail_behavior == 'values':
                    tfm = fitted.predict_proba(target_col.values.reshape(-1,1))[:,0] 
                
                
    return tfm, report

def compare_models(models, norm_col):
    modz = []
    labs = []
    for lab, mod in models.items():
        modz.append(mod)
        labs.append(lab)
    
    bix = []
    for model in modz:
        bic = model.fit(norm_col.values.reshape(-1,1)).bic(norm_col.values.reshape(-1,1))
        bix.append(bic)
    winner_id = np.argmin(bix)
    winning_mod = modz[winner_id]
    winning_label = labs[winner_id]
    
    return winning_mod, winning_label

def mixed_probability_transform(p_matrix, original_matrix, mp_thresh, report):
    for col in original_matrix.columns:
        if report.loc[col,'n_components'] == 2:
            newcol = pandas.Series(
                    [0 if p_matrix.loc[x, col] < mp_thresh else original_matrix.loc[x,col] for x in original_matrix.index]
                                  )
            if len(newcol[newcol>0]) > 0:
                newcol[newcol>0] = ecdf_tfm(newcol[newcol>0], newcol[newcol>0])

            p_matrix.loc[:,col] = newcol
    
    return p_matrix
        

def Evaluate_Model(roi, models, bins=None):
    '''
    Given an array of values and a dictionary of models, this script
    will generate a plot of the fitted distribution(s) from each 
    model (seperately) over the supplied data.
    
    roi -- an array, series or list values
    models -- a dict object of string label: (unfitted) sklearn.gaussian 
        model pairs
    bins -- Number of bins for the histogram.
        Passing None (default) sets bin to length(roi) / 2
    '''
    
    if type(roi) == np.ndarray or type(roi) == list:
        roi = pandas.Series(roi)
    plt.close()
    if not bins:
        bins = int(len(roi)/2)
    
    for label,model in models.items():
        mmod = model.fit(roi.values.reshape(-1,1))
        if mmod.n_components == 2:
            m1, m2 = mmod.means_
            w1, w2 = mmod.weights_
            c1, c2 = mmod.covariances_
            histdist = plt.hist(roi, bins, normed=True)
            plotgauss1 = lambda x: plt.plot(x,w1*mlab.normpdf(x,m1,np.sqrt(c1))[0], linewidth=3)
            plotgauss2 = lambda x: plt.plot(x,w2*mlab.normpdf(x,m2,np.sqrt(c2))[0], linewidth=3)
            plotgauss1(histdist[1])
            plotgauss2(histdist[1])
        elif mmod.n_components == 1:
            m1 = mmod.means_
            w1 = mmod.weights_
            c1 = mmod.covariances_
            histdist = plt.hist(roi, bins, normed=True)
            plotgauss1 = lambda x: plt.plot(x,w1*mlab.normpdf(x,m1,np.sqrt(c1))[0][0], linewidth=3)
            plotgauss1(histdist[1])
        plt.title(label)
        plt.show()

def Plot_Probabilites(prob_matrix, col_order = [], ind_order = [], 
					  vmin=None, vmax=None, figsize=()):
    '''
    Given the output matrix of Convert_ROI_values_to_Probabilities, will plot
    a heatmap of all probability values sorted in such a manner to demonstrate
    a progression of values.
    '''
    ## NOTE TO SELF: ADD ARGUMENT FOR FIGSIZE AND THRESHOLDING HEATMAP
    ## ALSO ARGUMENT TO SORT BY DIFFERENT COLUMNS OR ROWS

    if type(prob_matrix) == np.ndarray:
        prob_matrix = pandas.DataFrame(prob_matrix)
    
    if len(figsize) == 0:
    	figsize = (14,6)
    elif len(figsize) > 2:
    	raise IOError('figsize must be a tuple with two values (x and y)')

    good_cols = [x for x in prob_matrix.columns if not all([x==0 for x in prob_matrix[x]])] 
    prob_matrix = prob_matrix[good_cols]
    
    plt.close()
    if len(ind_order) == 0:
    	sorter = pandas.DataFrame(prob_matrix,copy=True)
    	sorter.loc[:,'mean'] = prob_matrix.mean(axis=1)
    	ind_order = sorter.sort_values('mean',axis=0,ascending=True).index
    if len(col_order) == 0:
    	sorter2 = pandas.DataFrame(prob_matrix,copy=True)
    	sorter2.loc['mean'] = prob_matrix.mean(axis=0)
    	col_order = sorter2.sort_values('mean',axis=1,ascending=False).columns
    fig, ax = plt.subplots(figsize=figsize) 
    forplot = prob_matrix.loc[ind_order, col_order]
    sns.heatmap(forplot, vmin, vmax)
    plt.xlabel('Regions (highest - lowest p)')
    plt.ylabel('Subjects (lowest - highest p)')
    plt.show()
    
    return forplot.columns



def Evaluate_Probabilities(prob_matrix, to_test, alpha_threshold = 0.05, FDR=None, info='medium'):
    '''
    This script will quickly calculate significant (as defined by user)
    associations between all columns in a DataFrame or matrix and variables
    passed by the user. The script will try to guess the appropriate test to
    run. Depending on inputs, the script will display the number of 
    significant columns, which columns are significant and the alpha values;
    for each passed variable.
    Multiple comparisons correction is supported.
    
    prob_matrix -- a Subject x ROI matrix or DataFrame
    
    to_test -- a dict object of where values are columns, arrays or lists with
    the same length as prob_matrix, and keys are string labels IDing them.
    
    alpha_threshold -- determines what is significant. NOTE: If an argument is
    passed for FDR, alpha_threshold refers to Q, otherwise, it refers to p.
    
    FDR -- If no argument is passed (default), no multiple comparisons
    correction is performed. If the user desires multiple comparisons correction,
    the user can select the type by entering any of the string arguments described
    here: http://www.statsmodels.org/0.8.0/generated/statsmodels.sandbox.stats.multicomp.multipletests.html
    
    info -- Determines how much information the script will display upon
    completion. 
        light: script will only display the number of significant regions
        medium: script will also display which regions were significnat
        heavy: script will also display the alpha value for each region
    '''
        

    if info not in ['light','medium','heavy']:
        print('WARNING: a value of %s was passed for argument "info"'%(info))
        print('Script will proceed with minimal information displayed')
        print('in the future, please pass one of the following:')
        print('"light", "medium", "heavy"')
        info = 'light'
    if type(prob_matrix) == np.ndarray:
        prob_matrix = pandas.DataFrame(prob_matrix)
    good_cols = [x for x in prob_matrix.columns if not all([x==0 for x in prob_matrix[x]])] 
    prob_matrix = prob_matrix[good_cols]
    
    for label, var in to_test.items():
        if type(var) == np.ndarray or type(var) == list:
            var = pandas.Series(var)
        ps = []
        n_vals = len(np.unique(var))
        if n_vals < 7:
            vals = np.unique(var)
            if n_vals == 2:
                print('for %s, using t-test...'%(label))
                for col in prob_matrix.columns:
                    p = stats.ttest_ind(prob_matrix.loc[var==vals[0]][col],
                                        prob_matrix.loc[var==vals[1]][col])[-1]
                    ps.append(p)
            elif n_vals == 3:
                print('for %s, using ANOVA...'%(label))
                for col in prob_matrix.columns:
                    p = stats.f_oneway(prob_matrix.loc[var==vals[0]][col],
                                        prob_matrix.loc[var==vals[1]][col],
                                      prob_matrix.loc[var==vals[2]][col])[-1]
                    ps.append(p)
            elif n_vals == 4:
                print('for %s, using ANOVA...'%(label))
                for col in prob_matrix.columns:
                    p = stats.f_oneway(prob_matrix.loc[var==vals[0]][col],
                                        prob_matrix.loc[var==vals[1]][col],
                                      prob_matrix.loc[var==vals[2]][col],
                                      prob_matrix.loc[var==vals[3]][col])[-1]
                    ps.append(p)
            elif n_vals == 5:
                print('for %s, using ANOVA...'%(label))
                for col in prob_matrix.columns:
                    p = stats.f_oneway(prob_matrix.loc[var==vals[0]][col],
                                        prob_matrix.loc[var==vals[1]][col],
                                      prob_matrix.loc[var==vals[2]][col],
                                      prob_matrix.loc[var==vals[3]][col],
                                      prob_matrix.loc[var==vals[4]][col])[-1]
                    ps.append(p)
            elif n_vals == 6:
                print('for %s, using ANOVA...'%(label))
                for col in prob_matrix.columns:
                    p = stats.f_oneway(prob_matrix.loc[var==vals[0]][col],
                                        prob_matrix.loc[var==vals[1]][col],
                                      prob_matrix.loc[var==vals[2]][col],
                                      prob_matrix.loc[var==vals[3]][col],
                                      prob_matrix.loc[var==vals[4]][col],
                                       prob_matrix.loc[var==vals[4]][col])[-1]
                    ps.append(p)
        else:
            print('for %s, using correlation...'%(label))
            for col in prob_matrix.columns:
                p = stats.pearsonr(prob_matrix[col],var)[-1]
                ps.append(p)
        if not FDR:
            hits = [i for i in range(len(ps)) if ps[i] < alpha_threshold]
        else:
            correction = multipletests(ps,alpha_threshold,FDR)
            hits = [i for i in range(len(ps)) if correction[0][i]]
        
        print('=============%s============'%label)
        print('for %s, %s regions were significant'%(label,len(hits)))
        if info == 'medium':
            print(prob_matrix.columns[hits])
        if info == 'heavy':
            if not FDR:
                print([(prob_matrix.columns[i], ps[i]) for i in hits])
            else:
                print([(prob_matrix.columns[i], correction[1][i]) for i in hits])
        print('\n\n')
                  
    return ps

def Prepare_Inputs_for_ESM(prob_matrices, ages, output_dir, file_name, 
                           conn_matrices = [], conn_mat_names = [], 
                           conn_out_names = [], figure = True):
    '''
    This script will convert data into a matfile compatible with 
    running the ESM, and will print outputs to be entered into
    ESM launcher script. The script will also adjust connectomes
    to accomodate missing (masked) ROIs.
    
    prob_matrices -- a dict object matching string labels to 
    probability matrices (pandas DataFrames). These will be 
    converted into a matlab structure. Columns with all 0s will be 
    removed automatically.
        NOTE: All prob_matrices should the same shape, and a 
        matching number of non-zero columns. If they do not, run the
        script separately for these matrices.
    
    ages -- an array the same length of prob_matrices that contains
    the age of each subject.
    
    output_dir -- an existing directory where all outputs will be
    written to
    
    file_name -- the name of the output matfile. Do not include a
    file extension
    
    conn_matrices -- a list of paths to matfiles or csvs containing
    connectomes that match the atlas used to intially extract data. 
    if your probability matrix does not have columns with 0s
    (because, for example, you used a mask), this argument can be
    left unchanged. Otherwise, the script will chop up the
    connectomes so they match the dimensions of the non-zero columns
    in the probability matrices.
        NOTE: passing this argument requires passing an argument for
        conn_out_names
        
    con_mat_names -- a list the same length of conn_matrices that 
    contains string labels
    
    '''

    if type(prob_matrices) != dict:
        raise IOError('prob_matrices must be a dict object')
    
    col_lens = []
    for lab, df in prob_matrices.items():
        good_cols = [y for y in df.columns if not all([x==0 for x in df[y]])]
        col_lens.append(len(good_cols))
        prob_matrices.update({lab: df[good_cols].values.T})
    if not all([x == col_lens[0] for x in col_lens]):
        raise IOError('all probability matrices entered must have the same # of non-zero columns')
    
    goodcols = [y for y in range(len(df.columns)) if not all([x==0 for x in df[df.columns[y]]])]
    
    if len(conn_matrices) > 0:
        if not len(conn_matrices) == len(conn_out_names):
            raise ValueError('equal length lists must be passed for conn_matrices and out_names')
        for i,mtx in enumerate(conn_matrices):
            if mtx[-3:] == 'csv':
                connmat = pandas.read_csv(mtx)
                x,y = connmat.shape
                if x < y:
                	connmat = pandas.read_csv(mtx,header=None)
                if all(connmat.loc[:,connmat.columns[0]] == range(connmat.shape[0])):
                	connmat = pandas.read_csv(mtx, index_col=0).values
                	x,y = connmat.shape
                	if x < y:
                		connmat = pandas.read_csv(mtx, index_col=0, header=None).values
                else:
                	connmat = connmat.values
                jnk = {}
            elif mtx[-3:] == 'mat':
                jnk = loadmat(mtx)
                connmat = jnk[conn_mat_names[i]]
            newmat = np.array([thing[goodcols] for thing in connmat[goodcols]])
            jnk[file_name] = newmat
            savemat(os.path.join(output_dir,conn_out_names[i]), jnk)
            print('new connecitity matrix size: for %s'%conn_out_names[i],newmat.shape)
            if figure:
                plt.close()
                try:
                	sns.heatmap(newmat)
                	plt.show()
                except:
                	sns.heatmap(newmat.astype(float))
                	plt.show()

    if type(ages) == np.ndarray or type(ages) == list:
        ages = pandas.Series(ages)
    if len(ages.dropna()) != len(df):
        raise ValueError('length mismatch between "ages" and prob_matrices. Does "ages" have NaNs?')
    prob_matrices.update({'ages': ages.values})
    fl_out = os.path.join(output_dir,file_name)
    savemat(fl_out,prob_matrices)
    print('ESM input written to',fl_out)
    print('===inputs:===')
    for x in prob_matrices.keys():
        print(x)
    if len(conn_matrices) > 0:
        print('===connectivity matrices===')
        for i in range(len(conn_matrices)):
            print(os.path.join(output_dir,conn_out_names[i]), conn_out_names[i])

def Evaluate_ESM_Results(results, sids, save=True, 
                         labels = None, lit = False, plot = True):
    
    '''
    This script will load the matfile outputted from the ESM, will 
    display the main model results (r2, RMSE and "eval"), the
    chosen epicenter(s) and will return the model outputs as a 
    pandas DataFrame if desired.
    
    results -- a .mat file created using the ESM script
    
    sids -- a list of subject IDs that matches the subjects input to
    the ESM
    
    save -- if True, will return a pandas DataFrame with model 
    results
    
    labels -- ROI labels that match those from the ESM input matrix.
    
    lit -- If only one epicenter was sent (for example, for 
    hypothesis testing), set this to True. Otherwise, leave as False.
    
    plot -- If True, function will plot several charts to evaluate
    ESM results on an ROI and subject level.
    '''

    mat = loadmat(results)
    
    if not lit:
        res = pandas.DataFrame(index = sids)
        for i in range(len(mat['ref_pattern'][0])):
            # Model fits
            sid = sids[i]
            r,p = stats.pearsonr(mat['ref_pattern'][:,i], mat['Final_solutions'][:,i])
            res.loc[sid,'model_r'] = r
            res.loc[sid,'model_r2'] = r**2
        res.loc[:, 'model_RMSE'] = mat['Final_RMSEs'].flatten()
        res.loc[:, 'model_eval'] = mat['Final_CORRs'].flatten()

        if save:
            # params
            res.loc[:, 'beta'] = mat['Final_parameters'][0,:].flatten()
            res.loc[:, 'delta'] = mat['Final_parameters'][1,:].flatten()
            res.loc[:, 'sigma'] = mat['Final_parameters'][2,:].flatten()

            # other
            res.loc[:, 'ref_age'] = mat['AGEs'].flatten()
            res.loc[:, 'times'] = mat['Final_times'].flatten()
            res.loc[:, 'Onset_age'] = mat['ONSETS_est'].flatten()

        print('average r2 = ', res.model_r2.mean())
        print('average RMSE =', res.model_RMSE.mean())
        print('average eval =', res.model_eval.mean())

        if type(labels) != type(None):
        	if type(labels) == np.ndarray or type(labels) == list:
        		labels = pandas.Series(labels)
        	print('model identfied the following epicenters')
        	for l in mat['models'][0,0][0][0]:
        		print(labels.loc[labels.index[l-1]])

        if plot:
            plot_out = Plot_ESM_results(mat, labels, sids, lit)

        if save:
        	if plot:
        		res = {'model_output': res, 'eval_output': plot_out}
        	return res
        
    else:
        res = pandas.DataFrame(index = sids)
        for i in range(len(mat['ref_pattern'][0])):
            # Model fits
            sid = sids[i]
            r,p = stats.pearsonr(mat['ref_pattern'][:,i], mat['model_solutions0'][:,i])
            res.loc[sid,'model_r'] = r
            res.loc[sid,'model_r2'] = r**2
        res.loc[:, 'model_RMSE'] = mat['model_RMSEs0'].flatten()
        res.loc[:, 'model_eval'] = mat['model_CORRs0'].flatten()

        if save:
            # params
            res.loc[:, 'beta'] = mat['model_parameters0'][0,:].flatten()
            res.loc[:, 'delta'] = mat['model_parameters0'][1,:].flatten()
            res.loc[:, 'sigma'] = mat['model_parameters0'][2,:].flatten()

            # other
            res.loc[:, 'ref_age'] = mat['AGEs'].flatten()
            res.loc[:, 'times'] = mat['model_times0'].flatten()
            #res.loc[:, 'Onset_age'] = mat['ONSETS_est'].flatten()

        print('average r2 = ', res.model_r2.mean())
        print('average RMSE =', res.model_RMSE.mean())
        print('average eval =', res.model_eval.mean())

        #if type(labels) != type(None):
        #    print('model identfied the following epicenters')
        #    for l in mat['models'][0,0][0][0]:
        #        print(labels.iloc[l-1]['label'])

        if plot:
            plot_out = Plot_ESM_results(mat, labels, sids, lit)
        
        if save:
            if plot:
                res = {'model_output': res, 'eval_output': plot_out}
            return res

def Plot_ESM_results(mat, labels, subids, lit):
    
    if not lit:
        mat.update({'model_solutions0': mat['Final_solutions']})
    sheets = {}
    # regional accuracy across subjects
    plt.close()
    sns.regplot(mat['ref_pattern'].mean(1), mat['model_solutions0'].mean(1))
    plt.xlabel('Avg ROI tau Probability Across Subjects')
    plt.ylabel('Avg Predicted ROI tau Probability Across Subjects')
    plt.title('Regional accuracy across subjects')
    plt.show()
    r,p = stats.pearsonr(mat['ref_pattern'].mean(1), mat['model_solutions0'].mean(1))
    print('r2 = ',r**2,'/n')
    fp = pandas.DataFrame(pandas.concat([pandas.Series(mat['ref_pattern'].mean(1)), 
                                     pandas.Series(mat['model_solutions0'].mean(1))
                                    ], axis = 1))
    fp.columns = ['reference','predicted']
    if type(labels) != type(None):
        fp.loc[:,'labels'] = labels
    sheets.update({'regional accuracy': fp})
    
    
    # Average ROI values across subject
    r2s = []
    for i in range(mat['ref_pattern'].shape[0]):
        r = stats.pearsonr(mat['ref_pattern'][i,:],mat['model_solutions0'][i,:])[0]
        r2s.append(r**2)
    if type(labels) == type(None):
        labels = range(mat['ref_pattern'].shape[0])
    roi_test = pandas.concat([pandas.Series(labels).astype(str),pandas.Series(r2s)],
                             axis=1)
    roi_test.columns = ['label','r2']
    
    plt.close()
    g = sns.factorplot(x='label', y='r2',data=roi_test, ci=None, 
                       order = roi_test.sort_values('r2',ascending=False)['label'])
    g.set_xticklabels(rotation=90)
    g.fig.set_size_inches((14,6))
    plt.title('ROI values across subjects')
    plt.show()
    print(roi_test.r2.mean(),'\n')
    sheets.update({'ROI_acc': roi_test})
    
    # average subjects across ROIs
    
    r2s = []
    for i in range(mat['ref_pattern'].shape[-1]):
        r2s.append(stats.pearsonr(mat['ref_pattern'][:,i], mat['model_solutions0'][:,i]
                                 )[0]**2)
    sub_test = pandas.concat([pandas.Series(subids).astype(str), pandas.Series(r2s)],
                            axis=1)
    sub_test.columns = ['subid','model_r2']
    
    plt.close()
    #sns.set_context('notebook')
    g = sns.factorplot(x='subid', y='model_r2', data=sub_test, ci=None,
                      order = sub_test.sort_values('model_r2',ascending=False)['subid'])
    g.set_xticklabels(rotation=90)
    g.fig.set_size_inches((14,6))
    plt.show()
    print(sub_test.model_r2.mean())
    
    return sheets

def Plot_Individual(matrix, index, style='ROI', label = None):
    '''
    Plot a single ROI across subjects, or a single subject across
    ROIs.
    
    matrix -- a dict object representing ESM results
    index -- the index of the ROI or subject to plot
    style -- set to 'ROI' or 'subject'
    label -- Title to put over the plot
    '''
    
    if style not in ['ROI', 'subject']:
        raise IOError('style argument must be set to "ROI" or "subject"')
    
    if 'model_solutions' not in matrix.keys():
        matrix.update({'model_solutions': matrix['model_solutions0']})
    
    if style == 'ROI':
        x = matrix['ref_pattern'][index,:]
        y = matrix['model_solutions'][index,:]
    else: # subject
        x = matrix['ref_pattern'][:,index]
        y = matrix['model_solutions'][:,index]
    
    plt.close()
    sns.regplot(x,y)
    plt.xlabel('Observed')
    plt.ylabel('Predicted')
    if label:
        plt.title(label)
    plt.show()

def Prepare_PET_Data(files_in, atlas, ref = None, msk = None, dimension_reduction = False,
                     ECDF_in = None, output_type = 'py', out_dir = './', out_name = 'PET_data', 
                     save_matrix = False, save_ECDF = False, save_images = False, ref_index = [],
                    mx_model = 0, orig_atlas = None):
    ''' This is a function that will take several PET images and an atlas and will
    return a subject X region matrix. If specified, the function will also calculate 
    probabilities (via ECDF) either voxelwise, or using a specified reference region
    
    files_in = input can either be 
        - a path to a directory full of (only) nifti images OR
        - a "search string" using wildcards
        - a list of subject paths OR
        - a subject X image matrix
        
    altas = a path to a labeled regional atlas in the same space as the PET data
    
    ref = multiple options:
        - If None, no probabilities will be calculated, and script will simply extract
        regional PET data using the atlas.
        - If a path to a reference region mask, will calculate voxelwise probabilities
        based on values within the reference region. Mask must be in the same space as 
        as PET data and atlas
        - If a list of integers, will combine these atlas labels with these integers to 
        make reference region 
        - if 'voxelwise', voxelwise (or atom-wise from dimension reduction) probabilities
        will be estimated. In other words, each voxel or atom will use serve as its own
        reference.
        
    msk = A path to a binary mask file in the same space as PET data and atlas. If None,
        mask will be computed as a binary mask of the atlas.
        ** PLEASE NOTE: The mask will be used to mask the reference region! **
    
    dimension_reduction = whether or not to first reduce dimensions of data using
    hierarchical clustering. This results in an initial step that will be very slow, but 
    will may result in an overall speedup for the script, but perhaps only if ref is set 
    to 'voxelwise'.
        - If None, do not perform dimension reduction
        - If integer, the number of atoms (clusters) to reduce to
    
    ECDF_in = If the user wishes to apply an existing ECDF to the PET data instead of
        generating one de novo, that can be done here. This crucial if the user wishes to
        use multiple datasets. Think of it like scaling in machine learning.
        - If None, will generate ECDF de novo.
        - If np.array, will use this array to generate the ECDF.
        - If statsmodel ECDF object, will use this as ECDF
        - If a path, will use the
    
    output_type = type of file to save final subject x region matrix into. multiple options:
        -- 'py' will save matrix into a csv
        -- 'mat' will save matrix into a matfile
    
    out_dir = location to save output files. Defaults to current directory
    
    out_name = the prefix for all output files
    
    save_matrix = Whether to save or return subject x image matrix. Useful if running multiple 
        times, as this matrix can be set as files_in, bypassing the costly data import
        -- if 'return', will return subject x image matrix to python environment
        -- if 'save', will write subject x image matrix to file. 
        -- if None, matrix will not be stored
    
    save_ECDF = whether to save the ECDF used to create the probabilities. This is crucial if 
        using multiple datasets. The resulting output can be used as input for the ECDF argument.
        -- if 'return, will return np.array to python environment
        -- if 'save', will write array to file
        -- if None, array will not be stored
    
    '''
    # Check input arguments
    print('initiating...')
    if output_type != 'py' and output_type != 'mat':
        raise IOError('output_type must be set to py or mat')
    
    
    # Initialize variables
    
    # Load data
    print('loading data...')
    i4d = load_data_old(files_in) # load PET data
    if save_matrix == 'save':
        otpt = os.path.join(out_dir,'%s_4d_data'%out_name)
        print('saving 4d subject x scan to nifti image: \n',otpt)
        i4d.to_filename(otpt)
    
    # load atlas
    atlas = ni.load(atlas).get_data().astype(int)
    if orig_atlas == True:
        orig_atlas = np.array(atlas, copy=True)
    if atlas.shape != i4d.shape[:3]:
        raise ValueError('atlas dimensions do not match PET data dimensions')
    
    # load reference region
    if type(ref) == str and ref != 'voxelwise': 
        print('looking for reference image...')
        if not os.path.isdir(ref):
            raise IOError('Please enter a valid path for ref, or select a different option for this argument')
        else:
            ref_msk = ni.load(ref).get_data()
            if ref_msk.shape != i4d.shape[:3]:
                raise ValueError('ref region image dimensions do not match PET data dimensions')
    elif type(ref) == list:
        ref_msk = np.zeros_like(atlas)
        for i in ref:
            ref_msk[atlas == i] = 1
    else:
        ref_msk = None
    
    
    # Mask data
    print('masking data...')
    if msk == None:
        img_mask = np.array(atlas,copy=True)
        img_mask[img_mask<1] = 0
        img_mask[img_mask>0] = 1
    else:
        img_mask = ni.load(msk).get_data()
        atlas[img_mask < 1] = 0
    
    if type(ref_msk) != type(None):
        ref_msk[img_mask < 1] = 0
    
    mask_tfm = input_data.NiftiMasker(ni.Nifti1Image(img_mask,i4d.affine))
    mi4d = mask_tfm.fit_transform(i4d)
    
    # dimension reduction (IN BETA!)
    if dimension_reduction:
        print('reducing dimensions...')
        shape = img_mask.shape
        connectivity = grid_to_graph(n_x=shape[0], n_y=shape[1],
                                   n_z=shape[2], mask=img_mask)
    # main ECDF calculation (or mixture model calc)
    skip = False
    if ref != 'voxelwise':
        if type(ECDF_in) != type(None): 
            print('generating ECDF...')
            print('using user-supplied data...')
            if type(ECDF_in) == ed.ECDF:
                mi4d_ecdf, ecref = ecdf_simple(mi4d, ECDF_in, mx=mx_model)
                input_distribution = 'not generated'
            elif type(ECDF_in) == np.ndarray:
                mi4d_ecdf, ecref = ecdf_simple(mi4d, ECDF_in, mx=mx_model)
                input_distribution = ECDF_in
    #       elif # add later an option for importing an external object 
            else:
                try:
                    mi4d_ecdf, ecref = ecdf_simple(mi4d, ECDF_in, mx=mx_model)
                    print('Could not understand ECDF input, but ECDF successful')
                    input_distribution = 'not generated'
                except:
                    raise IOError(
                            'Invalid argument for ECDF in. Please enter an ndarray, an ECDF object, or a valid path')
        else:
            if type(ref_msk) != type(None):
                print('generating ECDF...')
                ref_tfm = input_data.NiftiMasker(ni.Nifti1Image(ref_msk,i4d.affine))
                refz = ref_tfm.fit_transform(i4d)
                mi4d_ecdf, ecref = ecdf_simple(mi4d, refz, mx=mx_model)
                input_distribution = refz.flat
            else:
                print('skipping ECDF...')
                skip = True
    
    else:
        print('generating voxelwise ECDF...')
        mi4d_ecdf, ECDF_array = ecdf_voxelwise(mi4d, ref_index, save_ECDF, mx=mx_model)
        input_distribution = 'not generated'
        
    if not skip:
#       if save_ECDF:
#           create an array and somehow write it to a file
        
    # transform back to image-space
        print('transforming back into image space')
        f_images = mask_tfm.inverse_transform(mi4d_ecdf)
    
    else:
        #if type(ECDF):
        print('transforming back into image space')
        f_images = mask_tfm.inverse_transform(mi4d)
    
    # generate output matrix
    print('generating final subject x region matrix')
    if type(orig_atlas) == type(None):
        f_mat = generate_matrix_from_atlas_old(f_images, atlas)
    else:
        f_mat = generate_matrix_from_atlas_old(f_images, orig_atlas)
    
    # compile (and save) outputs
    print('preparing outputs')
    output = {}
    if output_type == 'py':
        f_mat.to_csv(os.path.join(out_dir, '%s_roi_data.csv'%out_name),index=False)
        output.update({'roi_matrix': f_mat})
    else:
        output.update({'roi_matrix': fmat.values})
        output.update({'roi_matrix_columns': fmat.columns})
    if save_matrix == 'return':
        output.update({'4d_image_matrix': i4d})
    if save_ECDF == 'return':
        if output_type == 'py':
            output.update({'ECDF_function': ECDF_array})
        else:
            output.update({'input_distribution': input_distribution})
    
def load_data_old(files_in):
    
    fail = False
    
    if type(files_in) == str:
        if os.path.isdir(files_in):
            print('It seems you passed a directory')
            search = os.path.join(files_in,'*')
            num_f = len(glob(search))
            if num_f == 0:
                raise IOError('specified directory did not contain any files')
            else:
                print('found %s images!'%num_f)
            i4d = image.load_img(search)
        elif '*' in files_in:
            print('It seems you passed a search string')
            num_f = len(glob(files_in))
            if num_f == 0:
                raise IOError('specified search string did not result in any files')
            else:
                print('found %s images'%num_f)
            i4d = image.load_img(files_in)
        else:
            fail = True
    elif type(files_in) == list:
        print('processing %s subjects'%len(files_in))
        i4d = ni.concat_images(files_in)
    elif type(files_in) == ni.nifti1.Nifti1Image:
        print('processing %s subjects'%files_in.shape[-1])
        i4d = files_in
    else:
        fail = True
        
    if fail:
        print('files_in not recognized.', 
                    'Please enter a search string, valid directory, list of subjects, or matrix')
        raise ValueError('I do not recognize the files_in input.')
    
    return i4d

def dim_reduction(mi4d, connectivity, dimension_reduction):
    ward = FeatureAgglomeration(n_clusters=dimension_reduction/2,
            connectivity=connectivity, linkage='ward', memory='nilearn_cache')
    ward.fit(mi4d)
    ward = FeatureAgglomeration(n_clusters=dimension_reduction,
            connectivity=connectivity, linkage='ward', memory='nilearn_cache')
    ward.fit(mi4d)                                                         
    mi4d = ward.transform(mi4d)

    return mi4d

def ecdf_simple(mi4d, refz, mx=0):

    if type(refz) == ed.ECDF:
        ecref = refz
    else:
        if len(refz.shape) > 1:
            ecref = ed.ECDF(refz.flat)
        else:
            ecref = ed.ECDF(refz)
    print('transforming images...')
    if mx == 0:
        mi4d_ecdf = ecref(mi4d.flat).reshape(mi4d.shape[0],mi4d.shape[1])
    else:
        print('are you sure it makes sense to use a mixture model on reference region?')
        mod = GaussianMixture(n_components=mx).fit(ecref)
        mi4d_ecdf = mod.predict_proba(mi4d.flat)[:,-1].reshape(mi4d.shape[0],mi4d.shape[1])

    return mi4d_ecdf, ecref   

def ecdf_voxelwise(mi4d, ref_index, save_ECDF, mx=0):
    
    X,y = mi4d.shape
    if mx != 0: 
        mmod = GaussianMixture(n_components=mx)
    
    if len(ref_index) == 0:
        if not save_ECDF:
            if mx == 0:
                mi4d_ecdf = np.array([ed.ECDF(mi4d[:,x])(mi4d[:,x]) for x in range(y)]).transpose()
            else:
                mi4d_ecdf = np.array([mmod.fit(mi4d[:,x].reshape(-1,1)).predict_proba(mi4d[:,x].reshape(-1,1)
                                                                                    )[:,-1] for x in range(y)]).transpose()
            ECDF_array = None    
        else:
            if mx == 0:
                ECDF_array = np.array([ed.ECDF(mi4d[:,x]) for x in range(y)]).transpose()
                print('transforming data...')
                mi4d_ecdf = np.array([ECDF_matrix[x](mi4d[:,x]) for x in range(y)]
                                         ).transpose()
            else:
                raise IOError('at this stage, cant save mixture model info....sorry...')
    else:
        if mx == 0:
            # if you don't want to include subjects used for reference, un-hash this, hash
            # the next line, and fix the "transpose" line so that the data gets back into the matrix properly
            #good_ind = [x for x in list(range(X)) if x not in ref_index]
            good_ind = range(X)
            if not save_ECDF:    
                mi4d_ecdf = np.array([ed.ECDF(mi4d[ref_index,x])(mi4d[good_ind,x]) for x in range(y)]
                                    ).transpose()
                ECDF_array = None
            else:
                ECDF_array = [ed.ECDF(mi4d[ref_index,x]) for x in range(y)]
                print('transforming data...')
                mi4d_ecdf = ecdf_voxelwise = np.array([ECDF_matrix[x](mi4d[good_ind,x]) for x in range(y)]
                                         ).transpose()
        else:
            ### COMING SOON!
            raise IOError('have not yet set up implementation for mixture models and reg groups')
        
    return mi4d_ecdf, ECDF_array

def generate_matrix_from_atlas_old(files_in, atlas):
    
    files_in = files_in.get_data()
    atlas = atlas.astype(int)
    f_mat = pandas.DataFrame(index = range(files_in.shape[-1]),
                             columns = ['roi_%s'%x for x in np.unique(atlas) if x != 0])
    tot = np.bincount(atlas.flat)
    for sub in range(files_in.shape[-1]):
        mtx = files_in[:,:,:,sub]
        sums = np.bincount(atlas.flat, weights = mtx.flat)
        rois = (sums/tot)[1:]
        f_mat.loc[f_mat.index[sub]] = rois
        
    return f_mat


def W_Transform(roi_matrix, covariates, norm_index = [], 
                columns = [], verbose = False):
    
    '''
    Depending on inputs, this function will either regress selected 
    variables out of an roi_matrix, or will perform a W-transform on an 
    roi_matrix.
    
    W-transform is represented as such:
    
    (Pc - A) / SDrc
    
    Where Pc is the predicted value of the roi *based on the covariates 
    of the norm sample*; A = actual value of the roi; SDrc = standard 
    deviation of the residuals *or the norm sample*
    
    roi_matrix = a subjects x ROI array
    
    covariates = a subject x covariates array
    
    norm_index = index pointing exclusively to subjects to be used for
    normalization. If norm index is passed, W-transformation will be 
    performed using these subjects as the norm_sample (see equation 
    above). If no norm_index is passed, covariates will simply be
    regressed out of all ROIs.
    
    columns = the columns to use fron the covariate matrix. If none, 
    all columns if the covariate matrix will be used.
    
    verbose = If True, will notify upon the completion of each ROI
    transformation.
    '''
    
    if type(roi_matrix) != pandas.core.frame.DataFrame:
        raise IOError('roi_matrix must be a subjects x ROIs pandas DataFrame')
    if type(covariates) != pandas.core.frame.DataFrame:
        raise IOError('covariates must be a subjects x covariates pandas DataFrame')
    
    covariates = clean_time(covariates)
    roi_matrix = clean_time(roi_matrix)
    
    if len(columns) > 0:
        covs = pandas.DataFrame(covariates[columns], copy=True)
    else:
        covs = pandas.DataFrame(covariates, copy=True)
    
    if covs.shape[0] != roi_matrix.shape[0]:
        raise IOError('length of indices for roi_matrix and covariates must match')
    else:
        data = pandas.concat([roi_matrix, covs], axis=1)
    
    output = pandas.DataFrame(np.zeros_like(roi_matrix.values),
                             index = roi_matrix.index,
                             columns = roi_matrix.columns)
    
    if len(norm_index) == 0:
        for roi in roi_matrix.columns:
            eq = '%s ~'%roi
            for i,col in enumerate(covs.columns):
                if i != len(covs.columns) - 1:
                    eq += ' %s +'%col
                else:
                    eq += ' %s'%col
            mod = smf.ols(eq, data = data).fit()
            output.loc[:,roi] = mod.resid
            if verbose:
                print('finished',roi)
    else:
        for roi in roi_matrix.columns:
            eq = '%s ~'%roi
            for i,col in enumerate(covs.columns):
                if i != len(covs.columns) - 1:
                    eq += ' %s +'%col
                else:
                    eq += ' %s'%col
            mod = smf.ols(eq, data=data.loc[norm_index]).fit()
            predicted = mod.predict(data)
            w_score = (data.loc[:,roi] - predicted) / mod.resid.std()
            output.loc[:,roi] = w_score
            if verbose:
                print('finished',roi)
    
    return output

def clean_time(df):
    
    df = pandas.DataFrame(df, copy=True)
    symbols = ['.','-',' ', ':', '/','&']
    ncols = []
    for col in df.columns:
        for symbol in symbols:
            if symbol in col:
                col = col.replace(symbol,'_')
        ncols.append(col)
    
    df.columns = ncols
    
    return df
                

def Weight_Connectome(base_cx, weight_cx, method = 'min', symmetric = True,
                     transform = MinMaxScaler(), transform_when = 'post',
                     illustrative = False, return_weight_mtx = False):
    
    if method not in ['min','mean','max']:
        raise IOError('a value of "min" or "mean" must be passed for method argument')
    
    choices = ['prae','post','both','never']
    if transform_when not in choices:
        raise IOError('transform_when must be set to one of the following: %s'%choices)
    
    if len(np.array(weight_cx.shape)) == 1 or np.array(weight_cx).shape[-1] == 1:
        print('1D array passed. Transforming to 2D matrix using %s method'%method)
        weight_cx = create_connectome_from_1d(weight_cx, method, symmetric)
    
    if transform_when == 'pre' or transform_when == 'both':
        weight_cx = transform.fit_transform(weight_cx)
    
    if base_cx.shape == weight_cx.shape:
        if illustrative:
            plt.close()
            sns.heatmap(base_cx)
            plt.title('base_cx')
            plt.show()
            
            plt.close()
            sns.heatmap(weight_cx)
            plt.title('weight_cx')
            plt.show()
            
        weighted_cx = base_cx * weight_cx
        
        if illustrative:
            plt.close()
            sns.heatmap(weighted_cx)
            plt.title('final (weighted) cx')
            plt.show()
    else:
        raise ValueError('base_cx (%s) and weight_cx %s do not have the sampe shape'%(
                                                                            base_cx.shape,
                                                                            weight_cx.shape))
    
    if transform_when == 'post' or transform_when == 'both':
        transform.fit_transform(weighted_cx)
    
    if return_weight_mtx:
        return weighted_cx, weight_cx
    else:
        return weighted_cx
    
def create_connectome_from_1d(cx, method, symmetric):
    
    nans = [x for x in range(len(cx)) if not pandas.notnull(cx[x])]
    if len(nans) > 1:
        raise ValueError('Values at indices %s are NaNs. Cannot compute'%nans)
    
    weight_cx = np.zeros((len(cx),len(cx)))
    if method == 'min':
        if symmetric:
            for i,j in list(itertools.product(range(len(cx)),repeat=2)):
                weight_cx[i,j] = min([cx[i],cx[j]])
        else:
            for i,j in itertools.combinations(range(len(cx)),2):
                weight_cx[i,j] = min([cx[i],cx[j]])
                rotator = np.rot90(weight_cx, 2)
                weight_cx = weight_cx + rotator
    elif method == 'mean':
        if symmetric:
            for i,j in list(itertools.product(range(len(cx)),repeat=2)):
                weight_cx[i,j] = np.mean([cx[i],cx[j]])
        else:
            for i,j in itertools.combinations(range(len(cx)),2):
                weight_cx[i,j] = np.mean([cx[i],cx[j]])
                rotator = np.rot90(weight_cx, 2)
                weight_cx = weight_cx + rotator
    elif method == 'max':
        if symmetric:
            for i,j in list(itertools.product(range(len(cx)),repeat=2)):
                weight_cx[i,j] = max([cx[i],cx[j]])
        else:
            for i,j in itertools.combinations(range(len(cx)),2):
                weight_cx[i,j] = max([cx[i],cx[j]])
                rotator = np.rot90(weight_cx, 2)
                weight_cx = weight_cx + rotator
    
    return weight_cx