import os
import sys
import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from glob import glob
from scipy import stats
from scipy.io import loadmat, savemat
from dateutil.parser import parse
from nilearn import plotting, image
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.sandbox.stats.multicomp import multipletests
sys.path.insert(0,'/home/users/jvogel/git/data_driven_pathology/esm/')
import ESM_utils as esm

def feature_prep(data, features, outdir, outnm, samples = {}, 
                 norm_sample = {}, kind = 'EBM', regr_type = 'none', 
                 regr_cols = [], norm_index = [], models = None,
                 return_data = True, save_data = True, log_tfm = False):
    
    if type(features) != dict:
        raise IOError('argument "features" must be (name,data) dictionaries')
    
    if kind not in ['EBM','Sus']:
        raise IOError('"kind" must be set to "EBM" or "Sus"')
    
    rt_list = ['none','w','regr']
    if regr_type not in rt_list:
        raise IOError('"regr_type" must be set to one of',rt_list)
    
    if regr_type == 'regr' and len(regr_cols)==0:
        raise IOError('if "regr_type" is set to "regr", "regr_cols" must be passed')
        
    if regr_type == 'w' and len(norm_index)==0:
        raise IOError('if "regr_type" is set to "w", "norm_index" must be passed')
    
    if not samples:
        samples.update({'all': data.index})
    
    supplementary = {}
    if kind == 'EBM':
        data_y = {}
        data_n = {}
    elif kind == 'Sus':
        data_in = {}
        data_params = {}
        
    for fnm, fset in features.items():
        print('working on ',fnm)
        for snm,samp in samples.items():
            print('using {} subjects'.format(snm))
            nm = '{}_{}_regr{}'.format(fnm,snm,regr_type)
            Xdata = data.loc[samp,fset]
            if log_tfm:
            	for col in Xdata.columns:
            		Xdata.loc[:,col] = np.log(Xdata.loc[:,col].values)

            if len(norm_sample) > 0:
                Ndata = norm_sample[snm][fset]

            if regr_type == 'regr':
                Xdata = esm.W_Transform(roi_matrix = Xdata, 
                           covariates = data.loc[samp], 
                           columns = regr_cols)
                Xdata.columns = fset
                if len(norm_sample) > 0:
                    if not all([True for x in regr_cols if x in norm_sample[snm].columns]):
                        raise IOError('not all of {} found in norm_sample passed'.format(regr_cols))
                    Ndata = esm.W_Transform(roi_matrix = Ndata, 
                           covariates = norm_sample[snm], 
                           columns = regr_cols)
                    Xdata.columns = fset
                    Ndata.columns = fset
            elif regr_type == 'w':
                if len(regr_cols) > 0:
                    Xdata = esm.W_Transform(roi_matrix = Xdata,
                           covariates = data.loc[samp], 
                           columns = regr_cols,
                           norm_index = norm_index)
                    Xdata.columns = fset
                    if len(norm_sample) > 0:
                        if not all([True for x in regr_cols if x in norm_sample[samp].columns]):
                            raise IOError('not all of {} found in norm_sample passed'.format(regr_cols))
                        N_index = Ndata.index
                        Ndata = esm.W_Transform(roi_matrix = pandas.concat([Ndata,Xdata]),
                           covariates = norm_sample[snm], 
                           columns = regr_cols,
                           norm_index = norm_index)
                        Ndata.columns = fset
                        Ndata = Ndata.loc[N_index] 
                else:
                    Xdata = esm.W_Transform(roi_matrix = Xdata,
                           covariates = data.loc[samp], 
                           norm_index = norm_index)
                    Xdata.columns = fset
                    if len(norm_sample) > 0:
                        N_index = Ndata.index
                        Ndata = esm.W_Transform(roi_matrix = pandas.concat([Ndata,Xdata]), 
                           columns = regr_cols,
                           norm_index = norm_index)
                        Ndata.columns = fset
                        Ndata = Ndata.loc[N_index] 
                    

            if kind == 'EBM':
                if not models:
                    models = {'one_comp': GaussianMixture(n_components=1,random_state=123),
                              'two_comp': GaussianMixture(n_components=2,random_state=123, 
                                                          tol=0.00001, max_iter=1000)}
#                 if len(Left_index):
#                     if not all([(x[:2]=='L_') | (x[:2]=='R_') | (x[-2:]=='_L') | (x[-2:]=='_R') for x in fset]):
#                         left = [x for x in fset if x in Left_index]
#                         right = [x for x in fset if 'R_' in x or '_R' in x]
#                         Xdata.loc[:,'Asym'] = (abs(Xdata[left].values - Xdata[right].values)).mean(1)
#                     else:
#                         print('no hemispheres detected. Moving on without adding asymmetry feature')
                        
                if len(norm_sample) > 0:
                    xps, report = esm.Convert_ROI_values_to_Probabilities(Xdata,norm_matrix=Ndata,
                                                                          models=models)
                else:
                    xps, report = esm.Convert_ROI_values_to_Probabilities(Xdata,models=models)
                data_y.update({'EBM_%s'%nm: xps.values})
                if len(norm_sample) > 0:
                    xpNs,jnk = esm.Convert_ROI_values_to_Probabilities(Xdata, norm_matrix=Ndata,
                                                                       models=models,
                                                                       target_distribution='left')
                else:
                    xpNs,jnk = esm.Convert_ROI_values_to_Probabilities(Xdata,models=models,
                                                                       target_distribution='left')
                data_n.update({'EBM_%s'%nm: xpNs.values})
                supplementary.update({'EBM_%s_report'%nm: report.to_dict()})
                supplementary.update({'EBM_%s_idx'%nm: Xdata.index.tolist()})
                supplementary.update({'EBM_%s_cols'%nm: Xdata.columns.tolist()})
                if len(norm_sample) > 0:
                    supplementary.update({'EBM_%s_normdat'%nm: Ndata.values})
                    supplementary.update({'EBM_%s_normidx'%nm: Ndata.index.tolist()})
                    supplementary.update({'EBM_%s_normcols'%nm: Ndata.columns.tolist()})
            
            elif kind == 'Sus':
            	# temporary
                data_in.update({'Sus_%s'%nm: Xdata.values})
                supplementary.update({'Sus_%s_idx'%nm: Xdata.index.tolist()})
                supplementary.update({'Sus_%s_cols'%nm: Xdata.columns.tolist()})
                if len(norm_sample) > 0:
                    supplementary.update({'Sus_%s_normdat'%nm: Ndata.values})
                    supplementary.update({'Sus_%s_normidx'%nm: Ndata.index.tolist()})
                    supplementary.update({'Sus_%s_normcols'%nm: Ndata.columns.tolist()})
                # mins = []
                # maxs = []
                # FINISH THIS LATER
    
        if kind == 'EBM':
            files_out = dict(zip(['data-y','data-n','supplementary'],
                                 [data_y,data_n,supplementary]))
        elif kind == 'Sus':
            files_out = dict(zip(['data-in','params','supplementary'],
                                 [data_in,data_params,supplementary]))
    if save_data:
        for flnm, fl in files_out.items():
            new_pth = os.path.join(outdir,'{}_{}_{}'.format(outnm, nm,flnm))
            savemat(new_pth, fl)
    
    if return_data:
        return files_out

from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(true, predicted, labels,
                          cmap='Blues', figsize=(8,8), 
                          normalize=False, cbar=False,
                         save=''):
    plt.close()
    cm = confusion_matrix(true, predicted)
    fig,ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    if cbar:
        ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=labels, yticklabels=labels,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    
    if save:
        plt.savefig(save, bbox_inches='tight')
    
    plt.show()