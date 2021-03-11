import pandas
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_components(clf, x, y, n_iterations=500, check = 100,
                       evaluate = True, plot = True, thr = 0.95,
                       metric=None, random_state=123):
    
    if type(x) != type(pandas.DataFrame()):
        x = pandas.DataFrame(x)
    
    # fit model
    clf.fit(x,y)
    n_comps = clf.n_components
    
    # prepare output
    results = pandas.DataFrame(index = range(n_comps * (n_iterations+1)),
                              columns = ['score', 'component', 'model'])
    results.loc[:,'component'] = list(range(n_comps))*(n_iterations+1)
    results.loc[range(n_comps),'model'] = ['True']*n_comps
    results.loc[range(n_comps,n_comps*(n_iterations+1)), 'model'
               ] = ['Null']*(n_comps*n_iterations)
    if not metric:
    	true_scores = [stats.pearsonr(clf.x_scores_[:,x], clf.y_scores_[:,x]
                                     )[0]**2 for x in range(n_comps)]
    else:
        true_scores = [metric(clf.x_scores_[:,x], clf.y_scores_[:,x]
                             ) for x in range(n_comps)]
    results.loc[results[results.model=='True'].index,'score'] = true_scores
    
    k = clf.n_components
    # permute and refit model
    rs = np.random.RandomState(random_state)
    x.index = range(len(x.index))
    for i in range(n_iterations):
        new_ind = rs.permutation(x.index)
        new_x = x.iloc[new_ind]
        newmod = clf.fit(new_x,y)
        if not metric:
            new_scores = [stats.pearsonr(newmod.x_scores_[:,x], 
                                         newmod.y_scores_[:,x]
                                        )[0]**2 for x in range(n_comps)]
        else:
            new_scores = [metric(newmod.x_scores_[:,x], newmod.y_scores_[:,x]
                                ) for x in range(n_comps)]
        results.loc[range(k, k+n_comps), 'score'] = new_scores
        if check:
            if i % check == 0:
                print('finished iteration',i)
        
        k += n_comps
        
    if evaluate:
        if plot:
            cr = display_results(results, thr)
        else:
            cr = display_results(results, thr, False)
        
    return results, cr

def display_results(results, thr = 0.95, plot=True):
    
    if plot:
        # plot components
        sns.set_context('paper')
        plt.close()
        sns.catplot(x='component', y = 'score', hue='model', data=results,kind='point')
        plt.show()
    
    # get p-values
    comp_results = pandas.DataFrame(index=results.component.unique(),
                                   columns = ['r','p','sig'])
    for i in results.component.unique():
        nullz = results[(results.component==i) & (results.model=='Null')
                       ]['score'].sort_values().values
        real = results[(results.component==i) & (results.model=='True')]['score'].values[0]
        comp_results.loc[i,'r'] = real
        p = (len(nullz[nullz>real])+1) / len(nullz)
        if p < (1 - thr):
            comp_results.loc[i,['p','sig']] = [p, 1]
            print('component %s: p = %s  ***'%(i,p))
        else:
            comp_results.loc[i,['p','sig']] = [p, 0]
            print('component %s: p = %s'%(i,p))
        
    
    return comp_results


def bootstrap_features(clf, fit_model, X, y, n_iterations=500, check = 100, on ='x'):
    
    if type(X) != type(pandas.DataFrame()):
        X = pandas.DataFrame(X)
    if type(y) != type(pandas.DataFrame()):
        y = pandas.DataFrame(y)

    # fit model
    orig = fit_model

    # prepare output
    n_feats_x = X.shape[-1]
    n_feats_y = y.shape[-1]
    all_results_x = {}
    all_results_y = {}

    for i in range(orig.n_components):
        results = pandas.DataFrame(index = range(n_iterations), columns = range(n_feats_x))
        all_results_x.update({i: results})
        results = pandas.DataFrame(index = range(n_iterations), columns = range(n_feats_y))
        all_results_y.update({i: results})
    bs_ratio_x = pandas.DataFrame(index = range(orig.n_components), 
                                columns = range(n_feats_x))
    bs_ratio_y = pandas.DataFrame(index = range(orig.n_components), 
                                columns = range(n_feats_y))

    # bootstrap
    for i in range(n_iterations):
        n_ind = np.random.choice(X.index, len(X.index))
        n_samp = pandas.DataFrame(X.loc[n_ind],copy=True)
        ny = pandas.DataFrame(y.loc[n_ind],copy=True)
        newmod = clf.fit(n_samp,ny)
        for c in range(orig.n_components):
            xcorrs = [stats.pearsonr(orig.x_loadings_[:,c],
                                     newmod.x_loadings_[:,x])[0]**2 for x in range(orig.n_components)]
            closest = np.argmax(xcorrs)
            # account for possible inversion
            if stats.pearsonr(orig.x_loadings_[:,c],
                         newmod.x_loadings_[:,closest])[0] < 0:
                new_loadingsX = newmod.x_loadings_[:,closest] * -1
                new_loadingsy = newmod.y_loadings_[:,closest] * -1
            else:
                new_loadingsX = newmod.x_loadings_[:,closest]
                new_loadingsy = newmod.y_loadings_[:,closest]

            all_results_x[c].loc[i] = new_loadingsX
            all_results_y[c].loc[i] = new_loadingsy
        if check:
            if i % check == 0:
                print('finished iteration',i)

    # calculate bootstrap ratio
    bs_ratio_x = bootstrap_ratio(all_results_x, bs_ratio_x)
    bs_ratio_y = bootstrap_ratio(all_results_y, bs_ratio_y)
    
    return bs_ratio_x, bs_ratio_y, all_results_x, all_results_y

def bootstrap_ratio(all_results, bs_ratio):

    all_stats = {'mean': pandas.DataFrame(index = bs_ratio.index, columns = bs_ratio.columns),
                'se': pandas.DataFrame(index = bs_ratio.index, columns = bs_ratio.columns),
                 'bsr': pandas.DataFrame(index = bs_ratio.index, columns = bs_ratio.columns),
                'p': pandas.DataFrame(index = bs_ratio.index, columns = bs_ratio.columns)}
    
    for i,res in all_results.items():
        mn = res.mean()
        sem = res.sem()
        all_stats['bsr'].loc[i] = mn / sem
        all_stats['mean'].loc[i] = mn
        all_stats['se'].loc[i] = sem
        ps = []
        for x in range(res.shape[-1]):
            jnk = res.values[:,x]
            if mn[x] > 0:
                p = (len(jnk[jnk<0])+1) / len(jnk)
            else:
                p = (len(jnk[jnk>0])+1) / len(jnk)
            ps.append(p)
        all_stats['p'].loc[i] = ps
            
        
    return all_stats

        
