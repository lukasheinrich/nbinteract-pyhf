import pyhf
import numpy as np
import nbinteract as nbi
import nbinteract.plotting as nbip
import bqplot as bq
import requests



def read_wspace(wspace):
    spec = {
        'channels': wspace['channels'],
        'parameters': wspace['toplvl']['measurements'][0]['config'].get(
            'parameters', []
        ),
    }
    pdf = pyhf.Model(spec, poiname = 'SigXsecOverSM')
    data = wspace['data']['channel1']
    selector = {k: v['slice'].start for k,v in pdf.config.par_map.items()}
    init = pdf.config.suggested_init()
    pars = {k: init[v] for k,v in selector.items()}
    return pdf, data, selector, pars


def plot_lhood(pdf, obs_data, selector = None, **par_settings):
    pars = pyhf.tensorlib.astensor(pdf.config.suggested_init())
    for k,v in par_settings.items():
        pars[selector[k]] = v
    mc_counts = get_mc_counts(pdf,pars)
    return mc_counts[:,0,:]

def get_mc_counts(pdf, pars):
    deltas, factors = pdf._modifications(pars)
    allsum = pyhf.tensorlib.concatenate(deltas + [pyhf.tensorlib.astensor(pdf.thenom)])
    nom_plus_delta = pyhf.tensorlib.sum(allsum,axis=0)
    nom_plus_delta = pyhf.tensorlib.reshape(nom_plus_delta,(1,)+pyhf.tensorlib.shape(nom_plus_delta))
    allfac = pyhf.tensorlib.concatenate(factors + [nom_plus_delta])
    return pyhf.tensorlib.product(allfac,axis=0)


  

def viz_likelihood(json_data):
    pdf, data, selector, pars  = read_wspace(json_data)

    options = {
        'xlim': (-5,55),
        'ylim': (0, 1000),
        'bins': 20
    }


    def yields(xs,mc2_shape_conv,mc2_weight_var1,mc1_shape_conv,mc1_weight_var1,SigXsecOverSM,lumi):
        pars = {'mc2_shape_conv': mc2_shape_conv,
                'mc2_weight_var1': mc2_weight_var1,
                'mc1_shape_conv': mc1_shape_conv,
                'mc1_weight_var1': mc1_weight_var1,
                'SigXsecOverSM': SigXsecOverSM,
                'lumi': lumi}
        return plot_lhood(pdf,data,selector,**pars)



    ranges = {k: tuple(pdf.config.par_map[k]['paramset'].suggested_bounds[0]) for k in pars.keys()}


    counts = plot_lhood(pdf,data,selector)

    fig = nbi.Figure(options = options)
    fig.bar(np.arange(len(counts[0])), yields, **ranges)
    fig.scatter(np.arange(len(counts[0])), data)
    return fig

