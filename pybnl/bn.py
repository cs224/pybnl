
import locale
locale.setlocale(locale.LC_ALL, 'C')

import numpy as np, xarray as xr, pandas as pd
import networkx as nx, networkx.algorithms.dag, graphviz
import sklearn.base
import itertools, collections, tempfile


import rpy2, rpy2.rinterface, rpy2.robjects, rpy2.robjects.packages, rpy2.robjects.lib, rpy2.robjects.lib.grid, \
    rpy2.robjects.lib.ggplot2, rpy2.robjects.pandas2ri, rpy2.interactive.process_revents, \
    rpy2.interactive, rpy2.robjects.lib.grdevices
# rpy2.interactive.process_revents.start()
rpy2.robjects.pandas2ri.activate()

# import R's "base" package
base = rpy2.robjects.packages.importr('base')
# import R's utility package
utils = rpy2.robjects.packages.importr('utils')
# select a mirror for R packages
utils.chooseCRANmirror(ind=1) # select the first mirror in the list

grdevices   = rpy2.robjects.packages.importr('grDevices')
bnlearn     = rpy2.robjects.packages.importr('bnlearn')
gRain       = rpy2.robjects.packages.importr('gRain')


def generate_model_string(target_var, parent_vars):
    parent_model_string = ':'.join(parent_vars)
    parent_child_string = '|'.join([target_var, parent_model_string])
    if len(parent_vars) == 0:
        parent_child_string = target_var
    return '[' + parent_child_string + ']'


class DiscreteTabularCPM():

    def __init__(self, ldf, lcpm_xr=None, transform=True):
        self.transform = transform
        self.ldf = ldf

        self.lcpm_xr  = lcpm_xr
        if self.lcpm_xr is not None:
            self.ldf = convert_xarray_array_to_pandas_dtcpm(lcpm_xr)

        self.cpt_to_r()
        self.cpt_to_model_string()

    def extract_xarray_and_nparray_no_transform(self, xrin):
        self.lcpm_xr = xrin
        self.nparray = np.array(xrin)

    def extract_xarray_and_nparray_transform(self, xrin):
        tmp0 = xrin
        old_order = tmp0.dims
        new_order = old_order[-1:] + old_order[:-1]
        lcpm_xr = tmp0.transpose(*new_order)  #lcpm_xds[var_name]

        self.lcpm_xr = lcpm_xr

    def cpt_to_r(self):
        if self.lcpm_xr is None:
            lcpm_ds = self.ldf.groupby(list(self.ldf.columns[:-1])).max()  # .to_xarray().p.values
            lcpm_xds = lcpm_ds.to_xarray()
            var_name = list(lcpm_xds.data_vars.keys())[0]
            if self.transform:
                self.extract_xarray_and_nparray_transform(lcpm_xds[var_name])
            else:
                self.extract_xarray_and_nparray_no_transform(lcpm_xds[var_name])
        self.nparray = np.array(self.lcpm_xr)

        # the dim shape
        dim_shape = self.nparray.shape
        rdim_shape = rpy2.robjects.IntVector(dim_shape)

        # the dimnames
        dim_name_values_pair = {}
        dim_names = self.lcpm_xr.dims
        for dim_name in dim_names:
            dim = self.lcpm_xr.coords[dim_name]
            dim_values = [str(dv) for dv in dim.values]
            dim_name_values_pair.update({dim_name: rpy2.robjects.StrVector(dim_values)})

        # now create the R array
        rarray = rpy2.robjects.r['array'](
            self.nparray,
            dim=rdim_shape,
            dimnames=rpy2.robjects.r['list'](**dim_name_values_pair)
        )
        self.rarray = rarray


    def cpt_to_model_string(self):
        columns = self.ldf.columns
        self.p_var_name = columns[-1]
        target_var = columns[-2]
        self.target = target_var
        parent_vars = list(columns[:-2])
        self.parents = parent_vars

        self.model_string = generate_model_string(target_var, parent_vars)

    def to_cpm_table(self):
        tmp = self.ldf.pivot_table(index=self.parents,columns=[self.target],values=[self.p_var_name]).reset_index()
        #tmp.columns = [self.target]
        return tmp


class BayesNetworkStructure():
    def __init__(self, rnet):
        self.rnet     = rnet
        self.dag()

    def dag(self):
        # nodes, unidirectional_edges, bidirectional_edges = dag(self.rnet)
        self.rnet = rnet2dagrnet(self.rnet)
        self.directed = True
        return self

    def cpdag(self):
        self.rnet = rnet2cpdagrnet(self.rnet)
        self.directed = False
        return self

    def dot(self, engine='fdp'):
        if self.directed:
            return dot(*rnet2dag(self.rnet), engine=engine)
        else:
            return dot(*rnet2cpdag(self.rnet), engine=engine)

    def score(self, ldf):
        return score(self.rnet, ldf)

    def vstructs(self):
        return vstructs(self.rnet)

    def drop_arc(self, frm, to, inplace=False):
        rnet = drop_arc(self.rnet, frm=frm, to=to)
        if not inplace:
            return BayesNetworkStructure(rnet)
        self.rnet = rnet
        return self


class BayesNetworkBase():

    def __init__(self):
        self.df    = None
        self.rnet  = None
        self.rfit  = None
        self.grain = None

    def exact_query(self,evidence, nodes, only_python_result=True):
        rlistfn = rpy2.robjects.r['list']
        rsetevidencefn = rpy2.robjects.r['setEvidence']
        rquerygrainfn = rpy2.robjects.r['querygrain']

        revidence = rlistfn(**evidence)
        # setEvidence(net1, evidence=list(asia="yes", dysp="yes"))
        lgrain = rsetevidencefn(self.grain, evidence=revidence)
        rnodes = rpy2.robjects.StrVector(nodes)
        # querygrain(jtree, nodes=c("GRASSWET"))#$GRASSWET
        rresult = rquerygrainfn(lgrain, nodes=rnodes)
        result_list = []
        for rr in list(rresult):
            names = list(rr.names[0])
            result_list += [dict(zip(names,list(rr)))]

        result = dict(zip(rresult.names, result_list))
        if only_python_result:
            return result
        else:
            return result, rresult

    def to_xrds(self):
        xrds,_ = convert_to_xarray_dataset(self.rfit)
        return xrds

    def to_dtcpm_dict(self):
        _,dtcpm_dict = convert_to_xarray_dataset(self.rfit)
        return dtcpm_dict

    def write_net(self, file_name):
        rwritefn = rpy2.robjects.r['write.net']
        rwritefn(file_name, self.rfit)

    def write_netcdf(self, file_name):
        xrds = self.to_xrds()
        xrds.to_netcdf(file_name)

    def structure(self):
        return BayesNetworkStructure(self.rnet)

class CustomDiscreteBayesNetwork(BayesNetworkBase):

    def __init__(self, ldf_list, xrds=None):
        if xrds is not None:
            self.dtcpm_list = [DiscreteTabularCPM(None, lcpm_xr=xrds[lcpm_xr_var]) for lcpm_xr_var in xrds.data_vars]
        else:
            self.dtcpm_list = [DiscreteTabularCPM(ldf) for ldf in ldf_list]
        self.generate_model_string()
        self.generate_bnlearn_net()
        self.generate_custom_fit()

    def generate_model_string(self):
        model_string_list = [dtcpm.model_string for dtcpm in self.dtcpm_list]
        self.model_string = ''.join(model_string_list)

    def generate_bnlearn_net(self):
        model2network = rpy2.robjects.r['model2network']
        r_model_string = rpy2.robjects.StrVector([self.model_string])
        self.rnet = model2network(r_model_string)

    def generate_custom_fit(self):
        # dfit = custom.fit(net, dist = list("disease_present" = mR2, "test_result" = t(mR)))
        r_dist_list_0 = {}
        for dtcpm in self.dtcpm_list:
            r_dist_list_0.update({dtcpm.target : dtcpm.rarray})

        rlistfn = rpy2.robjects.r['list']
        r_dist_list = rlistfn(**r_dist_list_0)

        rcustomfitfn = rpy2.robjects.r['custom.fit']
        self.rfit = rcustomfitfn(self.rnet, dist=r_dist_list)

        # compile(as.grain(dfit))
        rcompilefn = rpy2.robjects.r['compile']
        rasgrainfn = rpy2.robjects.r['as.grain']
        self.grain = rcompilefn(rasgrainfn(self.rfit))


# https://realpython.com/instance-class-and-static-methods-demystified/
# https://pandas.pydata.org/pandas-docs/stable/categorical.html

def discretize(ldf, breaks=3, method='hartemink', ibreaks=5, idisc='quantile'):
    tmp_var_name_in  = next(tempfile._get_candidate_names())
    tmp_var_name_out = next(tempfile._get_candidate_names())
    rpy2.robjects.globalenv[tmp_var_name_in] = ldf

    rdiscretize = rpy2.robjects.r['discretize']
    rlevels = rpy2.robjects.r['levels']

    rpy2.robjects.globalenv[tmp_var_name_out] = rdiscretize(rpy2.robjects.globalenv[tmp_var_name_in], breaks=breaks, method=method, ibreaks=ibreaks, idisc=idisc)
    rdf_ = rpy2.robjects.globalenv[tmp_var_name_out]
    rdf = rpy2.robjects.pandas2ri.ri2py(rdf_)

    columns = rdf.columns
    for i, column in enumerate(columns):
        print((i,column))
        levels = rlevels(rdf_[i])
        print(levels)
        rdf[column] = pd.Categorical(rdf[column], categories=levels, ordered=True)

    rpy2.robjects.globalenv[tmp_var_name_in]  = np.nan
    rpy2.robjects.globalenv[tmp_var_name_out] = np.nan

    return rdf


class LearningBayesNetworkBase(BayesNetworkBase, sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, ldf):
        self.df = ldf

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def generate_fit(self):
        rfitfn = rpy2.robjects.r['bn.fit']
        self.rfit = rfitfn(self.rnet, data=self.df)

        # compile(as.grain(dfit))
        rcompilefn = rpy2.robjects.r['compile']
        rasgrainfn = rpy2.robjects.r['as.grain']
        self.grain = rcompilefn(rasgrainfn(self.rfit))


class NetAndDataDiscreteBayesNetwork(BayesNetworkBase):

    def __init__(self, dg, ldf):
        self.dg = dg
        self.df = ldf
        self.generate_model_string()
        self.generate_bnlearn_net()
        self.generate_fit()

    def generate_model_string_for_node(self, node):
        target_var = node
        parent_vars = list(self.dg.predecessors(node))
        return generate_model_string(target_var, parent_vars)

    def generate_model_string(self):
        sorted_node_list = list(networkx.algorithms.dag.topological_sort(self.dg))
        model_string = ''
        for node in sorted_node_list:
            model_string += self.generate_model_string_for_node(node)

        self.model_string = model_string


    def generate_bnlearn_net(self):
        model2network = rpy2.robjects.r['model2network']
        r_model_string = rpy2.robjects.StrVector([self.model_string])
        self.rnet = model2network(r_model_string)

    def generate_fit(self):
        rfitfn = rpy2.robjects.r['bn.fit']
        self.rfit = rfitfn(self.rnet, data=self.df)

        # compile(as.grain(dfit))
        rcompilefn = rpy2.robjects.r['compile']
        rasgrainfn = rpy2.robjects.r['as.grain']
        self.grain = rcompilefn(rasgrainfn(self.rfit))


# mc-mi: monte-carlo mutual information
def constrained_base_structure_learning_si_hiton_pc(ldf, test="mc-mi", undirected=False):
    rhitonpcfn = rpy2.robjects.r['si.hiton.pc']
    return rhitonpcfn(rpy2.robjects.pandas2ri.py2ri(ldf), test=test, undirected=undirected)

# http://www.bnlearn.com/documentation/man/constraint.html
# si.hiton.pc(x, cluster = NULL, whitelist = NULL, blacklist = NULL, test = NULL, alpha = 0.05, B = NULL, max.sx = NULL, debug = FALSE, optimized = FALSE, strict = FALSE, undirected = TRUE)
class ConstraintBasedNetFromDataDiscreteBayesNetwork(LearningBayesNetworkBase):

    def __init__(self, ldf, algorithm='HITON-PC'):
        super(ConstraintBasedNetFromDataDiscreteBayesNetwork, self).__init__(ldf)
        self.algorithmfn = None
        if algorithm == 'HITON-PC':
            self.algorithmfn = lambda ldf: constrained_base_structure_learning_si_hiton_pc(ldf)

    def fit(self, X=None, y=None):
        self.rnet = self.algorithmfn(self.df)
        self.rnet = rnet2dagrnet(self.rnet)
        self.generate_fit()
        return self

# http://www.bnlearn.com/documentation/man/hc.html
# hc(rdf_lt, score = "bic", iss=1, restart = 10, perturb = 5, start = random.graph(names(rdf_lt)))
def score_base_structure_learning_hill_climbing(ldf, score='bic', iss=1, restart=10, perturb=5, start='random_graph'):
    rdf = rpy2.robjects.pandas2ri.py2ri(ldf)
    rhcfn = rpy2.robjects.r['hc']
    if start == 'random_graph':
        rrandomgraphfn = rpy2.robjects.r['random.graph']
        rnamesfn = rpy2.robjects.r['names']
        start = rrandomgraphfn(rnamesfn(rdf))
    else:
        start = rpy2.rinterface.NULL
    return rhcfn(rpy2.robjects.pandas2ri.py2ri(ldf), score=score, iss=iss, restart=restart, perturb=perturb, start=start)


class ScoreBasedNetFromDataDiscreteBayesNetwork(LearningBayesNetworkBase):

    def __init__(self, ldf, algorithm='hc'):
        super(ScoreBasedNetFromDataDiscreteBayesNetwork, self).__init__(ldf)
        self.algorithmfn = None
        if algorithm == 'hc':
            self.algorithmfn = lambda ldf: score_base_structure_learning_hill_climbing(ldf)

    def fit(self, X=None, y=None):
        self.rnet = self.algorithmfn(self.df)
        self.rnet = rnet2dagrnet(self.rnet)
        self.generate_fit()
        return self


# http://www.bnlearn.com/documentation/man/hybrid.html
# rsmax2(rdf_lt, restrict = "si.hiton.pc", restrict.args = list(test = "x2", alpha = 0.01), maximize = "tabu", maximize.args = list(score = "bic", tabu = 10))

def convert_xarray_array_to_pandas_dtcpm(ar):
    dims = ar.dims
    new_dims = dims[1:] + dims[:1]
    ar = ar.transpose(*new_dims)
    stacked_ar = ar.stack(idx=new_dims)

    ldf = stacked_ar.to_pandas().reset_index()
    old_columns = list(ldf.columns)
    new_columns = old_columns[:-1] + ['p']
    # print(old_columns, new_columns)
    ldf.columns = new_columns

    return ldf

def convert_pandas_dtcpm_to_xarray(ldf):
    dtcpm = DiscreteTabularCPM(ldf)
    return dtcpm.lcpm_xr

def convert_xarray_dataset_to_pandas_dtcpm_dict(ds):
    rpd = {}
    for ar_name in ds.data_vars: # .keys()
        ar = ds[ar_name]
        ldf = convert_xarray_array_to_pandas_dtcpm(ar)
        rpd[ar_name] = ldf

    return rpd

def convert_to_xarray_dataset(rfit):
    rnodesfn = rpy2.robjects.r['nodes']
    nodes = list(rnodesfn(rfit))

    ds = xr.Dataset()
    for node in nodes:
        # rpy2.robjects.pandas2ri.ri2py()
        prob = rfit.rx(node)[0].rx('prob')[0]
        # print(prob)
        dims = prob.names # map from dim to levels
        # print(dims)
        coords = {}
        dim_names = []
        if len(dims) == 1:
            dname = node
            dim_names += [dname]
            levels = list(dims[0])
            coords.update({dname: levels})
        else:
            for dname in dims.names:
                dim_names += [dname]
                levels = list(dims.rx(dname)[0])
                coords.update({dname: levels})
        values = rpy2.robjects.pandas2ri.ri2py(prob)

        ar = xr.DataArray(values, dims = dim_names, coords= coords)
        ds['cpt' + node] = ar

    lpd = convert_xarray_dataset_to_pandas_dtcpm_dict(ds)

    did_adapt_ds_p = False
    for df_name in lpd.keys():
        ldf = lpd[df_name]
        null_index_combinations = ldf[pd.isnull(ldf['p'])][ldf.columns[:-2]]

        if len(null_index_combinations) == 0:
            continue
        null_index_combinations = null_index_combinations.drop_duplicates()
        did_adapt_ds_p = True
        lar = ds[df_name]
        target_var = lar.dims[0]
        target_var_levels = lar.coords[target_var]
        target_var_level_count = len(target_var_levels)
        fill_in_p = 1.0/target_var_level_count
        for index, row in null_index_combinations.iterrows():
            # print('row', row)
            dims = list(row.index)
            dim_values = row.values
            # print('dims,dim_values',dims,dim_values)
            coords = dict(zip(dims, dim_values))
            # print('coords',coords)
            # print('lar',lar)
            lar.loc[coords] = fill_in_p

    if did_adapt_ds_p:
        lpd = convert_xarray_dataset_to_pandas_dtcpm_dict(ds)

    return ds, lpd

def bnnet_from_pandas_dtcpm_list(dtcpm_list):
    return CustomDiscreteBayesNetwork(dtcpm_list)

def bnnet_from_netcdf_file(netcdf_file_name):
    xrds = xr.open_dataset(netcdf_file_name, autoclose=True)
    print(xrds)
    return CustomDiscreteBayesNetwork(None, xrds=xrds)

def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return list(zip(a, b))

def grouped(iterable, n):
    "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
    return list(zip(*[iter(iterable)]*n))


def cpdag(bn):
    rnet = rfit2rnet(bn.rfit)
    return rnet2cpdag(rnet)

def rnet2cpdagrnet(rnet):
    rcpdagfn = rpy2.robjects.r('cpdag')
    return rcpdagfn(rnet)

def rnet2cpdag(rnet):
    nodes, unidirectional_edges, bidirectional_edges = rnet2dag(rnet)

    rcpdagfn = rpy2.robjects.r('cpdag')
    rarcsfn  = rpy2.robjects.r('arcs')
    rnodesfn  = rpy2.robjects.r('nodes')
    tmp = rarcsfn(rcpdagfn(rnet))

    frm = list(tmp.rx(True,'from'))
    to  = list(tmp.rx(True,'to'))

    r1 = set(zip(frm,to))
    r2 = set(zip(to, frm))
    bde = r1 & r2

    bidirectional_edges = set(unidirectional_edges) & bde
    unidirectional_edges = set(unidirectional_edges) - bidirectional_edges

    nodes = list(rnodesfn(rnet))

    return nodes, unidirectional_edges, bidirectional_edges

def dag(bn):
    rnet = rfit2rnet(bn.rfit)
    return rnet2dag(rnet)

def rnet2dagrnet(rnet):
    rcextendfn = rpy2.robjects.r('cextend')
    return rcextendfn(rnet)

def rnet2dag(rnet):
    rarcsfn    = rpy2.robjects.r('arcs')
    rcextendfn = rpy2.robjects.r('cextend')
    rnodesfn  = rpy2.robjects.r('nodes')

    tmp = rarcsfn(rcextendfn(rnet))

    frm = list(tmp.rx(True,'from'))
    to  = list(tmp.rx(True,'to'))

    unidirectional_edges = list(zip(frm,to))

    nodes = list(rnodesfn(rnet))

    return nodes, unidirectional_edges, []

def vstructs(bn):
    rvstructsfn  = rpy2.robjects.r('vstructs')
    rvstructs = rvstructsfn(bn) # bn.rfit
    # print(rvstructs)
    x = list(rvstructs.rx(True, 'X'))
    z = list(rvstructs.rx(True, 'Z'))
    y = list(rvstructs.rx(True, 'Y'))
    return pd.DataFrame(collections.OrderedDict(X=x,Z=z,Y=y))


def score(bn, ldf, type='loglik'):
    rscorefn  = rpy2.robjects.r('score')
    return rscorefn(bn.rnet, data=ldf, type=type)[0]

def rfit2rnet(rfit):
    rbnnetfn = rpy2.robjects.r('bn.net')
    return rbnnetfn(rfit)

def drop_arc(rnet, frm, to):
    rdroparcfn  = rpy2.robjects.r('drop.arc')
    return rdroparcfn(rnet, **{'from': frm, 'to': to})


def dot(nodes, unidirectional_edges, bidirectional_edges, engine='fdp', graph_name='graph'):
    dg_dot = graphviz.Digraph(engine=engine, comment=graph_name)

    for node in nodes:
        dg_dot.node(node)

    for edge in unidirectional_edges:
        dg_dot.edge(edge[0], edge[1])

    for edge in bidirectional_edges:
        dg_dot.edge(edge[0], edge[1], dir='none')

    return dg_dot
