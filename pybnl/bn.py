
import locale
locale.setlocale(locale.LC_ALL, 'C')

import numpy as np, xarray as xr, pandas as pd, scipy, scipy.sparse
import networkx as nx, networkx.algorithms.dag, graphviz
import sklearn.base, sklearn.metrics, sklearn.metrics.cluster
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
import itertools, collections, tempfile, random, math
import warnings, re

import rpy2, rpy2.rinterface, rpy2.robjects, rpy2.robjects.packages, rpy2.robjects.lib, rpy2.robjects.lib.grid, \
    rpy2.robjects.lib.ggplot2, rpy2.robjects.pandas2ri, rpy2.interactive.process_revents, \
    rpy2.interactive, rpy2.robjects.lib.grdevices, rpy2.rlike.container
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


def validate_node_or_level_name(name):
    if re.match(r'[A-Za-z][A-Za-z0-9_]*', name) is None:
        raise RuntimeError('You should only use node and/or factor level names that match the regex [A-Za-z][A-Za-z0-9_]*. You used: {}'.format(name))

def validate_node_or_level_names(name_list):
    for name in name_list:
        validate_node_or_level_name(name)

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

        validate_node_or_level_names(self.ldf.columns)

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

def bf(rnet1, rnet2, ldf):
    rbffn = rpy2.robjects.r('BF')
    r_df = pydf_to_factorrdf(ldf)
    return rbffn(rnet1, rnet2, r_df, score="bde", iss=1)[0]


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

    def score(self, ldf, type='bic'):
        return rnetscore(self.rnet, ldf, type=type)

    def vstructs(self):
        return vstructs(self.rnet)

    def drop_arc(self, frm, to, inplace=False):
        rnet = drop_arc(self.rnet, frm=frm, to=to)
        if not inplace:
            return BayesNetworkStructure(rnet)
        self.rnet = rnet
        return self

    def nodes(self):
        rnodesfn = rpy2.robjects.r('nodes')
        nodes = list(rnodesfn(self.rnet))
        return nodes

    def bf(self,other, ldf):
        return bf(self.rnet, other.rnet, ldf)

class BayesNetworkBase():

    def __init__(self):
        self.df    = None
        self.rnet  = None
        self.rfit  = None
        self.grain = None

    def exact_query(self, evidence, nodes, only_python_result=True):
        evidence_nodes = evidence.keys()
        net_nodes = self.structure().nodes()
        for node in evidence_nodes:
            if node not in net_nodes:
                raise RuntimeError('evidence node: {} is not present in network: {}'.format(node, nodes))

        for node in nodes:
            if node not in net_nodes:
                raise RuntimeError('query node: {} is not present in network: {}'.format(node, nodes))

        for node, value in evidence.items():
            # print('node, value: {}, {}'.format(node, value))
            if len(self.rfit.rx(node)[0].rx('prob')[0].names) == 1:
                allowed_values = list(self.rfit.rx(node)[0].rx('prob')[0].names[0])
            elif len(self.rfit.rx(node)[0].rx('prob')[0].names) > 1:
                allowed_values = list(self.rfit.rx(node)[0].rx('prob')[0].names.rx(node)[0])
            else:
                raise RuntimeError('Should never happen! node, value: {}, {}'.format(node,value))
            if value not in allowed_values:
                raise RuntimeError('evidence node: {} value: {} does not exist in the categories: {}'.format(node, value, allowed_values))

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

    def score(self, Xt=None, y=None, type = 'bic'):
        return self.structure().score(self.df, type=type)

    def bf(self, other, ldf=None):
        if ldf is None:
            ldf = self.df
        return self.structure().bf(other.structure(), ldf)

    def arc_strength_info(self, criterion='loglik'):
        arc_mi_df        = bn_arcs_mutual_information_infos(self)
        arc_strengths_df = bn_arcs_strengths(self, criterion=criterion)
        ldf = pd.merge(arc_strengths_df, arc_mi_df, on=['from', 'to'])[['from', 'to', 'strength', 'relative_mutual_information_from', 'relative_mutual_information_to']]
        ldf.rename(columns={'relative_mutual_information_from': 'rmif', 'relative_mutual_information_to': 'rmit'}, inplace=True)
        max_strength = ldf.strength.min()
        ldf['rs'] = ldf.strength / max_strength
        return ldf[['from', 'to', 'strength', 'rs', 'rmif', 'rmit']]

    def dot(self, engine='fdp', criterion='loglik'):
        return dot_with_arc_strength_info(*rnet2dag(self.rnet), self.arc_strength_info(criterion=criterion), engine=engine)


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
# https://rpy2.github.io/doc/v2.9.x/html/vector.html#dataframe

def pydf_to_factorrdf(ldf):
    check_dtype_categorical(ldf)
    latent = identify_latent_variables(ldf)
    NA_lds = None
    if len(latent) > 0:
        print('identified latent variables: {}'.format(latent))
        NA_lds = lds = [rpy2.rinterface.NA_Character for _ in range(len(ldf))]
    # r_df = rpy2.robjects.pandas2ri.py2ri(ldf)
    # rpy2.rlike.container.OrdDict([('value', robjects.IntVector((1,2,3))), ('letter', robjects.StrVector(('x', 'y', 'z')))])
    # dataf = robjects.DataFrame(od)

    cols = []
    colnames = list(ldf.columns)
    for colname in colnames:
        cat_type = ldf[colname].dtype
        levels = rpy2.robjects.StrVector(list(cat_type.categories))
        ordered = cat_type.ordered
        # if colname == 'Bsmt_Full_Bath':
        #     print('col: {}, levels: {}, ordered: {}'.format(colname, levels, ordered))

        lds = ldf[colname]
        factorized_column = None
        if colname in latent:
            factorized_column =  rpy2.robjects.vectors.FactorVector(rpy2.robjects.StrVector(NA_lds), levels=levels, ordered=ordered)
        else:
            factorized_column =  rpy2.robjects.vectors.FactorVector(rpy2.robjects.StrVector(lds), levels=levels, ordered=ordered)
        cols += [factorized_column]

    od = rpy2.rlike.container.OrdDict([(colnames[i], col) for i,col in enumerate(cols)])
    r_df = rpy2.robjects.DataFrame(od)

    return r_df

def factorrdf_to_pydf(r_df):
    colnames = list(r_df.colnames)
    ldf = pd.DataFrame(columns=colnames)# rpy2.robjects.pandas2ri.ri2py(r_df)
    for i, colname in enumerate(colnames):
        levels = list(r_df[i].levels)
        ordered = r_df[i].isordered
        cdt = pd.api.types.CategoricalDtype(levels, ordered=ordered)

        all_values_na = rpy2.robjects.r('all')(rpy2.robjects.r('is.na')(r_df[i]))[0]
        if all_values_na:
            ldf[colname] = pd.Series(np.nan).astype(cdt)
        else:
            ldf[colname] = pd.Series(r_df[i].iter_labels()).astype(cdt)

    return ldf


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
        # print((i,column))
        levels = rlevels(rdf_[i])
        # print(levels)
        rdf[column] = pd.Categorical(rdf[column], categories=levels, ordered=True)

    rpy2.robjects.globalenv[tmp_var_name_in]  = rpy2.rinterface.NULL
    rpy2.robjects.globalenv[tmp_var_name_out] = rpy2.rinterface.NULL

    return rdf


# if isinstance(l,(list,pd.core.series.Series,np.ndarray)):
def sklearn_fit_helper_transform_X(X):
    # print('sklearn_fit_helper_transform_X')
    ldf = None
    # if isinstance(l,(list,pd.core.series.Series,np.ndarray)):
    if isinstance(X, pd.DataFrame):
        ldf = X
    elif isinstance(X, np.ndarray):
        column_names = ['X{}'.format(i) for i in range(X.shape[1])]
        ldf = pd.DataFrame(X, columns=column_names)
    else:
        raise ValueError('Only accepting pandas ')

    validate_node_or_level_names(X.columns)

    # check_array(ldf)
    if len(ldf.shape) != 2:
        raise RuntimeError("X has invalid shape!")
    return ldf


# https://stats.stackexchange.com/questions/5253/how-do-i-get-the-number-of-rows-of-a-data-frame-in-r
# https://stackoverflow.com/questions/14808945/check-if-variable-is-dataframe
# https://stackoverflow.com/questions/1549801/what-are-the-differences-between-type-and-isinstance
class HarteminkBinTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, breaks, ibreaks=None):
        self.breaks  = breaks
        self.ibreaks = ibreaks

    def fit(self, X=None, y=None, r_df=None, seed=None):
        # X = check_array(X)
        # if len(X.shape) != 2:
        #     raise RuntimeError("X has invalid shape!")

        self.df_ = sklearn_fit_helper_transform_X(X)

        if r_df is None:
            df = self.df_
            # find all non categorical numeric columns and assume that they are to be binned
            numeric_columns = []
            categorical_columns = []
            for col in df.columns:
                if df[col].dtype.name == 'category':
                    categorical_columns += [col]
                    continue
                if not np.issubdtype(df[col].dtype, np.number):
                    continue
                numeric_columns += [col]
            r_df = rpy2.robjects.pandas2ri.py2ri(df[numeric_columns])

        if self.ibreaks is None:
            rnrow       = rpy2.robjects.r['nrow']
            unqiue_length = rnrow(r_df)[0]
            colnames = list(r_df.colnames)
            for i, colname in enumerate(colnames):
                ul = len(np.unique(r_df[i]))
                unqiue_length = np.min([unqiue_length, ul])
        else:
            unqiue_length = self.ibreaks

        if seed is None:
            rsetseed()
        else:
            rsetseed(seed=seed)
        rdiscretize = rpy2.robjects.r['discretize']
        self.r_ddf_ = None
        while self.r_ddf_ is None and unqiue_length > self.breaks:
            try:
                self.r_ddf_ = rdiscretize(r_df, breaks=self.breaks, method='hartemink', ibreaks=unqiue_length, idisc='quantile')
            except rpy2.rinterface.RRuntimeError as e:
                unqiue_length -= 1
                warnings.warn("ibreaks problem, reducing ibreaks to {}".format(unqiue_length), rpy2.rinterface.RRuntimeWarning)

        self.ddf_ = factorrdf_to_pydf(self.r_ddf_)

        return self

    def transform(self, X=None):
        try:
            getattr(self, "ddf_")
        except AttributeError:
            raise RuntimeError("You must call fit before calling transform!")

        # X = check_array(X)

        return self.ddf_

def initial_random_impute(ldf, cutoff=30):
    imputed = ldf.copy()
    _, discrete_with_null, continuous_non_null, continuous_with_null, _ = discrete_and_continuous_variables_with_and_without_nulls(ldf, cutoff=cutoff)
    latent_levels = dict()
    for ln in discrete_with_null:
        levels = levels_of_latent_variable(ldf, ln)
        latent_levels[ln] = levels

    for ln in discrete_with_null:
        levels = latent_levels[ln]
        random.seed(global_default_random_seed_init)
        lds = ldf[ln]
        k = sum(pd.isnull(lds))
        imputed.loc[pd.isnull(lds), ln] = random.choices(levels, k=k)
        imputed[ln] = imputed[ln].astype(ldf[ln].dtype)

    return imputed

def initial_random_impute_fit(ldf, dg=None, rnet=None, cutoff=30):
    imputed = initial_random_impute(ldf, cutoff=cutoff)
    f = NetAndDataDiscreteBayesNetwork(imputed, dg=dg, rnet=rnet)
    f.fit()
    return f

# fitted = bn.fit(empty.graph(names(ldmarks)), imputed)
#


class ExactInferenceImputer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, f, ldf):
        # super(ExactInferenceImputer, self).__init__()
        self.f = f
        self.df  = ldf

    def __repr__(self):
        class_name = self.__class__.__name__
        return class_name

    def fit(self, X=None, y=None, seed=None):
        if X is None:
            ldf = self.df
        else:
            ldf = sklearn_fit_helper_transform_X(X)

        ldf_imputed = ldf.copy()

        rs_init = len(ldf)
        rs = np.random.RandomState(rs_init)

        lds_rows_with_null_values         = ldf.apply(pd.isnull, axis=1).any(axis=1)
        lds_rows_with_null_values_indices = ldf.index[lds_rows_with_null_values]
        for idx in list(lds_rows_with_null_values_indices):
            row = ldf.loc[idx,:]
            row_null_values       = pd.isnull(row)
            null_column_names     = ldf.columns[row_null_values]
            non_null_column_names = ldf.columns[~row_null_values]
            non_null_values       = row[~row_null_values]
            evidence    = dict(zip(non_null_column_names, non_null_values))
            query_nodes = list(null_column_names)
            # print('evidence: {}, query_nodes: {}'.format(evidence, query_nodes))
            answers = self.f.exact_query(evidence, query_nodes)
            # print('answers: {}'.format(answers))
            imputed_values = []
            for node in query_nodes:
                answer = answers[node]
                levels = list(answer.keys())
                probabilities = [answer[level] for level in levels]
                # print('node: {}, levels: {}, probabilities: {}'.format(node, levels, probabilities))
                imputed_value =  rs.choice(levels,1, p=probabilities)[0]
                # print('node: {}, levels: {}, probabilities: {}: impute_value: {}'.format(node, levels, probabilities, impute_value))
                imputed_values += [imputed_value]

            ldf_imputed.loc[idx, query_nodes] = imputed_values

        self.imputed_ = ldf_imputed

        return self

    def transform(self, X=None):
        try:
            getattr(self, "imputed_")
        except AttributeError:
            raise RuntimeError("You must call fit before calling transform!")

        return self.imputed_


class BNLearnImputer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, f, ldf):
        self.f = f
        self.df  = ldf

    def fit(self, X=None, y=None, r_df=None, seed=None):
        # X = check_array(X)
        # if len(X.shape) != 2:
        #     raise RuntimeError("X has invalid shape!")

        if self.df.columns.isin(['Bsmt_Full_Bath']).any():
            print('Imputer in: {}'.format(self.df['Bsmt_Full_Bath'].dtype.categories))
        if r_df is None:
            r_df = pydf_to_factorrdf(self.df)

        if seed is None:
            rsetseed()
        else:
            rsetseed(seed=seed)
        self.r_df_ = r_df
        self.rimputed_ = rpy2.robjects.r('impute')(self.f.rfit, r_df, method='bayes-lw')
        self.imputed_ = factorrdf_to_pydf(self.rimputed_)
        if self.df.columns.isin(['Bsmt_Full_Bath']).any():
            print('Imputer out: {}'.format(self.imputed_['Bsmt_Full_Bath'].dtype.categories))
            print('Bsmt_Full_Bath diff: {}'.format(self.imputed_['Bsmt_Full_Bath'].astype(str).values == self.df['Bsmt_Full_Bath'].astype(str).values))

        return self

    def transform(self, X=None):
        try:
            getattr(self, "imputed_")
        except AttributeError:
            raise RuntimeError("You must call fit before calling transform!")

        # X = check_array(X)

        return self.imputed_


global_default_random_seed_init = 42

# random.seed np.random.seed np.random.RandomState
def rsetseed(seed=global_default_random_seed_init):
    rsetseedfn = rpy2.robjects.r('set.seed')
    rsetseedfn(seed)

def check_dtype_categorical(ldf):
    # print(type(ldf))
    for column in list(ldf.columns):
        if ldf[column].dtype.name != 'category' or not all(isinstance(item, str) for item in list(ldf[column].dtype.categories)):
            raise ValueError('Dataframe needs to contain only categorical data columns and the categories have to be string values! column: {}'.format(column))

class LearningBayesNetworkBase(BayesNetworkBase, sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, ldf):
        if ldf is not None:
            validate_node_or_level_names(ldf.columns)
        self.df = ldf
        self.df_for_metadata = ldf
        # print('LearningBayesNetworkBase: pydf_to_factorrdf')

    def fit(self, X, y=None, seed=None):
        if seed is None:
            rsetseed()
        else:
            rsetseed(seed=seed)

        if X is not None:
            self.df = sklearn_fit_helper_transform_X(X)

        if self.df is None:
            raise RuntimeError('You either have to specify the data frame when you call the constructur (the ldf parameter) or when you call the fit (the X parameter) method!')
        else:
            check_dtype_categorical(self.df)

        return self

    def transform(self, X):
        return X

    def generate_fit(self):
        rfitfn = rpy2.robjects.r['bn.fit']
        self.r_df_ = pydf_to_factorrdf(self.df)
        self.rfit = rfitfn(self.rnet, data=self.r_df_)

        # compile(as.grain(dfit))
        rcompilefn = rpy2.robjects.r['compile']
        rasgrainfn = rpy2.robjects.r['as.grain']
        self.grain = rcompilefn(rasgrainfn(self.rfit))


def generate_model_string_for_node(dg, node):
    target_var = node
    parent_vars = list(dg.predecessors(node))
    return generate_model_string(target_var, parent_vars)

def digraph2rnet(dg):
    sorted_node_list = list(networkx.algorithms.dag.topological_sort(dg))
    model_string = ''
    for node in sorted_node_list:
        model_string += generate_model_string_for_node(dg, node)

    model2network = rpy2.robjects.r['model2network']
    r_model_string = rpy2.robjects.StrVector([model_string])
    return model2network(r_model_string)

def digraph2netstruct(dg):
    rnet = digraph2rnet(dg)
    return BayesNetworkStructure(rnet)

class NetAndDataDiscreteBayesNetwork(LearningBayesNetworkBase):

    def __init__(self, ldf=None, dg=None, model_string=None, rnet=None):
        super(NetAndDataDiscreteBayesNetwork, self).__init__(ldf)
        self.dg = dg
        self.model_string = model_string
        self.rnet = rnet
        if dg is not None:
            self.generate_model_string()
            self.generate_bnlearn_net()
            return

        if model_string is not None:
            self.generate_bnlearn_net()
            return

        if rnet is not None:
            return

        raise ValueError('Exactly one of dg, model_string or rnet need to be supplied')

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

    def fit(self, X=None, y=None, seed=None):
        super(NetAndDataDiscreteBayesNetwork, self).fit(X=X, y=y, seed=seed)
        self.generate_fit()
        return self

# https://github.com/jacintoArias/bayesnetRtutorial/blob/master/index.Rmd
class ParametricEMNetAndDataDiscreteBayesNetwork(NetAndDataDiscreteBayesNetwork):

    def __init__(self, ldf=None, dg=None, model_string=None, rnet=None, discrete_variable_identification_cutoff=30):
        super(ParametricEMNetAndDataDiscreteBayesNetwork, self).__init__(ldf=ldf, dg=dg, model_string=model_string, rnet=rnet)

        _, discrete_with_null, continuous_non_null, continuous_with_null, _ = discrete_and_continuous_variables_with_and_without_nulls(ldf, cutoff=discrete_variable_identification_cutoff)
        if len(continuous_non_null) > 0 or len(continuous_with_null) > 0:
            raise RuntimeError('DiscreteBayesNetwork can only handle discrete variables, but you provided continuous ones: {}'.format(continuous_non_null + continuous_with_null))

        self.discrete_with_null = discrete_with_null
        self.latent_levels = dict()
        for ln in self.discrete_with_null:
            levels = levels_of_latent_variable(self.df, ln)
            self.latent_levels[ln] = levels

    def fit(self, X=None, y=None, seed=None):
        LearningBayesNetworkBase.fit(self, X=X, y=y, seed=seed)

        self.r_df_ = pydf_to_factorrdf(self.df)

        if self.df.columns.isin(['Bsmt_Full_Bath']).any():
            print('self.df: {}'.format(self.df.Bsmt_Full_Bath.dtype.categories))
        imputed = self.df.copy()
        if self.df.columns.isin(['Bsmt_Full_Bath']).any():
            print('imputed: {}'.format(imputed.Bsmt_Full_Bath.dtype.categories))

        for ln in self.discrete_with_null:
            levels = self.latent_levels[ln]
            random.seed(global_default_random_seed_init)
            lds = self.df[ln]
            k = sum(pd.isnull(lds))
            imputed.loc[pd.isnull(lds),ln] = random.choices(levels,k=k)
            imputed[ln] = imputed[ln].astype(self.df[ln].dtype)

        # fitted = bn.fit(empty.graph(names(ldmarks)), imputed)
        f = NetAndDataDiscreteBayesNetwork(imputed, rnet=self.rnet)
        self.f_ = f
        f.fit(seed=seed)

        return self
        for i in range(15):
            print('ParametricEMNetAndDataDiscreteBayesNetwork iteration: {}'.format(i))
            # expectation step.
            im = Imputer(f, self.df)
            self.im_ = im
            imputed = im.fit_transform(X=None, r_df=self.r_df_, seed=seed)
            if self.df.columns.isin(['Bsmt_Full_Bath']).any():
                print('imputed: {}'.format(imputed.Bsmt_Full_Bath.dtype.categories))
            # maximisation step
            f_new = NetAndDataDiscreteBayesNetwork(imputed, rnet=self.rnet)
            f_new.fit()
            if rall_equal_fits(f.rfit, f_new.rfit):
                break
            else:
                f = f_new
                self.f_ = f

        print(i)

        self.rfit = f.rfit
        rcompilefn = rpy2.robjects.r['compile']
        rasgrainfn = rpy2.robjects.r['as.grain']
        self.grain = rcompilefn(rasgrainfn(self.rfit))

        self.imputed_ = imputed

        return self


# mc-mi: monte-carlo mutual information
def constrained_base_structure_learning_si_hiton_pc(ldf, test="mc-mi", undirected=False):
    rhitonpcfn = rpy2.robjects.r['si.hiton.pc']
    return rhitonpcfn(rpy2.robjects.pandas2ri.py2ri(ldf), test=test, undirected=undirected)

# http://www.bnlearn.com/documentation/man/constraint.html
# si.hiton.pc(x, cluster = NULL, whitelist = NULL, blacklist = NULL, test = NULL, alpha = 0.05, B = NULL, max.sx = NULL, debug = FALSE, optimized = FALSE, strict = FALSE, undirected = TRUE)
class ConstraintBasedNetFromDataDiscreteBayesNetwork(LearningBayesNetworkBase):

    def __init__(self, ldf=None, algorithm='HITON-PC'):
        super(ConstraintBasedNetFromDataDiscreteBayesNetwork, self).__init__(ldf)
        self.algorithmfn = None
        if algorithm == 'HITON-PC':
            self.algorithmfn = lambda ldf: constrained_base_structure_learning_si_hiton_pc(ldf)

    def fit(self, X=None, y=None, seed=None):
        super(ConstraintBasedNetFromDataDiscreteBayesNetwork, self).fit(X=X, y=y, seed=seed)
        self.rnet = self.algorithmfn(self.df)
        self.rnet = rnet2dagrnet(self.rnet)
        self.generate_fit()
        return self

# http://www.bnlearn.com/documentation/man/hc.html
# hc(rdf_lt, score = "bic", iss=1, restart = 10, perturb = 5, start = random.graph(names(rdf_lt)))
def score_base_structure_learning_hill_climbing(ldf, score='bic', iss=1, restart=10, perturb=5, start='random_graph', whitelist=None):
    rdf = rpy2.robjects.pandas2ri.py2ri(ldf)
    rhcfn = rpy2.robjects.r['hc']
    if start == 'random_graph':
        rrandomgraphfn = rpy2.robjects.r['random.graph']
        rnamesfn = rpy2.robjects.r['names']
        start = rrandomgraphfn(rnamesfn(rdf))
    else:
        start = rpy2.rinterface.NULL
    if whitelist is not None:
        rwl = rpy2.robjects.pandas2ri.py2ri(whitelist)
    else:
        rwl = rpy2.rinterface.MissingArg
    return rhcfn(rdf, score=score, iss=iss, restart=restart, perturb=perturb, start=start, whitelist=rwl)


class ScoreBasedNetFromDataDiscreteBayesNetwork(LearningBayesNetworkBase):

    def __init__(self, ldf=None, algorithm='hc', whitelist=None):
        super(ScoreBasedNetFromDataDiscreteBayesNetwork, self).__init__(ldf)
        self.algorithmfn = None
        if algorithm == 'hc':
            self.algorithmfn = lambda ldf: score_base_structure_learning_hill_climbing(ldf, whitelist=whitelist)

    def fit(self, X=None, y=None, seed=None):
        super(ScoreBasedNetFromDataDiscreteBayesNetwork, self).fit(X=X, y=y, seed=seed)
        self.rnet = self.algorithmfn(self.df)
        self.rnet = rnet2dagrnet(self.rnet)
        self.generate_fit()
        return self

class StructuralEMNetFromDataDiscreteBayesNetworkBase(LearningBayesNetworkBase):
    def __init__(self, ldf=None):
        super().__init__(ldf)
        self.latent_names = identify_latent_variables(self.df)
        if len(self.latent_names) == 0:
            raise ValueError('Expecting a dataframe with some latent variables, but does not contain any!')
        self.latent_levels = dict()
        for ln in self.latent_names:
            levels = levels_of_latent_variable(self.df, ln)
            self.latent_levels[ln] = levels

    def sem_fit_base(self, X=None, y=None, seed=None):
        super().fit(X=X, y=y, seed=seed)
        r_df = pydf_to_factorrdf(self.df)
        imputed = self.sem_generate_inital_impute()
        f = self.sem_generate_inital_fit(imputed, seed)
        whitelist = self.sem_generate_whitelist()
        return (r_df, imputed, f, whitelist)

    # https://docs.python.org/3/library/random.html
    # random.choices(): with replacement
    # random.sample(): without replacement
    def sem_generate_inital_impute(self):
        imputed = self.df.copy()

        k = len(self.df)
        for ln in self.latent_names:
            levels = self.latent_levels[ln]
            random.seed(global_default_random_seed_init)
            imputed.loc[:,ln] = random.choices(levels,k=k)
            imputed[ln] = imputed[ln].astype(self.df[ln].dtype)
        return imputed

    def sem_generate_inital_fit(self, imputed, seed):
        # fitted = bn.fit(empty.graph(names(ldmarks)), imputed)
        f = NetAndDataDiscreteBayesNetwork(imputed, rnet=empty_graph(imputed.columns))
        f.fit(seed=seed)

        # fitted$LAT = array(c(0.5, 0.5), dim = 2, dimnames = list(c("A", "B")))
        for ln in self.latent_names:
            levels = self.latent_levels[ln]
            count = len(levels)
            a = np.full([count], float(1 / count))
            dimnames = rpy2.robjects.r['list'](rpy2.robjects.StrVector(levels))
            ra = rpy2.robjects.r['array'](a, dim=count, dimnames=dimnames)
            idx = list(imputed.columns).index(ln)
            prob_idx = list(f.rfit[idx].names).index('prob')
            f.rfit[idx][prob_idx] = ra

        return f

    def sem_generate_whitelist(self):
        wl = set(itertools.product(self.latent_names, self.df.columns))
        wl_substract = set(zip(self.df.columns, self.df.columns))
        whitelist = pd.DataFrame(list(wl - wl_substract), columns=['from', 'to']).sort_values(['from','to'])
        return whitelist

    def score(self, Xt=None, y=None, type = 'bic'):
        return self.structure().score(self.imputed_, type=type)


class StructuralEMNetFromDataDiscreteBayesNetwork(StructuralEMNetFromDataDiscreteBayesNetworkBase):

    def __init__(self, ldf=None):
        super().__init__(ldf)

    # https://docs.python.org/3/library/random.html
    # random.choices(): with replacement
    # random.sample(): without replacement
    def fit(self, X=None, y=None, seed=None):
        super().fit(X=X, y=y, seed=seed)
        r_df, imputed, f, whitelist = super().sem_fit_base(X=X, y=y, seed=seed)

        rstructuralemfn = rpy2.robjects.r('structural.em')
        rlistfn         = rpy2.robjects.r('list')

        rmaximizeargs_in = {
            'whitelist': whitelist
        }
        rmaximizeargs = rlistfn(**rmaximizeargs_in)

        remaining_args = {
            'maximize.args': rmaximizeargs,
            'max.iter': 15,
            'return.all': True
        }

        # XXX implementation based on bnlearn::structural.em
        # r = structural.em(ldmarks, fit="bayes", impute="bayes-lw", start=fitted, maximize.args = list(whitelist=data.frame( from = "LAT", to = names(dmarks2))), return.all = TRUE)
        r = rstructuralemfn(r_df, fit="bayes", impute="bayes-lw", start=f.rfit, **remaining_args) # maximize.args = list(whitelist=data.frame( from = "LAT", to = names(dmarks2))), return.all = TRUE
        # r$dag (an object of class bn),
        # r$imputed (a data frame containing the imputed data from the last iteration) and
        # r$fitted
        # self.r_ = r

        self.rfit = r.rx('fitted')[0]
        self.rnet = r.rx('dag')[0] # rfit2rnet(self.rfit)
        rcompilefn = rpy2.robjects.r['compile']
        rasgrainfn = rpy2.robjects.r['as.grain']
        self.grain = rcompilefn(rasgrainfn(self.rfit))

        self.imputed_ = factorrdf_to_pydf(r.rx('imputed')[0])

        return self



class PyStructuralEMNetFromDataDiscreteBayesNetwork(StructuralEMNetFromDataDiscreteBayesNetworkBase):

    def __init__(self, ldf=None):
        super().__init__(ldf)

    def fit(self, X=None, y=None, seed=None):
        super().fit(X=X, y=y, seed=seed)
        r_df, imputed, f, whitelist = super().sem_fit_base(X=X, y=y, seed=seed)

        for i in range(15):
            # expectation step.
            im = ExactInferenceImputer(f, self.df)
            imputed = im.fit_transform(X=None, r_df=r_df, seed=seed)

            # maximisation step
            f_new = ScoreBasedNetFromDataDiscreteBayesNetwork(imputed, whitelist=whitelist)
            f_new.fit()
            if rall_equal_fits(f.rfit, f_new.rfit):
                break
            else:
                f = f_new

        print(i)

        self.rfit = f.rfit
        self.rnet = rfit2rnet(self.rfit)
        rcompilefn = rpy2.robjects.r['compile']
        rasgrainfn = rpy2.robjects.r['as.grain']
        self.grain = rcompilefn(rasgrainfn(self.rfit))

        self.imputed_ = imputed

        return self

# http://www.bnlearn.com/documentation/man/hybrid.html
# rsmax2(rdf_lt, restrict = "si.hiton.pc", restrict.args = list(test = "x2", alpha = 0.01), maximize = "tabu", maximize.args = list(score = "bic", tabu = 10))
# rsmax2(rdf_lt, restrict = "mmpc", maximize = "hc")
# mmhc(rdf_lt)

# write factorrdf_to_pydf
# write impute function

def rall_equal_fits(rfit1, rfit2):
    rallequalfn = rpy2.robjects.r['all.equal']
    ristruefn = rpy2.robjects.r['isTRUE']
    return ristruefn(rallequalfn(rfit1, rfit2))[0]


def empty_graph(node_names):
    node_names = list(node_names)
    node_names = rpy2.robjects.StrVector(node_names)
    remptygraphfn = rpy2.robjects.r['empty.graph']
    return remptygraphfn(node_names)

def identify_latent_variables(ldf):
    r = []
    for column in ldf.columns:
        lds = ldf[column]

        if lds.isnull().all():
            r += [column]

    return r

def levels_of_latent_variable(ldf, column):
    lds = ldf[column]
    # hasattr(pd.Series(['a'], dtype='category'), 'cat')
    if lds.dtype.name != 'category':
        raise ValueError('column name {} is not of type category!'.format(column))
    return list(lds.cat.categories)

def augment_df_with_latent_variable(ldf, latent_variable_name, levels):
    l = len(ldf)
    lds = pd.Series(np.full([l], np.nan))
    slevels = ['l{0:0>3}'.format(i) for i in range(levels)]
    lds_ct = pd.api.types.CategoricalDtype(slevels, ordered=True)
    lds = lds.astype(lds_ct)
    ldf[latent_variable_name] = lds
    return ldf


def dict2rlist(d):
    rlistfn = rpy2.robjects.r['list']
    return rlistfn(**d)


def hybrid_structure_learning_mmhc(ldf):
    rdf = rpy2.robjects.pandas2ri.py2ri(ldf)
    rmmhcfn = rpy2.robjects.r['mmhc']
    return rmmhcfn(rdf)

def hybrid_structure_learning_rxmax2_sihitonpc_tabu(ldf):
    rdf = rpy2.robjects.pandas2ri.py2ri(ldf)
    rrsmax2fn = rpy2.robjects.r['rsmax2']
    restrict_args = {'restrict.args': dict2rlist(dict(test = "x2", alpha = 0.01))}
    maximize_args = {'maximize_args': dict2rlist(dict(score = "bic", tabu = 10))}
    args = {**restrict_args, **maximize_args}
    return rrsmax2fn(rdf,
                     restrict = "si.hiton.pc",
                     maximize = "tabu",
                     **args)


class HybridScoreAndConstainedBasedNetFromDataDiscreteBayesNetwork(LearningBayesNetworkBase):

    def __init__(self, ldf=None, algorithm='mmhc'):
        super(HybridScoreAndConstainedBasedNetFromDataDiscreteBayesNetwork, self).__init__(ldf)
        self.algorithmfn = None
        if algorithm == 'mmhc':
            self.algorithmfn = lambda ldf: hybrid_structure_learning_mmhc(ldf)
        if algorithm == 'rxmax2_sihitonpc_tabu':
            self.algorithmfn = lambda ldf: hybrid_structure_learning_rxmax2_sihitonpc_tabu(ldf)

    def fit(self, X=None, y=None, seed=None):
        super(HybridScoreAndConstainedBasedNetFromDataDiscreteBayesNetwork, self).fit(X=X, y=y, seed=seed)
        self.rnet = self.algorithmfn(self.df)
        self.rnet = rnet2dagrnet(self.rnet)
        self.generate_fit()
        return self


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

def rmatrix_to_xarray(rmatrix):
    dims = rmatrix.names  # map from dim to levels
    # print(dims)
    coords = {}
    dim_names = []
    if len(dims) == 1:
        dname = 'dimension_0'
        dim_names += [dname]
        levels = list(dims[0])
        coords.update({dname: levels})
    else:
        for dname in dims.names:
            dim_names += [dname]
            levels = list(dims.rx(dname)[0])
            coords.update({dname: levels})
    values = rpy2.robjects.pandas2ri.ri2py(rmatrix)

    ar = xr.DataArray(values, dims=dim_names, coords=coords)
    return ar


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
    rlengthfn  = rpy2.robjects.r('length')

    tmp = rarcsfn(rcextendfn(rnet))
    if rlengthfn(tmp)[0] == 0:
        frm = []
        to = []
    else:
        frm = list(tmp.rx(True,'from'))
        to  = list(tmp.rx(True,'to'))

    unidirectional_edges = list(zip(frm,to))

    nodes = list(rnodesfn(rnet))

    return nodes, unidirectional_edges, []

def from_to_df(rnet):
    _, unidirectional_edges, _ = rnet2dag(rnet)
    ldf = pd.DataFrame(columns=['from', 'to'])
    for i, edge in enumerate(unidirectional_edges):
        ldf.loc[i] = edge
    return ldf

def joint_probability_distribution(bn_base, nodes):
    rquerygrain = rpy2.robjects.r('querygrain')
    rnodes = rpy2.robjects.StrVector(nodes)
    r = rquerygrain(bn_base.grain, rnodes, type = "joint")
    xr = rmatrix_to_xarray(r)
    return xr

def vstructs(bn):
    rvstructsfn  = rpy2.robjects.r('vstructs')
    rvstructs = rvstructsfn(bn) # bn.rfit
    # print(rvstructs)
    x = list(rvstructs.rx(True, 'X'))
    z = list(rvstructs.rx(True, 'Z'))
    y = list(rvstructs.rx(True, 'Y'))
    return pd.DataFrame(collections.OrderedDict(X=x,Z=z,Y=y))


def rnetscore(rnet, ldf, type='loglik'):
    rscorefn  = rpy2.robjects.r('score')
    return rscorefn(rnet, data=pydf_to_factorrdf(ldf), type=type)[0]

def score(bn, ldf, type='loglik'):
    rscorefn  = rpy2.robjects.r('score')
    return rscorefn(bn.rnet, data=pydf_to_factorrdf(ldf), type=type)[0]

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

# https://stackoverflow.com/questions/2333025/graphviz-changing-the-size-of-edge
def generate_graphviz_attributes(edge, arc_strength_info):
    info = arc_strength_info[(arc_strength_info['from'] == edge[0]) & (arc_strength_info['to'] == edge[1])]
    weight_multiplier_1 = 4.0 * 1.0
    weight_multiplier_2 = 4.0
    attributes = {
        'weight'   : '{0:.2f}'.format(float(info['rs']) * weight_multiplier_1),
        # 'len'      :  '{0:.2f}'.format(1/float(info['rs'])),
        'penwidth' : '{0:.2f}'.format(float(info['rs']) * weight_multiplier_2),
        # 'arrowsize': '{0:.2f}'.format(float(info['rs']) * weight_multiplier_2),
        'taillabel': '{0:3.1f}'.format(info['rmif'].values[0] * 100.0), # {0:0>3}
        'headlabel': '{0:3.1f}'.format(info['rmit'].values[0] * 100.0),
        # 'label'    : '{0:3.1f} -> {0:3.1f}'.format(info['rmif'].values[0] * 100.0, info['rmit'].values[0] * 100.0),
        # 'label': '{0:3.1f}'.format(info['rmif'].values[0] * 100.0),
    }
    return attributes


def dot_with_arc_strength_info(nodes, unidirectional_edges, bidirectional_edges, arc_strength_info, engine='fdp', graph_name='graph', cut_pct=0.2):
    dg_dot = graphviz.Digraph(engine=engine, comment=graph_name)

    for node in nodes:
        dg_dot.node(node)

    for _, row in arc_strength_info.iterrows():
        frm = row['from']
        to  = row['to']
        edge = (frm, to)
        gvattrs =  generate_graphviz_attributes(edge, arc_strength_info)
        dg_dot.edge(frm, to, **gvattrs)

    # for edge in unidirectional_edges:
    #     gvattrs =  generate_graphviz_attributes(edge, arc_strength_info)
    #     dg_dot.edge(edge[0], edge[1], **gvattrs)
    #
    # for edge in bidirectional_edges:
    #     gvattrs =  generate_graphviz_attributes(edge, arc_strength_info)
    #     dg_dot.edge(edge[0], edge[1], dir='none', **gvattrs)

    return dg_dot

def convert_to_categorical_series(seq):
    if not isinstance(seq, pd.Series):
        try:
            seq_iterator = iter(seq)
        except TypeError as te:
            raise te
        seq = pd.Series(seq)

    if not seq.dtype.name == 'category':
        levels = seq.unique()
        cdt = pd.api.types.CategoricalDtype(levels, ordered=False)
        seq = seq.astype(cdt)

    return seq


def contingency_matrix(labels_true, labels_pred, eps=None, sparse=False):

    if eps is not None and sparse:
        raise ValueError("Cannot set 'eps' when sparse=True")

    if isinstance(labels_true, pd.Series) and labels_true.dtype.name == 'category':
        cdt = labels_true.dtype
        classes, class_idx = cdt.categories, labels_true.cat.codes.values
    else:
        classes, class_idx = np.unique(labels_true, return_inverse=True)

    if isinstance(labels_pred, pd.Series) and labels_pred.dtype.name == 'category':
        cdt = labels_pred.dtype
        clusters, cluster_idx = cdt.categories, labels_pred.cat.codes.values
    else:
        clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]
    # Using coo_matrix to accelerate simple histogram calculation,
    # i.e. bins are consecutive integers
    # Currently, coo_matrix is faster than histogram2d for simple cases
    contingency = scipy.sparse.coo_matrix((np.ones(class_idx.shape[0]),
                                           (class_idx, cluster_idx)),
                                          shape=(n_classes, n_clusters),
                                          dtype=np.int)
    if sparse:
        contingency = contingency.tocsr()
        contingency.sum_duplicates()
    else:
        contingency = contingency.toarray()
        if eps is not None:
            # don't use += as contingency is integer
            contingency = contingency + eps
    return contingency

def relative_mutual_information_(seq1, seq2):
    u = convert_to_categorical_series(seq1)
    v = convert_to_categorical_series(seq2)
    is_null_u = pd.isnull(u)
    is_null_v = pd.isnull(v)
    is_null_either = is_null_u | is_null_v
    u_ = u[~is_null_either]
    v_ = v[~is_null_either]

    classes  = np.unique(u_)
    clusters = np.unique(v_)
    # Special limit cases: no clustering since the data is not split.
    # This is a perfect match hence return 1.0.
    if (classes.shape[0] == clusters.shape[0] == 1 or classes.shape[0] == clusters.shape[0] == 0):
        return 1.0

    eps = np.finfo(np.float64).eps
    eu_ = np.max([sklearn.metrics.cluster.supervised.entropy(u_), eps])
    ev_ = np.max([sklearn.metrics.cluster.supervised.entropy(v_), eps])
    mi_ = sklearn.metrics.mutual_info_score(u_, v_)
    # print('{},{},{}'.format(eu_,ev_,mi_))
    return mi_, eu_, ev_

def relative_mutual_information(seq1, seq2):
    mi_, eu_, ev_ = relative_mutual_information_(seq1, seq2)
    rmu_ = np.max([mi_ / eu_, mi_ / ev_])
    return rmu_

def relative_mutual_information_distance(seq1, seq2):
    return 1.0 - relative_mutual_information(seq1, seq2)


def discrete_and_continuous_variables_with_and_without_nulls(ldf, cutoff=20):
    discrete_non_null = []
    discrete_with_null = []
    continuous_non_null = []
    continuous_with_null = []
    levels_map = dict()
    for col in ldf.columns:
        uq = ldf[col].unique()
        number_type = False
        if all([np.issubdtype(type(level), np.number) for level in uq]):
            number_type = True

        if len(uq) > cutoff:
            if pd.isnull(uq).any():
                continuous_with_null += [col]
            else:
                continuous_non_null += [col]
        else:
            if pd.isnull(uq).any():
                discrete_with_null += [col]
                if number_type:
                    levels_map[col] = sorted(list(set(uq) - set([np.nan])))
                else:
                    levels_map[col] = set(uq)  - set([np.nan])
            else:
                discrete_non_null += [col]
                if number_type:
                    levels_map[col] = sorted(list(uq))
                else:
                    levels_map[col] = list(uq)

    return discrete_non_null, discrete_with_null, continuous_non_null, continuous_with_null, levels_map

def convert_interval_index_categories_to_string_categories(ldf, inplace=True):
    if not inplace:
        ldf = ldf.copy()

    for column in ldf.columns:
        if ldf[column].dtype.name != 'category':
            continue
        levels = ['' + str(cat) for cat in ldf[column].dtype.categories]
        cdt = pd.api.types.CategoricalDtype(levels, ordered=True)
        ldf.loc[:,column] = ldf[column].apply(lambda x: str(x)).astype(cdt)

    return ldf

def entropy(pi):
    pi = pi[pi > 0]
    pi_sum = np.sum(pi)
    # log(a / b) should be calculated as log(a) - log(b) for
    # possible loss of precision
    return -np.sum((pi / pi_sum) * (np.log(pi) - math.log(pi_sum)))

def mutual_information_from_joint_distribution(xr_distribution):
    contingency = xr_distribution.values
    nzx, nzy = np.nonzero(contingency)
    nz_val = contingency[nzx, nzy]

    pi = np.ravel(contingency.sum(axis=1))
    pj = np.ravel(contingency.sum(axis=0))
    contingency_nm = nz_val
    log_contingency_nm = np.log(contingency_nm)
    outer = pi.take(nzx) * pj.take(nzy)
    log_outer = -np.log(outer)  # + math.log(pi.sum()) + math.log(pj.sum())
    mi = (contingency_nm * log_contingency_nm + contingency_nm * log_outer)
    return mi.sum(), entropy(pi), entropy(pj)

def from_to_mutual_information_infos(bn_base, frm_to):
    jpd_xr = joint_probability_distribution(bn_base, frm_to)

    # in principle this information could be also calculated via:
    #   relative_mutual_information_(ddf_without_null_values[frm], ddf_without_null_values[to])
    # but this would only be valid if you have data without null values.
    # If you use parametric or structural EM you will retrieve probability distributions, that may be different than the estimate from the data without null values
    mi, ei, ej = mutual_information_from_joint_distribution(jpd_xr)
    if np.isclose(ei,0.0) and np.isclose(ej,0.0):
        rmi_i = rmi_j = 1.0
    elif np.isclose(ei,0.0):
        rmi_i = 0.0
        rmi_j = mi / ej
    elif np.isclose(ej,0.0):
        rmi_i = mi / ei
        rmi_j = 0.0
    else:
        rmi_i = mi / ei
        rmi_j = mi / ej

    nmi     = np.sqrt(rmi_i * rmi_j) # normalized mututal information : the geometric mean
    max_rmi = np.max([rmi_i, rmi_j])

    return mi, ei, ej, rmi_i , rmi_j, nmi, max_rmi

def bn_arcs_mutual_information_infos(bn_base, short_column_names_p=False):
    if short_column_names_p:
        column_names = ['frm', 'to', 'mi', 'ef', 'et', 'rmif', 'rmit', 'nmi', 'mmi']
    else:
        column_names = ['from', 'to', 'mutual_information', 'entropy_from', 'entropy_to', 'relative_mutual_information_from', 'relative_mutual_information_to',
                        'normalized_mutual_information', 'max_mutual_information']
    ldf = pd.DataFrame(columns=column_names)
    frm_to_df = from_to_df(bn_base.rnet)
    for index, row in frm_to_df.iterrows():
        lrow = list(row)
        frm, to = lrow
        ldf.loc[index] = [frm, to] + list(from_to_mutual_information_infos(bn_base, lrow))
    return ldf

def bn_arcs_strengths(bn_base, ldf=None, criterion='loglik'):
    if ldf is None:
        ldf = bn_base.df
    if ldf is None:
        raise RuntimeError('You did neither provide an input value for ldf nor does the network contain a df value!')

    rarcstrengthfn = rpy2.robjects.r('arc.strength')
    rdf = rarcstrengthfn(bn_base.rnet, ldf, criterion=criterion)
    return rpy2.robjects.pandas2ri.ri2py(rdf).sort_values(['strength'], ascending=True)

# https://stackoverflow.com/questions/30510562/get-mapping-of-categorical-variables-in-pandas
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Categorical.from_codes.html
def from_codes_to_category(codes, cat_dtype):
    return pd.Categorical.from_codes(codes, cat_dtype.categories, ordered=cat_dtype.ordered)
