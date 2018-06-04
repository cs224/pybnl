
import locale
locale.setlocale(locale.LC_ALL, 'C')

import numpy as np,xarray as xr, pandas as pd

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

class DiscreteTabularCPM():

    def __init__(self, ldf, transform=True):
        self.transform = transform
        self.ldf = ldf
        self.cpt_to_r()
        self.cpt_to_model_string()

    def extract_xarray_and_nparray_no_transform(self, xrin):
        self.lcpm_xr = xrin
        self.nparray = np.array(xrin)

    def extract_xarray_and_nparray_transform0(self, xrin):
        tmp0 = xrin
        old_order = tmp0.dims
        new_order = old_order[-1:] + old_order[:-1]
        # tmp = np.reshape(tmp0.values.reshape(-1),tmp0.values.shape,order='F')
        # lcpm_xr = xr.DataArray(tmp,dims = tmp0.dims, coords= tmp0.coords)
        lcpm_xr = tmp0.transpose(*new_order)  #lcpm_xds[var_name]

        self.lcpm_xr = lcpm_xr

        # the array itself:
        nparray0 = np.array(lcpm_xr)
        # old_order = list(range(len(nparray0.shape)))
        # new_order = old_order[-1:] + old_order[:-1]
        # nparray1 = nparray0.transpose(new_order)

        nparray = np.reshape(nparray0.reshape(-1),nparray0.shape,order='F')
        self.nparray = nparray

    def extract_xarray_and_nparray_transform1(self, xrin):
        tmp0 = xrin
        old_order = tmp0.dims
        new_order = old_order[-1:] + old_order[:-1]
        lcpm_xr = tmp0.transpose(*new_order)  #lcpm_xds[var_name]

        self.lcpm_xr = lcpm_xr
        self.nparray = np.array(lcpm_xr)

    def cpt_to_r(self):
        lcpm_ds = self.ldf.groupby(list(self.ldf.columns[:-1])).max()  # .to_xarray().p.values
        lcpm_xds = lcpm_ds.to_xarray()
        var_name = list(lcpm_xds.data_vars.keys())[0]
        if self.transform:
            self.extract_xarray_and_nparray_transform1(lcpm_xds[var_name])
        else:
            self.extract_xarray_and_nparray_no_transform(lcpm_xds[var_name])

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
        parent_model_string = ':'.join(parent_vars)
        parent_child_string = '|'.join([target_var, parent_model_string])
        if len(parent_vars) == 0:
            parent_child_string = target_var
        self.model_string = '[' + parent_child_string + ']'

    def to_cpm_table(self):
        tmp = self.ldf.pivot_table(index=self.parents,columns=[self.target],values=[self.p_var_name]).reset_index()
        #tmp.columns = [self.target]
        return tmp


class DiscreteBayesNetwork():

    def __init__(self, ldf_list):
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




