
# The test case follows exercise 5.1 from "Doing Bayesian Data Analysis" by John K. Kruschke:
# https://sites.google.com/site/doingbayesiandataanalysis/exercises

import numpy as np, pandas as pd, pytest
import pybnl.bn

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# pd.set_option('display.float_format', lambda x: '%.2f' % x)
np.set_printoptions(edgeitems=10)
np.set_printoptions(suppress=True)
np.core.arrayprint._line_width = 180

import rpy2, rpy2.rinterface, rpy2.robjects, rpy2.robjects.packages, rpy2.robjects.lib, rpy2.robjects.lib.grid, \
    rpy2.robjects.lib.ggplot2, rpy2.robjects.pandas2ri, rpy2.interactive.process_revents, \
    rpy2.interactive, rpy2.robjects.lib.grdevices, rpy2.rlike.container
# rpy2.interactive.process_revents.start()
rpy2.robjects.pandas2ri.activate()


p_disease_present = 0.001
p_test_positive_given_disease_present = 0.99
p_test_positive_given_disease_absent = 0.05

def create_cpts(nr_tests=1):
    df_prior = pd.DataFrame(
        [
            ['T', p_disease_present],
            ['F', 1 - p_disease_present],
        ], columns=['disease_present', 'p']
    )

    df_disease_test_cpms = []
    for i in range(nr_tests):
        node_name = 'test_positive_{}'.format(i)
        df_disease_test_cpm = pd.DataFrame(
            [
                ['F', 'F', 1 - p_test_positive_given_disease_absent],
                ['F', 'T', p_test_positive_given_disease_absent],
                ['T', 'F', 1 - p_test_positive_given_disease_present],
                ['T', 'T', p_test_positive_given_disease_present],
            ], columns=['disease_present', node_name, 'p']
        )
        df_disease_test_cpms += [df_disease_test_cpm]

    return [df_prior] + df_disease_test_cpms

def calculate_posterior_via_xarray(input=['T', 'F']):
    test_count = len(input)
    cpts = [pybnl.bn.DiscreteTabularCPM(ldf) for ldf in create_cpts(nr_tests=test_count)]
    test_vars = ['test_positive_{}'.format(i) for i in range(test_count)]
    input_dict = dict(zip(test_vars, input))

    r = cpts[0].lcpm_xr.copy()
    for i in range(test_count):
        r = r * cpts[i + 1].lcpm_xr

    conditional_probability_1 = r.loc[input_dict]
    conditional_probability = conditional_probability_1 / conditional_probability_1.sum()
    return conditional_probability.values

def calculate_posterior_via_bayes_rule(input=['T', 'F']):
    prior, cpt_disease_present = [pybnl.bn.DiscreteTabularCPM(ldf) for ldf in create_cpts(nr_tests=1)]
    prior = prior.lcpm_xr
    cpt_disease_present = cpt_disease_present.lcpm_xr

    posterior = prior.values.copy()
    for x in input:
        selected_row = cpt_disease_present.loc[dict(test_positive_0=x)].values
        posterior = selected_row * posterior
        posterior = posterior / float(posterior.sum())

    return posterior


# calculate_posterior_via_xarray(input=['T', 'F'])
# calculate_posterior_via_bayes_rule(input=['T', 'F'])

def generate_test_sequences(max_length=10):
    r = []
    for l in range(max_length):
        rs = np.random.RandomState(l)
        r += [rs.choice(['F','T'],l)]
    return r

# generate_test_sequences(max_length=10)

def test_exact_inference_in_disease_tests_network():
    tss = generate_test_sequences(max_length=10)
    nodes_to_query = ['disease_present']
    for ts in tss:
        f_t_result = calculate_posterior_via_bayes_rule(input=ts)
        test_count = len(ts)
        dbn_disease_present = pybnl.bn.CustomDiscreteBayesNetwork(create_cpts(nr_tests=test_count))
        test_vars = ['test_positive_{}'.format(i) for i in range(test_count)]
        evidence = dict(zip(test_vars, ts))
        dbn_result = dbn_disease_present.exact_query(evidence, nodes_to_query, only_python_result=True)
        dbn_result_disease_present_1 = dbn_result['disease_present']
        dbn_result_disease_present = np.array([dbn_result_disease_present_1['F'], dbn_result_disease_present_1['T']])

        # print(dbn_result_disease_present)
        # print(f_t_result)
        np.testing.assert_array_almost_equal(dbn_result_disease_present, f_t_result, decimal=2)

# test_exact_inference_in_disease_tests_network()

def test_given_net_when_evidence_var_does_not_exist_then_raise_exception():
    dbn_disease_present = pybnl.bn.CustomDiscreteBayesNetwork(create_cpts(nr_tests=1))
    evidence = {'test_positive_':'T'} # no number, e.g. this evidence node does not exist
    nodes_to_query = ['disease_present']
    with pytest.raises(Exception) as e_info:
        dbn_result = dbn_disease_present.exact_query(evidence, nodes_to_query, only_python_result=True)

test_given_net_when_evidence_var_does_not_exist_then_raise_exception()

def test_given_net_when_query_var_does_not_exist_then_raise_exception():
    dbn_disease_present = pybnl.bn.CustomDiscreteBayesNetwork(create_cpts(nr_tests=1))
    evidence = {'test_positive_0':'T'}
    nodes_to_query = ['disease_present_'] # erroneous ending in underscore
    with pytest.raises(Exception) as e_info:
        dbn_result = dbn_disease_present.exact_query(evidence, nodes_to_query, only_python_result=True)

test_given_net_when_query_var_does_not_exist_then_raise_exception()

def test_given_net_when_evidence_var_value_does_not_exist_then_raise_exception():
    dbn_disease_present = pybnl.bn.CustomDiscreteBayesNetwork(create_cpts(nr_tests=1))
    evidence = {'test_positive_0':'T_'} # the var name exists, but the value 'T_' does not exist
    nodes_to_query = ['disease_present']
    with pytest.raises(Exception) as e_info:
        dbn_result = dbn_disease_present.exact_query(evidence, nodes_to_query, only_python_result=True)

test_given_net_when_evidence_var_value_does_not_exist_then_raise_exception()

#
# dbn_disease_present_tmp = pybnl.bn.CustomDiscreteBayesNetwork(create_cpts(nr_tests=1))
#
# print(dbn_disease_present_tmp.rfit.rx('test_positive_0')[0].names)
# list(dbn_disease_present_tmp.rfit.rx('test_positive_0')[0].rx('prob')[0].names.rx('test_positive_0')[0])
# rpy2.robjects.pandas2ri.ri2py(dbn_disease_present_tmp.rfit.rx('test_positive_0')[0].rx('prob')[0].names.rx('test_positive_0')[0])
# list()