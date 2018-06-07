
import numpy as np, pandas as pd, pytest
import pybnl.bn

def test_rain_sprinkler_grasswet():
    df_rain_pm = pd.DataFrame(
        [
            ['T', 0.2],
            ['F', 0.8],
        ], columns=['rain', 'p']
    )
    df_sprinkler_cpm = pd.DataFrame(
        [
            ['F', 'F', 0.6],
            ['F', 'T', 0.4],
            ['T', 'F', 0.99],
            ['T', 'T', 0.01],
        ], columns=['rain', 'sprinkler', 'p']
    )
    df_grasswet_cpm = pd.DataFrame(
        [
            ['F', 'F', 'F', 1.0],
            ['F', 'F', 'T', 0.0],
            ['F', 'T', 'F', 0.2],
            ['F', 'T', 'T', 0.8],

            ['T', 'F', 'F', 0.1],
            ['T', 'F', 'T', 0.9],
            ['T', 'T', 'F', 0.01],
            ['T', 'T', 'T', 0.99],
        ], columns=['sprinkler', 'rain', 'grasswet', 'p']
    )

    dbn_rain_springkler_grasswet = pybnl.bn.CustomDiscreteBayesNetwork([df_rain_pm, df_sprinkler_cpm, df_grasswet_cpm])

    evidence = dict(grasswet='T', sprinkler='F')
    nodes_to_query = ['rain']
    result = dbn_rain_springkler_grasswet.exact_query(evidence, nodes_to_query, only_python_result=True)
    assert result['rain']['T'] == pytest.approx(1.0, 0.0000001)


