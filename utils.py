import numpy as np
import networkx as nx
import pandas as pd
import scipy.stats as st

from node2vec import Node2Vec
from typing import Tuple


def get_mean_with_ci(a: np.ndarray, alpha: float = 0.95) -> Tuple[float, Tuple[float, float]]:
    """
    Given an array a, function computes the mean of a with confidence intervals

    :param a: input array
    :param alpha: confidence level

    :preturn a tuple with the mean of a and confidence intervals
    """
    ci_min, ci_max = st.t.interval(alpha, len(a)-1, loc=np.mean(a), scale=st.sem(a))

    return (a.mean(), (ci_min, ci_max))


def adjacency_matrix_to_train_set(g: nx.Graph, depth: int = 3) -> pd.DataFrame:
    """
    Transforms adjacency matrix of a graph into a training set for ML model

    :param g: input graph
    :param depth: max length of paths considered when generating training set

    :return dataframe with nodes, their embeddings, and their similarity
    """

    alpha = 10
    result = []

    model = Node2Vec(g).fit()

    A = nx.adjacency_matrix(g).todense()
    AA = A

    for i in range(depth):

        for (x,y), val in np.ndenumerate(AA):
            result.append((x, y, val*(1/alpha**(i))))

        AA = AA @ A

    df = pd.DataFrame(np.array(result), columns=['x','y','val'])

    dfg = df.groupby(['x','y'], as_index=False).sum()
    dfg['emb_x'] = dfg['x'].apply(lambda x: model.wv[str(int(x))])
    dfg['emb_y'] = dfg['y'].apply(lambda y: model.wv[str(int(y))])

    return dfg


if __name__ == '__main__':

    pass