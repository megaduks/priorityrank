import numpy as np
import networkx as nx
import pandas as pd
import scipy.stats as st

from node2vec import Node2Vec
from typing import Tuple, Dict, List
from collections import namedtuple

from scipy.stats import ks_2samp
from numpy.linalg import norm

Mean_with_CI = namedtuple('Mean_with_CI', 'mean ci_min ci_max')


def cosine_similarity(a: np.ndarray, b: np.ndarray):
    """
    Computes cosine similarity between two NumPy arrays

    :param a: first input array
    :param b: second input array

    :returns cosine distance between a and b
    """
    return np.dot(a, b) / (norm(a) * norm(b))


def get_mean_with_ci(a: np.ndarray, alpha: float = 0.95) -> Mean_with_CI:
    """
    Given an array a, function computes the mean of a with confidence intervals

    :param a: input array
    :param alpha: confidence level

    :preturn a tuple with the mean of a and confidence intervals
    """
    ci_min, ci_max = st.t.interval(alpha, len(a)-1, loc=np.mean(a), scale=st.sem(a))

    return Mean_with_CI(a.mean(), ci_min, ci_max)


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


def compare_graphs(g: nx.Graph, graphs: List[nx.Graph]) -> Dict:
    """
    Compares a given empirical graph to a set of artificially generated graphs

    :param g: input graph
    :param graphs: list of graphs
    :returns a dictionary with the results of comparisons
    """
    result = {}

    g_degree_distribution = [ d for n,d in nx.degree(g)]
    degree_distribution_pvals = [
        ks_2samp(g_degree_distribution, [d for n,d in nx.degree(h)]).pvalue
        for h in graphs
    ]

    g_betweenness_distribution = list(nx.betweenness_centrality(g).values())
    betweenness_distribution_pvals = [
        ks_2samp(g_betweenness_distribution, list(nx.betweenness_centrality(h).values())).pvalue
        for h in graphs
    ]

    g_closeness_distribution = list(nx.closeness_centrality(g).values())
    closeness_distribution_pvals = [
        ks_2samp(g_closeness_distribution, list(nx.closeness_centrality(h).values())).pvalue
        for h in graphs
    ]

    g_pagerank_distribution = list(nx.pagerank(g).values())
    pagerank_distribution_pvals = [
        ks_2samp(g_pagerank_distribution, list(nx.pagerank(h).values())).pvalue
        for h in graphs
    ]

    result['degree_distribution_pval'] = get_mean_with_ci(np.array(degree_distribution_pvals))
    result['betweenness_distribution_pval'] = get_mean_with_ci(np.array(betweenness_distribution_pvals))
    result['closeness_distribution_pval'] = get_mean_with_ci(np.array(closeness_distribution_pvals))
    result['pagerank_distribution_pval'] = get_mean_with_ci(np.array(pagerank_distribution_pvals))

    return result