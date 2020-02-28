import networkx as nx
import numpy as np
import os, dotenv
import neptune

from typing import List
from collections import Counter

from rankers import RandomRanker, EmbeddingRanker, DegreeRanker, MLRanker
from utils import compare_graphs

from tqdm import tqdm


def get_rank_probabilities(n: int, alpha: float = 1.5) -> List[float]:
    """
    Generates the list of probabilities for a given length of ranking

    :param n: length of the ranking
    :param alpha: attenuation parameter for changing probabilities of positions in the ranking

    :returns a list of diminishing probabilities
    """
    ranks = [1 / i**alpha for i in range(1, n + 1)]

    return [r / sum(ranks) for r in ranks]


class PriorityRank:
    """
    Implements the priority attachment mechanism for generating artificial networks
    """
    def __init__(self, graph: nx.Graph, ranker: object):
        self.graph = graph
        self.ranker = ranker(graph)

    def generate(self):
        """
        Generates an artificial network based on the empirical input network

        :returns artificial network generated by the priority attachment mechanism
        """

        g = nx.Graph()
        g.add_nodes_from(self.graph.nodes)

        num_nodes = g.number_of_nodes()

        degree_sequence = sorted([d for n, d in self.graph.degree()])
        degree_count = Counter(degree_sequence)
        deg, cnt = zip(*degree_count.items())

        degree_probs = [c / sum(cnt) for c in cnt]

        for i in range(num_nodes):
            num_edges = np.random.choice(a=deg, p=degree_probs) - g.degree[i]

            if num_edges > 0:
                ranking = self.ranker.get_ranking(i)
                probs = get_rank_probabilities(len(ranking), alpha=2.5)
                target_nodes = np.random.choice(a=ranking, p=probs, size=num_edges, replace=False)

                for j in target_nodes:
                    g.add_edge(i, j)

        return g


if __name__ == '__main__':

    project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
    dotenv_path = os.path.join(project_dir, '.env')
    dotenv.load_dotenv(dotenv_path)

    PARAMS = {
        'n_experiments': 50,
        'graph_size': [50, 100, 250]
    }

    neptune.init(dotenv['NEPTUNE_PROJECT'], api_token=dotenv['NEPTUNE_API_KEY'])

    with neptune.create_experiment(name='priorityrank-barabasi', params=PARAMS):

        for g_size in PARAMS['graph_size']:

            graphs = []

            g = nx.barabasi_albert_graph(n=g_size, m=2)

            pr = PriorityRank(g, MLRanker)

            for i in tqdm(range(PARAMS['n_experiments'])):
                graphs.append(pr.generate())

            print()

            for k,v in compare_graphs(g, graphs).items():
                print(k, v)
                neptune.log_metric(k + 'mean', v.mean)
                neptune.log_metric(k + 'ci_min', v.ci_min)
                neptune.log_metric(k + 'ci_max', v.ci_max)
