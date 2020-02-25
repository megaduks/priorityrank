import numpy as np
import networkx as nx
import torch

from node2vec import Node2Vec
from typing import List
from torchnca import NCA

from utils import cosine_similarity


class Ranker:
    """
    Implements an abstract concept of node ranking, must be instantiated
    """
    def __init__(self, g: nx.Graph):
        self.graph = g

    def get_ranking(self, n: int) -> List[int]:
        pass


class RandomRanker(Ranker):
    """
    Implements random ranking
    """
    def __init__(self, g: nx.Graph):
        super().__init__(g)
        self.ranking = [ n for n in g.nodes ]

    def get_ranking(self, n: int) -> List[int]:
        ranking = [ i for i in self.ranking if i != n ]
        np.random.shuffle(ranking)

        return ranking


class EmbeddingRanker(Ranker):
    """
    Implements ranking based on distances in the embedding space
    """
    def __init__(self, g: nx.Graph):
        super().__init__(g)
        self.model = Node2Vec(self.graph).fit()
        self.v = { n: self.model.wv[str(n)] for n in self.graph.nodes }

    def get_ranking(self, n: int) -> List[int]:

        def similarity(i):
            return cosine_similarity(self.v[n], self.v[i])

        ranking = sorted(
            [i for i in self.graph.nodes if i != n],
            key=similarity,
            reverse=True)

        return ranking
