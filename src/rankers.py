import numpy as np
import networkx as nx

from node2vec import Node2Vec
from typing import List

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

from src.utils import cosine_similarity, adjacency_matrix_to_train_set


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


class DegreeRanker(Ranker):
    """
    Implements ranking based on the preferential attachment principle
    """
    def __init__(self, g: nx.Graph):
        super().__init__(g)
        self.ranking = sorted(g.nodes, key=lambda x: g.degree[x], reverse=True)

    def get_ranking(self, n: int) -> List[int]:

        return self.ranking


class MLRanker(Ranker):
    """
    Implements ML approach to node ranking generation
    """
    def __init__(self, g):
        super().__init__(g)

        df = adjacency_matrix_to_train_set(g, depth=3)
        X = df[['emb_x','emb_y']].values
        y = df[['val']]

        _X = [ np.concatenate([a,b]) for a,b in X ]
        X_train = np.array(_X)

        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=256))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='relu'))

        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)

        model.compile(loss='mean_squared_error',
                      optimizer=sgd,
                      metrics=['mean_squared_error'])

        model.fit(X_train, y, epochs=500, batch_size=8)

        self.df = df
        self.model = model

    def get_ranking(self, n: int) -> List[int]:

        n_idx = self.df.x == n
        X = self.df[n_idx][['emb_x','emb_y']].values
        y = self.df[n_idx]['y']
        _X = [ np.concatenate([a,b]) for a,b in X ]
        X_test = np.array(_X)

        y_pred = self.model.predict(X_test).tolist()

        ranking = [ int(n) for n,v in sorted(zip(y, y_pred), key=lambda x: x[1], reverse=True)]

        return ranking
