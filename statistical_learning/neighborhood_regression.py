import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import networkx as nx
import logging

class NeighborhoodRegression:
    """
    Class that implements (graph) neighborhood inference by applying linear or logistic regression regression.

    """

    def __init__(self, data=None):
        self.__name__ = 'neighborhood_inference'
        self.signals_data = data
        #self.node_id_column = node_id_column
        #self.node_signal_columns = node_signal_columns

        #going to be set whenever infer_graph is called
        self.edges_df = None
        self._G_inferred = None
        self.mean_score = None #the average score as returned by sklearn to get a feeling for how "accurate" the estimation is.

    def _infer_binary_signals_graph(self, C, exclude_same_category_col=None, only_same_category_col=None, **kwargs):
        """
        Helper Method to infer graph applying logistic regression as described in P. Ravikumar et al: "High-dimensional Ising model selection using l1-regularized logistic regression" (https://arxiv.org/pdf/1010.0311.pdf).

        :param C: float. regularization parameter: the smaller C, the sparser the coefficients vector. typically C<<1 for learning sparse representations.
        :return: networkx MultiGraph.
        """

        all_edges_dfs = []
        model_scores = []
        #for each node (i.e. row in the dataframe), we want to learn its neighborhood. therefore, we aim to describe each node's signal as a linear combination of other nodes' signals
        for i in range(self.signals_data.shape[0]):

            y = self.signals_data.iloc[i].values #the target we want to predict using the other signals
            # if only one class is present, sklearn will complain. in such a case, we add a fake class to the target and train set (below)
            fake_val = None
            if np.unique(y).shape[0] < 2:
                fake_val = np.unique(y)[0] + 1
                y = np.append(y, fake_val)

            train_inds = [j for j in range(len(self.signals_data)) if
                          not i == j]  # all indices to use as independent variables (train data in ML slang)

            # construct train_df with all independet variables and add fake observation if neccessary
            X = self.signals_data.iloc[train_inds].T.values
            if not fake_val is None:
                X = np.append(X, [[fake_val] * X.shape[1]], axis=0)

            model = LogisticRegression(penalty='l1',  # l1 regularized as in paper
                                       C=C,
                                       # controls regularization, the smaller C, the sparser the coefficients vector
                                       solver='liblinear',  # because of l1
                                       fit_intercept=False)
            model.fit(X=X, y=y)
            model_scores.append(model.score(X, y))
            # identify non zero coefficients. those define the edges in the graph
            non_zero_connects = model.coef_.nonzero()[1]

            # map back to indices in original data
            orig_data_inds = [train_inds[ind] for ind in non_zero_connects]

            #build edges_df for node. later going to be concat. to one big edges dataframe
            edges_df = pd.DataFrame({'source': i,
                                     'target': orig_data_inds,
                                     'weight': model.coef_[0][non_zero_connects]})
            all_edges_dfs.append(edges_df)
        #build graph
        all_edges_df = pd.concat(all_edges_dfs).reset_index(drop=True)
        graph = nx.from_pandas_edgelist(all_edges_df,
                                        source='source',
                                        target='target',
                                        edge_attr='weight',
                                        create_using=nx.MultiGraph)

        self.edges_df = all_edges_df
        self.mean_score = np.mean(model_scores)
        self._G_inferred = graph

    def get_graph(self):
        """

        :return: networkx graph. Inferred graph.
        """
        if self._G_inferred is None:
            logging.warning("No graph has been inferred yet.")

        return self._G_inferred

    def infer_graph(self, data, signals='binary', C=0.1, **kwargs):
        """
        CURRENTLY ONLY BINARY SIGNALS IMPLEMENTED.
        Infers a graph from <data> using linear (continuous) or logistic (binary) regression. ADD EXPLANATION.

        :param data: pandas DataFrame with signals data
        :param signals: str. Must be one of 'binary'. Specifies whether the signals shall be treated as discrete or continous values.
        :param C: float. Regularization parameter. Specifies how sparse the coefficients (and graph) is in the end. The smaller C the fewer connections each node has.
        :param **kwargs:
        :return: networkx graph.
        """

        self.signals_data = data

        #check whether input is valid

        #for each node, learn representation of neighboring nodes using lasso regression
        if signals=='binary':
            return self._infer_binary_signals_graph(C, **kwargs)
        else:
            raise NotImplementedError
