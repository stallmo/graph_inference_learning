import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import networkx as nx

class NeighborhoodRegression:
    """
    Class that implements the lasso regression technique for graph inference learning.

    :param data: pandas DataFrame
    :param node_id_column: string. Name of a column in <data>. Defines which column to use as node identifier.
    :param node_signal_columns: list of str. Each str is the name of a column in <data>. Defines the graph signals on the nodes.
    """
    def __init__(self, data, node_id_column, node_signal_columns):
        self.__name__ = 'neighborhood_inference'
        self.data = data
        self.node_id_column = node_id_column
        self.node_signal_columns = node_signal_columns

        #going to be set whenever infer_graph is called
        self.edges_df = None

    def _infer_binary_signals_graph(self, C, **kwargs):
        """
        Method to infer graph applying logistic regression as described in <PAPER>.

        :param kwargs:
        :return: networkx MultiGraph.
        """

        all_edges_dfs = []
        model_scores = []
        #for each node (i.e. row in the dataframe), we want to learn its neighborhood. therefore, we aim to describe each node's signal as a linear combination of other nodes' signals
        for i in range(self.data.shape[0]):

            y = self.data[self.node_signal_columns].iloc[i].values #the target we want to predict using the other signals
            # if only one class is present, sklearn will complain. in such a case, we add a fake class to the target and train set (below)
            fake_val = None
            if np.unique(y).shape[0] < 2:
                fake_val = np.unique(y)[0] + 1
                y = np.append(y, fake_val)

            train_inds = [j for j in range(len(self.data)) if
                          not i == j]  # all indices to use as independent variables (train data in ML slang)

            # construct train_df with all independet variables and add fake observation if neccessary
            X = self.data[self.node_signal_columns].iloc[train_inds].T.values
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
        return graph

    def infer_graph(self, signals='binary', C=0.1, **kwargs):
        """
        CURRENTLY ONLY BINARY SIGNALS IMPLEMENTED.
        Infers a graph from <data> using linear (continuous) or logistic (binary) regression. ADD EXPLANATION.

        :param signals: str. Must be one of 'binary'. Specifies whether the signals shall be treated as discrete or continous values.
        :param C: float. Regularization parameter. Specifies how sparse the coefficients (and graph) is in the end. The smaller C the fewer connections each node has.
        :param **kwargs:
        :return: networkx graph.
        """

        #check whether input is valid

        #for each node, learn representation of neighboring nodes using lasso regression
        if signals=='binary':
            return self._infer_binary_signals_graph(C, **kwargs)
        else:
            raise NotImplementedError
