'''
This script creates a graph of the networks and does the embedding of the network.
usage- python get_embedding.py -s second_layer
'''

import os
import argparse as ap
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
import config

class EmbeddData():
    """class that represents embedding of the network

    :return: an embedding class
    :rtype: None
    """
    def __init__(self, second_layer_path, embedded_path):
        """funtion to construct all necessary attributes for a embedding object

        :param second_layer_path: path to the second layer relations
        :type second_layer_path: str
        :param embedded_path: path to save the embeddings
        :type embedded_path: str
        """
        self.data_path = second_layer_path
        self.graph = self.get_graph()
        self.out_path = embedded_path

    def get_graph(self):
        """function to convert dataframe to graph

        :return: a graph of the network
        :rtype: nxgraph
        """
        # read file
        data = pd.read_csv(self.data_path, sep='\t')
        return nx.from_pandas_edgelist(
            data,
            source='subject',
            target='object',
            edge_attr='local_mi',
            edge_key='local_mi',
            create_using=None,
        )

    def get_embedding(self):
        """function to perform embedding from the graph and save it
        """
        node2vec = Node2Vec(self.graph,
                            dimensions=100,
                            walk_length=80,
                            num_walks=10,
                            workers=4,
                            weight_key='local_mi',
                            p=0.5,
                            q=2)
        # fit model
        embedding = node2vec.fit()
        # embedding_filepath = self.out_path
        embedding.wv.save_word2vec_format(self.out_path)

def main():
    """function to catch argparser arguments and run script
    """
    argparser = ap.ArgumentParser(
                                description= "Script that performs embedding")
    argparser.add_argument("--SECOND_LAYER", "-s", action="store", type=str,
                             help="Name ofsecond layer file")
    argparser.add_argument("--RESULT_DIRECTORY", "-d", action="store", type=str,
                             help="Path to save result")
    parsed = argparser.parse_args()
    second_layer_path = parsed.SECOND_LAYER
    result_dir = config.RESULTS_DIRECTORY
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
        print("Result directory created")
    EmbeddData(result_dir+'/'+second_layer_path, f'{result_dir}/embedding.emb').get_embedding()

if __name__ == '__main__':
    main()
