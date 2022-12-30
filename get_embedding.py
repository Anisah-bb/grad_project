'''
usage
get_embedding.py -d /homes/fabadmus/Internship/RAtest
'''

import os
import argparse as ap
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from gensim.models import Word2Vec

class EmbeddData():
    
    ''' class to create graph from data and do embedding
    '''
    def __init__(self, second_layer_path, embedded_path):
        self.data_path = second_layer_path
        self.G = self.get_graph()
        self.out_path = embedded_path
        
        
    def get_graph(self):
        '''
        function to convert dataframe to graph
        '''
        # read file
        df = pd.read_csv(self.data_path)
        # remove diseases
        df = df[df.subject.str.startswith('TWDIS') == False]
        # convert to graph
        self.G = nx.from_pandas_edgelist(df, source='subject',
                                target='object', edge_attr='local_mi', edge_key='local_mi', create_using= None)
        return self.G

    def get_embedding(self):
        '''
        function to perform embedding from the graph
        '''
        node2vec = Node2Vec(self.G, dimensions=16, walk_length=80, num_walks=10, workers=4, weight_key='local_mi', p=0.5, q=2)

        # fit model
        embedding = node2vec.fit()
        EMBEDDING_FILEPATH = self.out_path
        # Save embeddings for later use
        # embedding.wv.save_word2vec_format(EMBEDDING_FILEPATH)
        # embedding.save_to_dir(EMBEDDING_FILEPATH)
        # Save model for later use
        embedding.save(EMBEDDING_FILEPATH)
    
def main():
    argparser = ap.ArgumentParser(
                                description= "Script that performs embedding")
    argparser.add_argument("--RESULT_DIRECTORY", "-d", action="store", type=str,
                             help="Path to save result")
    parsed = argparser.parse_args()
    result_dir = parsed.RESULT_DIRECTORY
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    print("Result directory created")
    EmbeddData(f'{result_dir}/second_layer', f'{result_dir}/embedding').get_embedding()
    
    # EmbeddData('/homes/fabadmus/Internship/second_layer', '/homes/fabadmus/Internship/embedding').get_embedding()
    #embedding.get_embedding()
    # G = get_graph('/homes/fabadmus/Internship/second_layer')
    # get_embedding(G, 'embedding')


if __name__ == '__main__':
    main()