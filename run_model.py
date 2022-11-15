# import libraries
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import requests
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from get_embedding import get_graph

G = get_graph('/homes/fabadmus/Internship/second_layer2')
embeddings = Word2Vec.load('embeddings2')
# Convert embeddings to dataframe
emb_df = (pd.DataFrame([embeddings.wv.get_vector(str(n))
                        for n in G.nodes()], index=G.nodes))

labelled_df = pd.read_csv('/homes/fabadmus/Internship/labeled_file2')
# get set of positive and negative concepts
pos = set(labelled_df[labelled_df['label'] == 'POS'].object)
neg = set(labelled_df[labelled_df['label'] == 'NEG'].object)

# create separate dataframes of embeddings bases on the sets
emb_df_pos = emb_df[emb_df.index.isin(pos)]
emb_df_pos['set'] = 'POS'
emb_df_neg = emb_df[emb_df.index.isin(neg)]
emb_df_neg['set'] = 'NEG'

# set the unkown rows as a validation dataframe
validation_df = pd.concat([ emb_df, emb_df_pos, emb_df_neg]).drop_duplicates(
    subset=emb_df.columns[:-1], keep='first')