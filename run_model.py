# import libraries
from sklearn.model_selection import GridSearchCV
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
import requests

G = get_graph('/homes/fabadmus/Internship/second_layer2')
embeddings = Word2Vec.load('embeddings2')
# remove target and control nodes
nodes = list(G.nodes)
sources = [i for i in nodes if i.startswith('TWDIS')]
for i in sources:
    nodes.remove(i)
# Convert embeddings to dataframe
emb_df = (pd.DataFrame([embeddings.wv.get_vector(str(n))
                        for n in G.nodes()], index=G.nodes))

#load  the first layer 
labelled_df = pd.read_csv('/homes/fabadmus/Internship/labeled_file2')
# get set of positive and negative concepts
pos = set(labelled_df[labelled_df['label'] == 'POS'].object)
neg = set(labelled_df[labelled_df['label'] == 'NEG'].object)

# create separate dataframes of embeddings bases on the sets
emb_df_pos = emb_df[emb_df.index.isin(pos)]
emb_df_pos['set'] = 'POS'
emb_df_neg = emb_df[emb_df.index.isin(neg)]
emb_df_neg['set'] = 'NEG'

# make training data from the embedding data
model_df = pd.concat([emb_df_pos, emb_df_neg])

# set the unkown rows as a validation dataframe
validation_df = pd.concat([ emb_df, emb_df_pos, emb_df_neg]).drop_duplicates(
    subset=emb_df.columns[:-1], keep='first')

#prepare data for modelling
# assign the independent and dependent variables
X = model_df.iloc[:,:-1].values
y = model_df['set']
# split into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)
#train model
param_grid = {'min_samples_leaf':[3,5,7,10,15],'max_features':[0.5,'sqrt','log2'],
          'max_depth':[10,15,20],
          'class_weight':[{"POS":1,"NEG":1},{"POS":1,"NEG":2},{"POS":5,"NEG":1},'balanced'],
          'criterion':['entropy','gini']}


model1 = GridSearchCV(RandomForestClassifier(),param_grid, verbose=1,n_jobs=-1,scoring='roc_auc')
model1.fit(X_train,y_train)
print ('\n',model1.best_estimator_)


# make predictions on the unknown
X_val = validation_df.iloc[:,:-1]
pred = model1.predict(X_val)
# get the prediction probabilities of the unknown
val_proba = model1.predict_proba(X_val)
# convert predictions and actual values to dataframe
val_proba_df = pd.DataFrame(val_proba, index=X_val.index,
                                columns=['NEG_prob', 'POS_prob'])
val_proba_df['predictions'] = pred
val_proba_df = val_proba_df.sort_values('POS_prob', ascending=False)
val_proba_df


session = requests.Session()
base_url = 'https://apimlqv2.tenwiseservice.nl/api/mlquery/'
session.headers['referer'] = 'https://apimlqv2.tenwiseservice.nl'
session.get(f"{base_url}start/")

payload = {'apikey': '',
           'csrfmiddlewaretoken': session.cookies.get_dict()['csrftoken']}

# annotate predictions
ids = list(val_proba_df.index)
payload['concept_ids'] = ",".join(ids)
results = session.post(f"{base_url}conceptset/annotation/", payload)
js = results.json()
annotation = js['result']['annotation']
# get ids
annotated_ids = []
for id in ids:
    annotated_ids.extend(annotation[id]['name'])
# add ids to the dataframe
val_proba_df['annotation'] = annotated_ids
val_proba_df.head(10)



