# import libraries
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import requests
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec


class TrainModel():
    def __init__(self, apikey, second_layer, emdedding_file, model_file, out_file) :
        self.apikey = apikey
        self.session = requests.Session()
        self.base_url = 'https://apimlqv2.tenwiseservice.nl/api/mlquery/'
        self.session.headers['referer'] = 'https://apimlqv2.tenwiseservice.nl/'
        self.session.get(f"{self.base_url}start/")
        self.payload = {'apikey': self.apikey,  # contact KMAP for API
            'csrfmiddlewaretoken': self.session.cookies.get_dict()['csrftoken']}
        self.second_layer = second_layer
        self.emdedding_file = emdedding_file
        self.prediction_path = out_file
        self.model_file = model_file
        self.emb_df = self.load_files()
        self.emb_df_pos, self.emb_df_neg =  self.extract_embedding_sets()
        self.model_df = self.get_modeldf()
        self.validation_df = self.get_validationdf()
        self.model = self.train_model()
        self.predictions = self.make_predictions()
   
    def load_files(self):
        second_layer = pd.read_csv(self.second_layer)
        second_layer = second_layer[second_layer.subject.str.startswith('TWDIS') == False]
        G = nx.from_pandas_edgelist(second_layer, source='subject',
                                target='object', edge_attr='local_mi', edge_key='local_mi', create_using= None)

        embeddings = Word2Vec.load(self.emdedding_file)
        # Convert embeddings to dataframe
        emb_df = (pd.DataFrame([embeddings.wv.get_vector(str(n))
                                for n in G.nodes()], index=G.nodes))
        return emb_df
        
        
    def extract_embedding_sets(self):
        #load  the first layer 
        labelled_df = pd.read_csv(self.model_file)
        # get set of positive and negative concepts
        pos = set(labelled_df[labelled_df['label'] == 'POS'].object)
        neg = set(labelled_df[labelled_df['label'] == 'NEG'].object)

        # create separate dataframes of embeddings bases on the sets
        emb_df_pos = self.emb_df[self.emb_df.index.isin(pos)]
        emb_df_pos['set'] = 'POS'
        print(len(emb_df_pos))
        emb_df_neg = self.emb_df[self.emb_df.index.isin(neg)]
        emb_df_neg['set'] = 'NEG'
        print(len(emb_df_neg))
        return emb_df_pos, emb_df_neg
        
    def get_modeldf(self):
        # make training data from the embedding data
        model_df = pd.concat([self.emb_df_pos, self.emb_df_neg])
        return model_df
    def get_validationdf(self):
        # set the unkown rows as a validation dataframe
        validation_df = pd.concat([self.emb_df_pos, self.emb_df_neg, self.emb_df]).drop_duplicates(
                subset=self.emb_df.columns[:-1], keep=False)
        
        return validation_df
    def train_model(self):
        #prepare data for modelling
        # assign the independent and dependent variables
        X = self.model_df.iloc[:,:-1].values
        y = self.model_df['set']
        # split into training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)
        #train model
        param_grid = {'min_samples_leaf':[3,5,7,10,15],'max_features':[0.5,'sqrt','log2'],
                'max_depth':[10,15,20],
                'class_weight':[{"POS":1,"NEG":1},'balanced'],
                'criterion':['entropy','gini']}

        # model = RandomForestClassifier(random_state=42, class_weight='balanced', criterion='entropy',
        #               max_depth=20, max_features=0.5, min_samples_leaf=3)
        model1 = GridSearchCV(RandomForestClassifier(),param_grid, verbose=1,n_jobs=-1,scoring='roc_auc')
        model1.fit(X_train,y_train)
        pred = model1.predict(X_test)
        print(classification_report(y_test, pred))
        print ('\n',model1.best_estimator_)
        return model1
        

    def make_predictions(self):
        # make predictions on the unknown
        X_val = self.validation_df.iloc[:,:-1]
        pred = self.model.predict(X_val)
        # get the prediction probabilities of the unknown
        val_proba = self.model.predict_proba(X_val)
        # convert predictions and actual values to dataframe
        val_proba_df = pd.DataFrame(val_proba, index=X_val.index,
                                        columns=['NEG_prob', 'POS_prob'])
        val_proba_df['predictions'] = pred
        val_proba_df = val_proba_df.sort_values('POS_prob', ascending=False)
        val_proba_df

        # annotate predictions
        ids = list(val_proba_df.index)
        self.payload['concept_ids'] = ",".join(ids)
        results = self.session.post(f"{self.base_url}conceptset/annotation/", self.payload)
        js = results.json()
        annotation = js['result']['annotation']
        # get ids
        annotated_ids = []
        for id in ids:
            annotated_ids.extend(annotation[id]['name'])
        # add ids to the dataframe
        val_proba_df['annotation'] = annotated_ids
        val_proba_df.head(10)
        pd.DataFrame.to_csv(val_proba_df, self.prediction_path)
        return val_proba_df
        
def main():
    train = TrainModel('Hanze_group_2022', '/homes/fabadmus/Internship/RA/second_layer', '/homes/fabadmus/Internship/RA/embedding', '/homes/fabadmus/Internship/RA/model_data_path', '/homes/fabadmus/Internship/RA/results')
    predictions = train.make_predictions()
    print(predictions.head(20))
    
if __name__ == '__main__':
    main()