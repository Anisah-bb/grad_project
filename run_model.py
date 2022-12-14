'''
This scripts performs classification and makes predictions
usage
run_model.py -s 'second_layer' -m 'model_data_path' -e 'embedding' -a adaboost 
'''

# import libraries
import os
import argparse as ap
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
from sklearn.ensemble import AdaBoostClassifier
import config


class TrainModel():
    def __init__(self, apikey, second_layer, emdedding_file, model_file, algorithm, out_file) :
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
        self.X_train, self.X_test, self.y_train, self.y_test = self.prep_data()
        self.alg = algorithm
        if self.alg == 'rf':
            self.model = self.do_random_forest()
        elif self.alg == 'adaboost':
            self.model = self.do_adaboost()
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
    
    def prep_data(self):
        #prepare data for modelling
        # assign the independent and dependent variables
        X = self.model_df.iloc[:,:-1].values
        y = self.model_df['set']
        # split into training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test
        
    def do_random_forest(self):
        #train model
        param_grid = {'min_samples_leaf':[3,5,7,10,15],'max_features':[0.5,'sqrt','log2'],
          'max_depth':[10,15,20],
          'class_weight':[{"POS":3,"NEG":1},{"POS":1,"NEG":1}],
          'criterion':['entropy','gini']}
        model1 = GridSearchCV(RandomForestClassifier(),param_grid, verbose=1,n_jobs=-1,scoring='roc_auc')
        model1.fit(self.X_train,self.y_train)
        pred1 = model1.predict(self.X_test)
        print(classification_report(self.y_test, pred1))
        # print ('\n',model1.best_estimator_)
        print("RF Accuracy:",metrics.accuracy_score(self.y_test, pred1))
        return model1
    def do_adaboost(self):
        # Create adaboost classifer object
        abc = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1)
        # Train Adaboost Classifer
        model2 = abc.fit(self.X_train, self.y_train)

        #Predict the response for test dataset
        pred2 = model2.predict(self.X_test)
        print(classification_report(self.y_test, pred2))
        print("AdaBoost Accuracy:",metrics.accuracy_score(self.y_test, pred2))
        return model2   
       

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
        metabs = val_proba_df[['POS_prob', 'annotation']]
        pd.DataFrame.to_csv(metabs, self.prediction_path, sep='\t')
        return val_proba_df
        
def main():
    argparser = ap.ArgumentParser(
                                description= "Script that does machine learning and makes prediction")
    argparser.add_argument("--SECOND_LAYER", "-s", action="store", type=str,
                             help="name of second layer relations")
    argparser.add_argument("--MODEL_DATA", "-m", action="store", type=str,
                             help="name of modelling data")
    argparser.add_argument("--EMBEDDINGS", "-e", action="store", type=str,
                             help="name of embeddings file")
    argparser.add_argument("--ALGORITHM", "-a",action="store",  type = str,
                            help="Algoritm for classification")
    # argparser.add_argument("--RESULT_DIRECTORY", "-d", action="store", type=str,
    #                          help="Path to save result")
    parsed = argparser.parse_args()
    api_key = config.API_KEY
    second_layer_path = parsed.SECOND_LAYER
    model_data_path = parsed.MODEL_DATA
    embedding = parsed.EMBEDDINGS
    algorithm = parsed.ALGORITHM
    result_dir = config.RESULTS_DIRECTORY
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
        print("Result directory created")
    # train = TrainModel(, '/homes/fabadmus/Internship/RA/second_layer', '/homes/fabadmus/Internship/RA/embedding', '/homes/fabadmus/Internship/RA/model_data_path', '/homes/fabadmus/Internship/RA/results')
    model = TrainModel(api_key, result_dir+'/'+second_layer_path,result_dir+'/'+embedding, result_dir+'/'+model_data_path, algorithm,  f'{result_dir}/results')
    predictions = model.make_predictions()
    print(predictions.head(20))
    
if __name__ == '__main__':
    main()