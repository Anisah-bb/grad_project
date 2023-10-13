'''
This scripts performs classification and makes predictions
usage- python run_mod.py -s 'second_layer' -m 'model_data_path' -e 'embedding.emb' -a adaboost -o out_file
'''

# import libraries
import os
import argparse as ap
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
import gensim.models.keyedvectors as word2vec
from sklearn.metrics import classification_report
import requests
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
import conf


class TrainModel():
    """class that represnts a classification model pipeline

    :return: a classification object
    :rtype: None
    """
    def __init__(self, apikey, second_layer, embedding_file, model_file, algorithm, out_file) :
        """_summary_

        :param apikey: key to access database API
        :type apikey: str
        :param second_layer: path to the second layer relations
        :type second_layer: str
        :param embedding_file: path to embedding file
        :type embedding_file: str
        :param model_file: path to relations for modelling
        :type model_file: str
        :param algorithm: algorithm for classification
        :type algorithm: str
        :param out_file: path to save predictions
        :type out_file: str
        """
        self.apikey = apikey
        self.session = requests.Session()
        self.base_url = 'http://www.tenwisedev.nl:6001/mlquery/'
        self.session.headers['referer'] = 'https://apimlq.tenwiseservice.nl/'
        self.mydict  = 'tagger_results_tf_2013'
        self.session.get(f"{self.base_url}start/")
        self.mydict = {'apikey': self.apikey,  # contact KMAP for API
            'csrfmiddlewaretoken': self.session.cookies.get_dict()['csrftoken']}
        self.second_layer = second_layer
        self.embedding_file = embedding_file
        self.prediction_path = out_file
        self.model_file = model_file
        self.embeddings_df = self.load_embeddings()
        self.label_embedding_sets()
        self.model_df = self.get_modeldf()
        self.validation_df = self.get_validationdf()
        self.train_features, self.test_features, self.train_target, self.test_target = self.prep_data()
        self.alg = algorithm
        if self.alg == 'randomforest':
            self.model = self.do_random_forest()
        elif self.alg == 'adaboost':
            self.model = self.do_adaboost()
        self.predictions = self.make_predictions()

    def load_embeddings(self):
        """function to load embedding file

        :return: embeddings for modelling
        :rtype: DataFrame
        """
        second_layer = pd.read_csv(self.second_layer, sep='\t')
        graph = nx.from_pandas_edgelist(second_layer, source='subject',
                                target='object', edge_attr='local_mi',
                                edge_key='local_mi', create_using= None)
        embeddings = word2vec.KeyedVectors.load_word2vec_format(self.embedding_file)
        return pd.DataFrame(
            [embeddings.get_vector(str(n)) for n in graph.nodes()],
            index=graph.nodes,
        )

    def label_embedding_sets(self):
        """function to label embeddings for training
        """
        #load the labeled first layer file
        labelled_df = pd.read_csv(self.model_file, '\t')
        # get set of positive and negative concepts
        pos = set(labelled_df[labelled_df['label'] == 'POS'].object)
        neg = set(labelled_df[labelled_df['label'] == 'NEG'].object)
        # label embeddings based on the laballed _df
        self.embeddings_df.loc[self.embeddings_df.index.isin(pos), 'SET'] = 'POS'
        print(len(self.embeddings_df.loc[self.embeddings_df.SET == 'POS']))
        self.embeddings_df.loc[self.embeddings_df.index.isin(neg), 'SET'] = 'NEG'
        print(len(self.embeddings_df.loc[self.embeddings_df.SET == 'NEG']))
        self.embeddings_df.loc[self.embeddings_df.SET.isnull(), 'SET'] = 'UNK'
        print(len(self.embeddings_df.loc[self.embeddings_df.SET == 'UNK']))

    def get_modeldf(self):
        """function to get the modelling data points

        :return: modelling embeddings
        :rtype: DataFrame
        """
        return self.embeddings_df.loc[
            (self.embeddings_df.SET == 'POS') | (self.embeddings_df.SET == 'NEG')
        ]
    def get_validationdf(self):
        """function to get the validation data

        :return: validation embeddings
        :rtype: DataFrame
        """
        # set the unkown rows as a validation dataframe
        validation_df = self.embeddings_df.loc[self.embeddings_df.SET == 'UNK']
        # validation_df = pd.concat([self.emb_df_pos,
        # self.emb_df_neg, self.emb_df]).drop_duplicates(
        # subset=self.emb_df.columns[:-1], keep=False)
        return validation_df

    def prep_data(self):
        """function to prepare modelling data for training

        :return: trining and testing data
        :rtype: list
        """
        #prepare data for modelling
        # assign the independent and dependent variables
        features = self.model_df.iloc[:,:-1].values
        target = self.model_df['SET']
        # split into training and testing
        train_features, test_features, train_target, test_target = train_test_split(
            features, target, test_size=0.3, random_state=42)
        return train_features, test_features, train_target, test_target

    def do_random_forest(self):
        """function to train a random forest model classifier model

        :return: rf model
        :rtype: scikit learn model
        """
        #train model
        model1 = RandomForestClassifier(random_state=42)
        model1.fit(self.train_features,self.train_target)
        pred1 = model1.predict(self.test_features)
        print(classification_report(self.test_target, pred1))
        print("RF Accuracy:",metrics.accuracy_score(self.test_target, pred1))

        clsf_report = pd.DataFrame(classification_report(self.test_target, pred1, output_dict=True)).transpose()
        report_path = self.prediction_path.strip('_predictions.txt') + '_report.txt'
        clsf_report.to_csv(report_path , sep='\t', index= True)
        
        roc = metrics.plot_roc_curve(model1, self.test_features, self.test_target)
        roc.figure_.savefig(self.prediction_path.strip('_predictions.txt') + '_roc')

        
        return model1
    def do_adaboost(self):
        """function to train an AdaBoost classifier model

        :return: adaboost model
        :rtype: scikit model
        """
        # Create adaboost classifer object
        abc = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1)
        # Train Adaboost Classifer
        model2 = abc.fit(self.train_features,self.train_target)
        #Predict the response for test dataset
        pred2 = model2.predict(self.test_features)
        print(classification_report(self.test_target, pred2))
        print("AdaBoost Accuracy:",metrics.accuracy_score(self.test_target, pred2))
        
        clsf_report = pd.DataFrame(classification_report( self.test_target, pred2, output_dict=True)).transpose()
        report_path = self.prediction_path.strip('_predictions.txt') + '_report.txt'
        clsf_report.to_csv(report_path , sep='\t', index= True)
        
        return model2

    def make_predictions(self):
        """function to make predictions from the classifier model

        :return: predictions dataframe
        :rtype: DataFrame
        """
        # make predictions on the unknown
        val_features = self.embeddings_df.iloc[:,:-1]
        predictions = self.model.predict(val_features)
        
        # get the prediction probabilities of the unknown
        val_proba = self.model.predict_proba(val_features)
        
        # convert predictions and actual values to dataframe
        val_proba_df = pd.DataFrame(val_proba, index=val_features.index,
                                        columns=['NEG_prob', 'POS_prob'])
        val_proba_df['predictions'] = predictions
        
        # combine actual and predicted values
        val_proba_df.index.names = ['ID']
        self.embeddings_df.index.names = ['ID']
        predictions_df = pd.merge(self.embeddings_df['SET'],val_proba_df, how = 'left', on = 'ID')
        
        # annotate predictions
        ids = list(predictions_df.index)
        self.mydict['concept_ids'] = ",".join(ids)
        results = self.session.post(f"{self.base_url}conceptset/annotation/", self.mydict)
        results = results.json()
        annotation = results['result']['annotation']
        
        # get ids
        annotated_ids = []
        for i in ids:
            annotated_ids.extend(annotation[i]['name'])
            
        # add ids to the dataframe
        predictions_df['annotation'] = annotated_ids
        predictions_df = predictions_df.sort_values('POS_prob', ascending=False)
        
        setting = self.prediction_path.strip('_predictions.txt').lstrip('/homes/fabadmus/Internship/grad_project/paper_output/')
        predictions_df['setting'] = setting
        
        metabs = predictions_df.sort_values('POS_prob', ascending=False)
        # metabs = predictions_df[['POS_prob', 'annotation']]
        pd.DataFrame.to_csv(metabs, self.prediction_path, sep='\t')
        # return predictions_df

def main():
    """function to catch argparser arguments and run script
    """
    argparser = ap.ArgumentParser(
                                description=
                                "Script that does machine learning and makes prediction")
    argparser.add_argument("--SECOND_LAYER", "-s", action="store", type=str,
                             help="name of second layer relations")
    argparser.add_argument("--MODEL_DATA", "-m", action="store", type=str,
                             help="name of modelling data")
    argparser.add_argument("--EMBEDDINGS", "-e", action="store", type=str,
                             help="name of embeddings file")
    argparser.add_argument("--ALGORITHM", "-a",action="store",  type = str,
                            help="Algoritm for classification")
    argparser.add_argument("--OUT_FILE", "-o",action="store",  type = str,
                            help="Path to save results")
    parsed = argparser.parse_args()
    api_key = conf.API_KEY
    second_layer_path = parsed.SECOND_LAYER
    model_data_path = parsed.MODEL_DATA
    embedding = parsed.EMBEDDINGS
    algorithm = parsed.ALGORITHM
    out_file = parsed.OUT_FILE
    # result_dir = config.RESULTS_DIRECTORY
    # if not os.path.isdir(result_dir):
    #     os.mkdir(result_dir)
    #     print("Result directory created")
    TrainModel(api_key, second_layer_path,
                       embedding,
                       model_data_path,
                       algorithm, out_file)
    # predictions = model.make_predictions()
    # print(predictions.head(20))

if __name__ == '__main__':
    main()
    