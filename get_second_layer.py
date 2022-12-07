import pandas as pd
import requests
import pandas as pd

#model_data_path = '/homes/fabadmus/Internship/labeled_file2'
class GetSecondLayer():
    '''class to get second layer relations of 
    positive and negative concepts 
    '''
    def __init__(self, apikey, pos_df_path, neg_df_path, model_data_path, download_path):
        self.session = requests.Session()
        self.search_url = "relations"
        self.base_url = 'https://apimlqv2.tenwiseservice.nl/api/mlquery/'
        self.session.headers['referer'] = 'https://apimlqv2.tenwiseservice.nl/'
        self.session.get(f"{self.base_url}start/")
        self.payload = {'apikey': apikey, 'csrfmiddlewaretoken': self.session.cookies.get_dict()['csrftoken']}
        self.pos_df = pd.read_csv(pos_df_path)
        self.neg_df = pd.read_csv(neg_df_path)
        self.model_data_path = model_data_path 
        self.file_path = download_path
        self.full_df = self.join_files()
        self.set_of_concepts = self.get_concept_set()
        self.save_model_df()
        self.second_relations_df = self.get_secondlayer_relation()
        self.clean_df = self.clean_second_layer()
        self.final_df = self.combine_dfs()
        
        # self.clean_df = self.clean_intra_df()
        # self.embedding_df = self.label_intra_df()
        
    def join_files(self):
        '''funtion to combine the positive and negative sets and drop
       the overlaps
       '''
        self.pos_df['label'] = 'POS'
        self.neg_df['label'] = 'NEG'
        self.full_df = pd.concat([self.pos_df, self.neg_df], ignore_index=True)
        self.full_df = self.full_df[['subject', 'object', 'local_mi', 'label']]
        return self.full_df
    
        
    def get_concept_set(self):
        ''' funtion to get the set of concepts in the first layer
        '''
        self.set_of_concepts = set(self.full_df['object'].unique())
        return self.set_of_concepts
    
    def save_model_df(self):
        ''' funtion to save the first layer dataframe for modelling
        '''
        # remove overlaps
        self.full_df.drop_duplicates('object', keep=False, inplace=True)
        pd.DataFrame.to_csv(self.full_df, self.model_data_path)

    def get_secondlayer_relation(self):
        ''' function to get the second layer relations 
        '''
        # get all the metabolites related to first layer metabolites
        self.payload['concept_ids_subject'] = ",".join(self.set_of_concepts)
        self.payload['vocab_ids'] = "ONT1006"
        results = self.session.post(f"{self.base_url}conceptset/{self.search_url}/", self.payload)
        rv = results.json()
        second_relations_edges = rv['result'][f'{self.search_url}']
        self.second_relations_df = pd.DataFrame(second_relations_edges)
        
        # get the intra relations of matbolites
        self.set_of_concepts2 = set(self.second_relations_df['object'].unique())
        self.payload['concept_ids_subject'] = ",".join(self.set_of_concepts2)
        # self.payload['concept_ids_object'] = ",".join(self.set_of_concepts)
        self.payload['vocab_ids'] = "ONT1006"
        results = self.session.post(f"{self.base_url}conceptset/{self.search_url}/", self.payload)
        rv = results.json()
        second_relations_edges2 = rv['result'][f'{self.search_url}']
        self.second_relations_df2 = pd.DataFrame(second_relations_edges2)
        # pd.DataFrame.to_csv(self.second_relations_df2, '/homes/fabadmus/Internship/test_file')
        return self.second_relations_df2
    

    def clean_second_layer(self):
        '''function to clean the data and sort according to the
        local_mi.
        '''
        self.df = self.second_relations_df2[['subject', 'object', 'score', 'overlap','local_mi']]
        self.df = self.df[self.df['score'] >= 1]
        self.df = self.df[self.df['overlap'] >= 100].reset_index(drop=True)
        # self.df = self.df.drop(columns = ['score', 'overlap'])
        self.df.sort_values('local_mi', ascending=False, inplace=True)
        # filter out self loops
        self.df = self.df[self.df['subject'] != self.df['object']]
        # remove duplicates since relations always goes both ways
        self.df = self.df.iloc[::2]
        # get relations with the highest local_mi
        self.df = self.df.groupby(['subject'])[['object', 'local_mi']].max().reset_index()
        return self.df
    
    def combine_dfs(self):
        self.final_df = pd.concat([self.full_df, self.df], ignore_index=True)
        self.final_df = self.final_df.drop(columns= 'label')
        return self.final_df
    
    def save_second_layer(self):
        ''' function to save the second layer relations. 
        '''
        return pd.DataFrame.to_csv(self.final_df, self.file_path)
    
    
    # def clean_intra_df(self):
    #     ''' function to clean the intra_df and drop uneccessary columns 
    #     '''
    #     # filter out self loops
    #     self.clean_df = self.intra_relations_df[self.intra_relations_df['subject'] != self.intra_relations_df['object']]
    #     # remove duplicates
    #     self.clean_df = self.clean_df.iloc[::2]
    #     # self.clean_df['weight'] = self.clean_df['score'] * self.clean_df['overlap']
    #     return self.clean_df
        
    # def label_intra_df(self):
    #     ''' function to label the intra relations as positive or
    #     negative concepts and return the dataframe for embedding
    #     '''
    #     # first combine both dataframes
    #     self.embedding_df = pd.concat([self.full_df, self.clean_df])
    #     # create positive and negative sets
    #     pos = set(self.full_df[self.full_df['label'] == 'POS'].object)
    #     neg = set(self.full_df[self.full_df['label'] == 'NEG'].object)
    #     # label based on the set
    #     self.embedding_df.loc[self.embedding_df.label.isnull() & self.embedding_df.subject.isin (pos), 'label'] = 'POS'
    #     self.embedding_df.loc[self.embedding_df.label.isnull() & self.embedding_df.subject.isin (neg), 'label'] = 'NEG'
    #     # select the required columns
    #     self.embedding_df = self.embedding_df[['subject', 'object', 'local_mi']]
    #     return self.embedding_df
        
    # def save_embedding_df(self):
    #     return pd.DataFrame.to_csv(self.embedding_df, self.file_path) 
    

    