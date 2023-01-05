'''
This script gets the second layer of relations from the first layer relations
usage
get_second_layer.py  -t TWDIS_06685 -c TWDIS_09536 -s 2 
'''
import os
import argparse as ap
import pandas as pd
import requests
import config


#model_data_path = '/homes/fabadmus/Internship/labeled_file2'
class GetSecondLayer():
    '''class to get second layer relations of 
    positive and negative concepts 
    '''
    def __init__(self, apikey, pos_df_path, neg_df_path, model_data_path, top_n, download_path):
        self.session = requests.Session()
        self.search_url = "relations"
        self.base_url = 'https://apimlqv2.tenwiseservice.nl/api/mlquery/'
        self.session.headers['referer'] = 'https://apimlqv2.tenwiseservice.nl/'
        self.session.get(f"{self.base_url}start/")
        self.payload = {'apikey': apikey, 'csrfmiddlewaretoken': self.session.cookies.get_dict()['csrftoken']}
        self.pos_df = pd.read_csv(pos_df_path, sep="\t")
        self.neg_df = pd.read_csv(neg_df_path, sep="\t")
        self.top_n = top_n
        self.model_data_path = model_data_path 
        self.file_path = download_path
        self.full_df = self.join_files()
        self.set_of_concepts = self.get_concept_set()
        self.save_model_df()
        self.second_relations_df = self.get_secondlayer_relation()
        self.clean_df = self.clean_second_layer()
        self.final_df = self.combine_dfs()
        
    def join_files(self):
        '''funtion to combine the positive and negative sets and drop
       the overlaps
       '''
        self.pos_df['label'] = 'POS'
        self.neg_df['label'] = 'NEG'
        full_df = pd.concat([self.pos_df, self.neg_df], ignore_index=True)
        full_df = full_df[['subject', 'object', 'local_mi', 'label']]
        return full_df
    
        
    def get_concept_set(self):
        ''' funtion to get the set of concepts in the first layer
        '''
        return set(self.full_df['object'].unique())
    
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
        df = self.second_relations_df2[['subject', 'object', 'score', 'overlap','local_mi']]
        # self.df = self.df[self.df['score'] >= 1]
        # self.df = self.df[self.df['overlap'] >= 100].reset_index(drop=True)
        # self.df = self.df.drop(columns = ['score', 'overlap'])
        #df.sort_values('local_mi', ascending=False, inplace=True)
        # filter out self loops
        df = df[df['subject'] != df['object']]
        # remove duplicates since relations always goes both ways
        # self.df = self.df.iloc[::2]
        # get relations with the highest local_mi
        # self.df = self.df.groupby(['subject'])[['object', 'local_mi']].max().reset_index()
        df = df.groupby(['subject']).apply(lambda x: x.sort_values(['local_mi'], ascending = False)[:self.top_n])
        df = df[['subject', 'object', 'local_mi']]
        self.df = df.set_index('subject').reset_index()
        return self.df
    
    def combine_dfs(self):
        self.final_df = pd.concat([self.full_df, self.df], ignore_index=True)
        self.final_df = self.final_df.drop(columns= 'label')
        return self.final_df
    
    def save_second_layer(self):
        ''' function to save the second layer relations. 
        '''
        return pd.DataFrame.to_csv(self.final_df, self.file_path)
    
def main():
    argparser = ap.ArgumentParser(
                                description= "Script that gets second layer relations for a target and a control")
    argparser.add_argument("--TARGET_FILE", "-t", action="store", type=str,
                             help="name of target relations")
    argparser.add_argument("--CONTROL_FILE", "-c", action="store", type=str,
                             help="name of control relations")
    argparser.add_argument("--SIZE", "-s", action="store", type=int,
                             help="Size of second layer")
    # argparser.add_argument("--RESULT_DIRECTORY", "-d", action="store", type=str,
    #                          help="Path to save result")
    parsed = argparser.parse_args()
    api_key = config.API_KEY
    targ = parsed.TARGET_FILE
    cont = parsed.CONTROL_FILE
    n = parsed.SIZE
    result_dir = config.RESULTS_DIRECTORY
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
        print("Result directory created")
    GetSecondLayer(api_key, pos_df_path=result_dir+'/'+targ, neg_df_path=result_dir+'/'+cont, model_data_path =f'{result_dir}/model_data_path', download_path = f'{result_dir}/second_layer', top_n=n).save_second_layer()    
if __name__ == '__main__':
    main()   