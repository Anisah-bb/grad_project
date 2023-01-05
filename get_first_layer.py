'''
This script gets the first layer of relations for a given concept_id.
usage
get_first_layer.py -p TWMET -i TWDIS_11098 -n 50 
'''
# import libraries
import os
import requests
import pandas as pd
import argparse as ap
import config

class GetFirstLayer(): 
    
    def __init__(self, api_key, concept_prefix,  *concept_id, top_n, download_path ):
        '''function to set up connection to the API and to
        catch the concept_id of interest
        '''
    # Set up the connection to the api
        self.session = requests.Session()
        self.base_url = 'https://apimlqv2.tenwiseservice.nl/api/mlquery/'
        self.session.headers['referer'] = 'https://apimlqv2.tenwiseservice.nl/'
        self.session.get(f"{self.base_url}start/")
        self.payload = {'apikey': api_key,  # contact KMAP for API
            'csrfmiddlewaretoken': self.session.cookies.get_dict()['csrftoken']}
        self.concept_id = ",".join([*concept_id])
        self.payload, self.relations = self.get_relations()
        self.annotation = self.annotate()
        self.target_prefix = str(concept_prefix)
        self.df = self.json_todf()
        self.top_n = top_n
        self.final_df = self.modify_df()
        self.file_path = download_path
        
    
    
    # create a function to get relations from API
    def get_relations(self):  
        '''
        function to get concept relations from API
        input : concept_id
        output: payload, relations
        '''
        self.payload['concept_ids_subject'] = self.concept_id
        results = self.session.post(f"{self.base_url}conceptset/relations/", self.payload)
        rv = results.json()
        self.relations = rv['result']['relations']
        object_ids = {x['object']:1 for x in self.relations}
        my_id_list = [self.concept_id] + list(object_ids.keys())
        self.payload['concept_ids'] = ",".join(my_id_list)

        return self.payload, self.relations
    
    def annotate(self):
        '''
        function to annotate payload
        '''
        results = self.session.post(f"{self.base_url}conceptset/annotation/", self.payload)
        rv = results.json()
        return rv['result']['annotation']
    
        # function to annotate concept
    def annotating(self, concept):
        '''
        function to annotate concept
        '''
        return self.annotation[concept]['name'][0]
    
    def json_todf(self):
        '''
        function to convert results from json to pandas data frame
        input: relations in json
        output: pandas dataframe of relations
        '''
        # Convert the results from json to pandas df
        self.df = pd.json_normalize(self.relations)
        self.df = self.df[['subject', 'object', 'score', 'overlap', 'local_mi']]
        self.df = self.df[self.df['score'] >= 1]
        # self.df = self.df[self.df['overlap'] >= 100].reset_index(drop=True)
        return self.df


    # function to modify the dataframe
    def modify_df(self):
        '''
        function to modify the relations dataframe by applying the
        annotating function, sorting according to score, overlap and
        weight, and filtering for food and metabolite relations only
        input: dataframe
        output: modified dataframe
        '''
        self.df['subject_annotated'] = self.df['subject'].apply(self.annotating)
        self.df['object_annotated'] = self.df['object'].apply(self.annotating)
        # self.df['weight'] = self.df['score'] * self.df['overlap']
        self.df.sort_values('score', ascending=False, inplace=True)
        self.df.sort_values('overlap', ascending=False, inplace=True)
        self.df.sort_values('local_mi', ascending=False, inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        # prefix_objects = ('TWMET', 'TWFOOD')
        self.final_df = self.df[self.df['object'].str.startswith(self.target_prefix)].reset_index(drop=True)
        return self.final_df.head(self.top_n)

    def save_df(self):
        '''
        function to save the relations dataframe
        '''
        return pd.DataFrame.to_csv(self.final_df, self.file_path, sep='\t')
        
def main():
    argparser = ap.ArgumentParser(
                                description= "Script that gets first layer relation for a concept_id")
    argparser.add_argument("--CONCEPT_PREFIX", "-p",  action="store", type=str,
                            help=" Prefix of concept of interest")
    argparser.add_argument("--CONCEPT_ID", "-i", action="store", type=str,
                             help="ID of the concept(s) of interest, separated by comma")
    argparser.add_argument("--TOP_N", "-n", action="store", type=int,
                             help="Number of relations to return")
    # argparser.add_argument("--RESULT_DIRECTORY", "-d", action="store", type=str,
    #                          help="Path to save result")
    parsed = argparser.parse_args()
    api_key = config.API_KEY
    concept_prefix = parsed.CONCEPT_PREFIX
    concept_id = parsed.CONCEPT_ID
    n = parsed.TOP_N
    result_dir = config.RESULTS_DIRECTORY
    # result_dir = parsed.RESULT_DIRECTORY
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
        print("Result directory created")
    # result_dir+'/'+concept_id
    GetFirstLayer(api_key, concept_prefix, concept_id, top_n=n, download_path= result_dir+'/'+concept_id).save_df()
    # GetFirstLayer(api_key, concept_prefix, control_concept_id,top_n=n, download_path= f'{result_dir}/control').save_df()
if __name__ == '__main__':
    main()
