'''
This script gets the first layer of relations for a given concept_id.
usage
python get_first_layer.py -p TWMET -i TWDIS_11098 -n 50 
'''
# import libraries
import os
import requests
import pandas as pd
import argparse as ap
import config

class GetFirstLayer():
    """  class to represnt first layer of relations
    for a concept_id

    :return: a first layer object
    :rtype: None
    """

    def __init__(self, api_key, concept_prefix,  *concept_id, top_n, download_path ):
        """function to set up connection to the API and to construct all 
        necessary attributes for the first layer of concept_id of interest

        :param api_key: key to access database API
        :type api_key: str
        :param concept_prefix: prefix that describes type of concept
        :type concept_prefix: str
        :param top_n: the number of relations to be returned after sorted from top to bottom
        :type top_n: int
        :param download_path: path to save the relations file
        :type download_path: str
        """
        
        # Set up the connection to the api
        self.session = requests.Session()
        self.base_url = 'https://apimlqv2.tenwiseservice.nl/api/mlquery/'
        self.session.headers['referer'] = 'https://apimlqv2.tenwiseservice.nl/'
        self.session.get(f"{self.base_url}start/")
        self.payload = {'apikey': api_key,  # contact KMAP for API
            'csrfmiddlewaretoken': self.session.cookies.get_dict()['csrftoken']}
        self.concept_id = ",".join([*concept_id])
        self.relations = self.get_relations()
        self.annotation = self.annotate()
        self.target_prefix = str(concept_prefix)
        self.df = self.json_todf()
        self.top_n = top_n
        if self.top_n == 0:
            self.final_df = self.modify_df()
        else:
            self.final_df = self.modify_df().head(top_n)
        self.file_path = download_path
        

    def get_relations(self): 
        """function to get concept relations from API

        :return: relations
        :rtype: list
        """

        self.payload['concept_ids_subject'] = self.concept_id
        results = self.session.post(f"{self.base_url}conceptset/relations/", self.payload)
        rv = results.json()
        relations = rv['result']['relations']
        object_ids = {x['object']:1 for x in relations}
        my_id_list = [self.concept_id] + list(object_ids.keys())
        self.payload['concept_ids'] = ",".join(my_id_list)
        # self.payload
        return  relations
    
    def annotate(self):
        """function to retreive concept annotations

        :return: annotation of results
        :rtype: dict
        """
        results = self.session.post(f"{self.base_url}conceptset/annotation/", self.payload)
        rv = results.json()
        return rv['result']['annotation']
    
    def annotating(self, concept):
        """function to retreive the names of concepts from annotation

        :param concept: concept for which name is to retreived
        :type concept: str
        :return: name of concept
        :rtype: str
        """
        return self.annotation[concept]['name'][0]
    
    def json_todf(self):
        """function to convert results from json to pandas data frame

        :return: pandas dataframe of relations
        :rtype: DataFrame
        """
        # Convert the results from json to pandas df
        df = pd.json_normalize(self.relations)
        df = df[['subject', 'object', 'score', 'overlap', 'local_mi']]
        df = df[df['score'] >= 1]
        print(type(df))
        return df

    def modify_df(self):
        """function to modify the relations dataframe by applying the
        annotating function, sorting according to score, overlap and
        weight, and filtering for metabolite relations only

        :return: relations dataframe
        :rtype: DataFrame
        """
        self.df['subject_annotated'] = self.df['subject'].apply(self.annotating)
        self.df['object_annotated'] = self.df['object'].apply(self.annotating)
        self.df.sort_values('score', ascending=False, inplace=True)
        self.df.sort_values('overlap', ascending=False, inplace=True)
        self.df.sort_values('local_mi', ascending=False, inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        # prefix_objects = ('TWMET', 'TWFOOD')
        self.final_df = self.df[self.df['object'].str.startswith(self.target_prefix)].reset_index(drop=True)
        return self.final_df

    def save_df(self):
        """function to save the relations dataframe
        """
        
        return pd.DataFrame.to_csv(self.final_df, self.file_path, sep='\t')
        
def main():
    argparser = ap.ArgumentParser(
                                description= "Script that gets first layer relation for a concept_id")
    argparser.add_argument("--CONCEPT_PREFIX", "-p",  action="store", type=str,
                            help=" Prefix of concept of interest")
    argparser.add_argument("--CONCEPT_ID", "-i", action="store", type=str,
                             help="ID of the concept(s) of interest, separated by comma")
    argparser.add_argument("--TOP_N", "-n", action="store", type=int, default=0,
                             help="Number of relations to return")

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
