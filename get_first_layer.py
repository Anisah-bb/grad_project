'''
This script gets the first layer of relations for a given concept_id.
usage- python get_first_layer.py -p TWMET -i TWDIS_11098 -n 50 -o 
'''
# import libraries
import os
import argparse as ap
import requests
import pandas as pd
import config

class GetFirstLayer():
    """  class that represents first layer of relations
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
        self.data = self.json_todf()
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
        results = results.json()
        relations = results['result']['relations']
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
        results = results.json()
        return results ['result']['annotation']

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
        data = pd.json_normalize(self.relations)
        data = data[['subject', 'object', 'score', 'overlap', 'local_mi']]
        # return data[data['score'] >= 1]
        return data[(data['score'] >= 1) & (data['overlap'] >= 100)]

    def modify_df(self):
        """function to modify the relations dataframe by applying the
        annotating function, sorting according to score, overlap and
        weight, and filtering for metabolite relations only

        :return: relations dataframe
        :rtype: DataFrame
        """
        self.data = self.data[self.data['object']
                              .str.startswith(self.target_prefix)].reset_index(drop=True)
        # print(self.annotation['TWMET_01238']['name'][0])
        self.data['subject_annotated'] = self.data['subject'].apply(self.annotating)
        self.data['object_annotated'] = self.data['object'].apply(self.annotating)
        self.data.sort_values('score', ascending=False, inplace=True)
        self.data.sort_values('overlap', ascending=False, inplace=True)
        self.data.sort_values('local_mi', ascending=False, inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        return self.data

    def save_df(self):
        """function to save the relations dataframe
        """

        return pd.DataFrame.to_csv(self.final_df, self.file_path, sep='\t')

def main():
    """function to catch argparser arguments and run script
    """
    argparser = ap.ArgumentParser(
                                description=
                                "Script that gets first layer relation for a concept_id")
    argparser.add_argument("--CONCEPT_PREFIX", "-p",  action="store", type=str,
                            help=" Prefix of concept of interest")
    argparser.add_argument("--CONCEPT_ID", "-i", action="store", type=str,
                             help="ID of the concept(s) of interest, separated by comma")
    argparser.add_argument("--TOP_N", "-n", action="store", type=int, default=0,
                             help="Number of relations to return")
    argparser.add_argument("--OUT_FILE", "-o", action="store", type=str, default=0,
                             help="Output file")

    parsed = argparser.parse_args()
    api_key = config.API_KEY
    concept_prefix = parsed.CONCEPT_PREFIX
    concept_id = parsed.CONCEPT_ID
    top_n = parsed.TOP_N
    out_file = parsed.OUT_FILE
    # result_dir = config.RESULTS_DIRECTORY
    # if not os.path.isdir(result_dir):
    #     os.mkdir(result_dir)
    #     print("Result directory created")
    GetFirstLayer(api_key,
                  concept_prefix,
                  concept_id,
                  top_n=top_n,
                  download_path= out_file).save_df()
if __name__ == '__main__':
    main()
