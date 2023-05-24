'''
This script gets the second layer of relations from the first layer relations
usage- python get_second_layer.py  -t TWDIS_06685 -c TWDIS_09536 -s 2
'''
import os
import argparse as ap
import pandas as pd
import requests
import config

class GetSecondLayer():
    """class that represents second layer relations of
    positive and negative concepts

    :return: a second layer object
    :rtype: None
    """
    def __init__(self, apikey, pos_df_path, neg_df_path, model_data_path, size, download_path):
        """function to construct all necessary attributes for a second layer

        :param apikey: key to access database API
        :type apikey: str
        :param pos_df_path: path to the positive(target) relations file
        :type pos_df_path: str
        :param neg_df_path: path to the negative(control) relations file
        :type neg_df_path: str
        :param model_data_path: path to save the relations for modelling
        :type model_data_path: str
        :param top_n: desired size for the second layer
        :type top_n: str
        :param download_path: path to save the second layer relations
        :type download_path: str
        """
        self.session = requests.Session()
        self.search_url = "relations"
        self.base_url = 'https://apimlqv2.tenwiseservice.nl/api/mlquery/'
        self.session.headers['referer'] = 'https://apimlqv2.tenwiseservice.nl/'
        self.session.get(f"{self.base_url}start/")
        self.payload = {'apikey': apikey,
                        'csrfmiddlewaretoken': self.session.cookies.get_dict()['csrftoken']}
        self.pos_df = pd.read_csv(pos_df_path, sep="\t")
        self.neg_df = pd.read_csv(neg_df_path, sep="\t")
        self.size = size
        self.model_data_path = model_data_path
        self.file_path = download_path
        self.full_df = self.join_files()
        self.set_of_concepts = self.get_concept_set()
        self.save_model_df()
        self.second_relations_df = self.get_secondlayer_relation()
        self.intranetwork_df = self.get_intranetwork()
        self.second_layer = self.combine_dfs()

    def join_files(self):
        """function to combine the positive and negative sets and drop
        the overlaps

        :return: dataframe containing modelling relations
        :rtype: str
        """

        self.pos_df['label'] = 'POS'
        self.neg_df['label'] = 'NEG'
        full_df = pd.concat([self.pos_df, self.neg_df], ignore_index=True)
        full_df = full_df[['subject', 'object', 'local_mi', 'label']]
        return full_df

    def get_concept_set(self):
        """function to get the set of concepts in the first layer

        :return: set of unique object concepts in the first layer
        :rtype: set
        """
        return set(self.full_df['object'].unique())

    def save_model_df(self):
        """funtion to save the first layer dataframe for modelling
        """
        # remove overlaps
        self.full_df.drop_duplicates('object', keep=False, inplace=True)
        pd.DataFrame.to_csv(self.full_df, self.model_data_path, sep='\t')

    def get_secondlayer_relation(self):
        """function to get the second layer relations

        :return: dataframe second layer relations
        :rtype: DataFrame
        """
        # get all the metabolites related to first layer metabolites
        self.payload['concept_ids_subject'] = ",".join(self.set_of_concepts)
        self.payload['vocab_ids'] = "ONT1006"
        results = self.session.post(f"{self.base_url}conceptset/{self.search_url}/", self.payload)
        results = results.json()
        second_relations_edges = results['result'][f'{self.search_url}']
        second_relations_df = pd.DataFrame(second_relations_edges)
        second_relations_df = second_relations_df[second_relations_df
                                                  ['subject'] != second_relations_df['object']]
        second_relations_df  = second_relations_df.groupby(
            ['subject']).apply(lambda x: x.sort_values(['local_mi'], ascending = False)[:self.size])
        second_relations_df = second_relations_df[['subject', 'object', 'local_mi']]
        second_relations_df = second_relations_df.set_index('subject').reset_index()
        return second_relations_df[['subject', 'object', 'local_mi']]

    def get_intranetwork(self):
        """function to get intranetwork of second layer

        :return: intranetwork relations of the second layer
        :rtype: DataFrmae
        """
        set_of_concepts2 = set(self.second_relations_df['object'].unique())
        self.payload['concept_ids_subject'] = ",".join(set_of_concepts2)
        self.payload['concept_ids_object'] = ",".join(set_of_concepts2)
        self.payload['vocab_ids'] = "ONT1006"
        results = self.session.post(
            f"{self.base_url}conceptset/{self.search_url}/", self.payload)
        results = results.json()
        intra_relations_edges = results['result'][f'{self.search_url}']
        intra_relations_df = pd.DataFrame(intra_relations_edges)
        # filter out self loops
        intra_relations_df = intra_relations_df[
            intra_relations_df['subject'] != intra_relations_df['object']]
        # remove duplicates since relations always goes both ways
        intra_relations_df = intra_relations_df.iloc[::2]
        # filter for important ralations
        intra_relations_df = intra_relations_df[intra_relations_df['score'] >= 1]
        return intra_relations_df[['subject', 'object', 'local_mi']]

    def combine_dfs(self):
        """function to combine first and second layers of the network

        :return: _description_
        :rtype: _type_
        """
        return pd.concat(
            [self.second_relations_df, self.intranetwork_df], ignore_index=True
        )

    def save_second_layer(self):
        """function to save the second layer relations
        """
        pd.DataFrame.to_csv( self.second_layer, self.file_path, sep='\t')

def main():
    """function to catch argparser arguments and run script
    """
    argparser = ap.ArgumentParser(
                                description=
                                "Script to get second layer relations")
    argparser.add_argument("--TARGET_FILE", "-t", action="store", type=str,
                             help="name of target relations")
    argparser.add_argument("--CONTROL_FILE", "-c", action="store", type=str,
                             help="name of control relations")
    argparser.add_argument("--SIZE", "-s", action="store", type=int, default=1,
                             help="Size of second layer")
    argparser.add_argument("--LABEL_PATH", "-l", action="store", type=str, 
                             help="Path to save lebel data")
    argparser.add_argument("--SL_PATH", "-o", action="store", type=str, 
                             help="Path to save second layer ")
    parsed = argparser.parse_args()
    api_key = config.API_KEY
    targ = parsed.TARGET_FILE
    cont = parsed.CONTROL_FILE
    size = parsed.SIZE
    label_path = parsed.LABEL_PATH
    sl_path = parsed.SL_PATH
    # result_dir = config.RESULTS_DIRECTORY
    # if not os.path.isdir(result_dir):
    #     os.mkdir(result_dir)
    #     print("Result directory created")
    GetSecondLayer(api_key, pos_df_path=targ,
                   neg_df_path= cont,
                   model_data_path =label_path,
                   download_path = sl_path, size=size).save_second_layer()
if __name__ == '__main__':
    main()
