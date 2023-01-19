import elqel.main_dense as main_dense
import argparse
import json
import os
import pickle
import sys
from termcolor import cprint
from elqel.utils import get_predictions_elq_df

class ELQEntityLinker():
    """_summary_
    """
    def __init__(self,
                 models_path,
                 biencoder_path,
                 prediction_type="multiple", 
                 num_cand_mentions=20,
                 num_cand_entities=10,
                 threshold_type="thresholded_entity_by_mention",
                 threshold=-2, 
                 mention_threshold=-1,
                 output_filename="elq_annotations"):
        
        self.models_path = models_path
        self.prediction_type = prediction_type
        self.config = {
            "interactive": False,
            "prediction_type": prediction_type,
            "biencoder_model": biencoder_path,
            "biencoder_config": models_path + "elq_large_params_finetune.txt",
            "cand_token_ids_path": models_path + "all_aats_ids.t7",
            "entity_catalogue": models_path + "AAT_KB.jsonl",
            "entity_encoding": models_path + "aat_enc.t7",
            "faiss_index": "none",
            "index_path": "",
            "num_cand_mentions": num_cand_mentions,
            "num_cand_entities": num_cand_entities,
            "threshold_type": threshold_type,
            "threshold": threshold,
            "mention_threshold": mention_threshold
        }
        
        self.args = argparse.Namespace(**self.config)
        self.model = main_dense.load_models(self.args, logger=1)
        
    def entity_linking(self, data_to_link, save_path=False):
        """Perform entity linking predictions with ELQ

        Args:
            - data_to_link (dict list): List of dictionaries including data to annotate. 
                example: [{"id": 0, "text": "bronze jar"}, {"id": 1, "text": "human figure"}]
            - save_path (str): Path to store predictions in pickle format. Defaults to False.
        """
        self.predictions = main_dense.run(self.args, *self.model, test_data=data_to_link)
        
        if save_path:
            with open(save_path, 'wb') as f:
                pickle.dump(self.predictions, f)
        cprint(f"Annotations (raw) saved: {save_path}", "green")
        
        return self.predictions
            
    def preds2dataframe(self, save_path=False):
        """Convert preditions to pandas DataFrame format

        Args:
            save_path (str, optional): Path to store DataFrame in CSV format. Defaults to False.
        """
        string_cases_dictionary = {" - ": "-",
                                    "( ": "(",
                                    " )": ")",
                                    "s'- spun": "'s'-spun",
                                    "z'- plied": "'z'-plied",
                                    }

        dic_patterns = {r'(\d),\s+(\d)': r'\1,\2'}

        with open(self.models_path + 'aat_id2wikidata.json') as d:
            dictData = json.load(d)
        aat_id2wikidata = dict((k, v) for k, v in dictData.items())
        
        self.predictions_df = get_predictions_elq_df(self.predictions, 
                                                string_cases_dictionary, dic_patterns, aat_id2wikidata, self.prediction_type)
        
        if save_path:
            self.predictions_df.to_csv(save_path)
            cprint(f"Annotations (DF) saved: {save_path}", "green")
        
        return self.predictions_df
