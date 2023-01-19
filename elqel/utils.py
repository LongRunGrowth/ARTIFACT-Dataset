import collections
import json
import pandas as pd
import re
import unicodedata
from collections import OrderedDict
from tqdm import tqdm


def clean_string(string, cases_dic):
    """Removes special charachters from string"""
    for case in cases_dic.keys():
        string = string.replace(case, cases_dic[case])
    return string

def chunk_list(l, n):
    """Iterate a list in chunks"""
    print("Creating chunks...")
    for i in tqdm(range(0, len(l), n)):
        yield l[i:i + n]

def get_predictions_elq_df(predictions_list, cases_dict, dict_patterns, aat_id2wikidata, pred_type):
    pred_df = pd.DataFrame()
    for pred in predictions_list:
        if pred_type == "unique":
            df_subdf = get_tag_position(pred, cases_dict, dict_patterns, aat_id2wikidata)
        elif pred_type == "multiple":
            df_subdf = get_tag_position_multiple(pred, cases_dict, dict_patterns, aat_id2wikidata)
        pred_df = pred_df.append(df_subdf, ignore_index=True)

    return pred_df

def get_tag_position(pred, cases_dict, patterns_dict, aat_id2wikidata_dict):
    pred_id = pred["id"]
    text_orig = pred['text']
    text = strip_accents(text_orig.lower())  # remove accents

    # get tagged items list
    tagged_items = []
    wiki_id_items = []

    for tuple_ in pred['pred_tuples_string']:
        tagged_items.append(tuple_[1])
    for tuple_ in pred['pred_triples']:
        wiki_id_items.append(tuple_[0])

    # adjust strings
    tagged_items = [clean_string(item, cases_dict)
                    for item in tagged_items]  # remove special cases
    # adjust special patterns
    tagged_items = [replace_patterns(item, patterns_dict)
                    for item in tagged_items]  # remove special cases

    # to extract aat after
    pred_tuple_str_wiki = zip(tagged_items, wiki_id_items)

    # check for repeated items
    count_tags = {}  # to store tag and reps
    for item in tagged_items:
        count_tags[item] = {}

    for item, count in collections.Counter(tagged_items).items():
        count_tags[item]["count"] = count
        count_tags[item]["aat_list"] = []

    # assing aat prediction
    for tuple_ in pred_tuple_str_wiki:
        item = tuple_[0]
        aat = aat_id2wikidata_dict[tuple_[1]]
        count_tags[item]["aat_list"].append(aat)

    # map text position
    tag_lst = []
    tag_start_lst = []
    tag_end_lst = []
    aat_list = []
    for item in count_tags.keys():
        len_item = len(item)
        if count_tags[item]["count"] == 1:
            tag_lst.append(item)
            pos_0 = text.find(item)
            pos_1 = pos_0 + len_item
            tag_start_lst.append(pos_0)
            tag_end_lst.append(pos_1)
            aat_list.append(count_tags[item]["aat_list"][0])
        elif count_tags[item]["count"] > 1:
            for app in range(count_tags[item]["count"]):
                aat_list.append(count_tags[item]["aat_list"][app])
                if app == 0:
                    tag_lst.append(item)
                    pos_0 = text.find(item)
                    pos_1 = pos_0 + len_item
                    tag_start_lst.append(pos_0)
                    tag_end_lst.append(pos_1)
                else:
                    tag_lst.append(item)
                    pos_0 = text.find(item, pos_1)
                    pos_1 = pos_0 + len_item
                    tag_start_lst.append(pos_0)
                    tag_end_lst.append(pos_1)

    pred_tag_pos_dic = {"id": [pred_id]*len(tagged_items),
                        "text": [text]*len(tagged_items),
                        "chunk_text": tag_lst,
                        "chunk_start": tag_start_lst,
                        "chunk_end": tag_end_lst,
                        "aat": aat_list}

    sub_pred_df = pd.DataFrame(pred_tag_pos_dic)
    sub_pred_df.sort_values(by=['chunk_start'], inplace=True)
    sub_pred_df["chunk_start"] = sub_pred_df["chunk_start"].astype(int)
    sub_pred_df["chunk_end"] = sub_pred_df["chunk_end"].astype(int)
    
    sub_pred_df = map_special_cases(sub_pred_df)

    return sub_pred_df

def get_tag_position_multiple(pred, cases_dict, patterns_dict, aat_id2wikidata_dict):
    pred_id = pred["id"]
    text_orig = pred['text']
    text = strip_accents(text_orig.lower())  # remove accents
        
    # get tagged items list
    tagged_items = []
    best_score_items = []
    wiki_id_items = []
    all_candidates = []
    wiki_id_all_candidates = []
    all_scores = []

    for tuple_ in pred['pred_tuples_string']:
        tagged_items.append(tuple_[1])
        all_candidates.append(tuple_[0])
    for tuple_ in pred['pred_triples']:
        wiki_id_items.append(tuple_[0][0])
        wiki_id_all_candidates.append(tuple_[0])
    for tuple_ in pred['scores']:
        all_scores.append(tuple_.tolist())
        best_score_items.append(tuple_.tolist()[0])
    
    aat_all_candidates_list = []
    for cand_list in wiki_id_all_candidates: 
        aat_all_candidates_list.append([aat_id2wikidata_dict[cand_wikiid] for cand_wikiid in cand_list])

    # candidates dictionary 
    cands_dict_list = []
    
    for idx in range(len(all_candidates)):
        cands_dict = OrderedDict()
        for cand, score, aat in zip(all_candidates[idx], all_scores[idx], aat_all_candidates_list[idx]):
            cands_dict[str(aat)] = (str(cand), float(score))

        cands_dict_list.append(dict(cands_dict))

    # adjust strings
    tagged_items = [clean_string(item, cases_dict)
                    for item in tagged_items]  # remove special cases
    # adjust special patterns
    tagged_items = [replace_patterns(item, patterns_dict)
                    for item in tagged_items]  # remove special cases

    # to extract aat after
    pred_tuple_str_wiki = zip(tagged_items, wiki_id_items, best_score_items, cands_dict_list, all_candidates)
    
    # check for repeated items
    count_tags = {}  # to store tag and reps
    for item in tagged_items:
        count_tags[item] = {}

    for item, count in collections.Counter(tagged_items).items():
        count_tags[item]["count"] = count
        count_tags[item]["aat_list"] = []
        count_tags[item]["aat_str_list"] = []
        count_tags[item]["bst_score_list"] = []
        count_tags[item]["cands_list"] = []
        
    # assign aat prediction
    for tuple_ in pred_tuple_str_wiki:
        item = tuple_[0]
        aat = aat_id2wikidata_dict[tuple_[1]]
        count_tags[item]["aat_list"].append(aat)
        count_tags[item]["aat_str_list"].append(tuple_[4][0])
        count_tags[item]["bst_score_list"].append(tuple_[2])
        count_tags[item]["cands_list"].append(tuple_[3])
        
    # map text position
    tag_lst = []
    tag_start_lst = []
    tag_end_lst = []
    aat_list = []
    best_score_list = []
    candidates_list = []
    aat_str_list = []
    
    for item in count_tags.keys():
        len_item = len(item)
        if count_tags[item]["count"] == 1:
            tag_lst.append(item)
            pos_0 = text.find(item)
            pos_1 = pos_0 + len_item
            tag_start_lst.append(pos_0)
            tag_end_lst.append(pos_1)
            aat_list.append(count_tags[item]["aat_list"][0])
            best_score_list.append(count_tags[item]["bst_score_list"][0])
            candidates_list.append(count_tags[item]["cands_list"][0])
            aat_str_list.append(count_tags[item]["aat_str_list"][0])
            
        elif count_tags[item]["count"] > 1:
            for app in range(count_tags[item]["count"]):
                aat_list.append(count_tags[item]["aat_list"][app])
                
                if app == 0:
                    tag_lst.append(item)
                    pos_0 = text.find(item)
                    pos_1 = pos_0 + len_item
                    tag_start_lst.append(pos_0)
                    tag_end_lst.append(pos_1)
                else:
                    tag_lst.append(item)
                    pos_0 = text.find(item, pos_1)
                    pos_1 = pos_0 + len_item
                    tag_start_lst.append(pos_0)
                    tag_end_lst.append(pos_1)
                    
                best_score_list.append(count_tags[item]["bst_score_list"][app])
                candidates_list.append(count_tags[item]["cands_list"][app])  
                aat_str_list.append(count_tags[item]["aat_str_list"][app])
    
    pred_tag_pos_dic = {"id": [pred_id]*len(tagged_items),
                        "text": [text]*len(tagged_items),
                        "chunk_text": tag_lst,
                        "chunk_start": tag_start_lst,
                        "chunk_end": tag_end_lst,
                        "aat": aat_list,
                        "aat_str": aat_str_list,
                        "best_score": best_score_list,
                        "candidates": candidates_list,
                        }
    
    sub_pred_df = pd.DataFrame(pred_tag_pos_dic)
    sub_pred_df.sort_values(by=['chunk_start'], inplace=True)
    sub_pred_df["chunk_start"] = sub_pred_df["chunk_start"].astype(int)
    sub_pred_df["chunk_end"] = sub_pred_df["chunk_end"].astype(int)
    
    sub_pred_df = map_special_cases(sub_pred_df)

    return sub_pred_df

def replace_patterns(string, dict_patterns):
    for case in dict_patterns.keys():
        string = re.sub(case, dict_patterns[case], string)
    return string


def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

def map_special_cases(df):
    for idx, row in df.iterrows():
        if row['chunk_text'] =='bronze age':
            df.at[idx,'aat'] = '300019275'
        elif row['chunk_text'] =='iron age':
            df.at[idx,'aat'] = '300019279'
        elif row['chunk_text'] =='stone age':
            df.at[idx,'aat'] = '300106724' 
        
    return df
