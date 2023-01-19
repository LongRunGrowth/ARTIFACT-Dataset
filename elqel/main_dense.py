# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
import unicodedata
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import BertTokenizer

from elqel.index.faiss_indexer import DenseFlatIndexer, DenseHNSWFlatIndexer, DenseIVFFlatIndexer
from elqel.biencoder.biencoder import load_biencoder
from elqel.biencoder.data_process import process_mention_data
from elqel.vcg_utils.measures import entity_linking_tp_with_overlap

models_path = "./models/"

HIGHLIGHTS = [
    "on_red",
    "on_green",
    "on_yellow",
    "on_blue",
    "on_magenta",
    "on_cyan",
]

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def _load_candidates(
    entity_catalogue, entity_encoding,
    faiss_index="none", index_path=None,
    logger=1,
):

    """ Loads (indexed) knowledge base"""

    if faiss_index == "none":
        candidate_encoding = torch.load(entity_encoding)
        indexer = None
    else:
        candidate_encoding = None
        assert index_path is not None, "Error! Empty indexer path."
        if faiss_index == "flat":
            indexer = DenseFlatIndexer(1)
        elif faiss_index == "hnsw":
            indexer = DenseHNSWFlatIndexer(1)
        elif faiss_index == "ivfflat":
            indexer = DenseIVFFlatIndexer(1)
        else:
            raise ValueError("Error! Unsupported indexer type! Choose from flat,hnsw,ivfflat.")
        indexer.deserialize_from(index_path)

    candidate_encoding = torch.load(entity_encoding)

    if not os.path.exists(f"{models_path}aat_id2title.json"):
        id2title = {}
        id2text = {}
        id2wikidata = {}
        local_idx = 0
        with open(entity_catalogue, "r") as fin:
            lines = fin.readlines()
            for line in lines:
                entity = json.loads(line)
                id2title[str(local_idx)] = entity["title"]
                id2text[str(local_idx)] = entity["text"]
                if "aat_id" in entity:
                    id2wikidata[str(local_idx)] = entity["aat_id"]
                local_idx += 1
        json.dump(id2title, open(f"{models_path}aat_id2title.json", "w"))
        json.dump(id2text, open(f"{models_path}aat_id2text.json", "w"))
        json.dump(id2wikidata, open(f"{models_path}aat_id2wikidata.json", "w"))
    else:
        if logger: print("Loading id2title")
        id2title = json.load(open(f"{models_path}aat_id2title.json"))
        if logger: print("Loading id2text")
        id2text = json.load(open(f"{models_path}aat_id2text.json"))
        id2text[-1] = "nan"
        if logger: print("Loading id2wikidata")
        id2wikidata = json.load(open(f"{models_path}aat_id2wikidata.json"))

    return (
        candidate_encoding, indexer, 
        id2title, id2text, id2wikidata,
    )


def _process_biencoder_dataloader(samples, tokenizer, biencoder_params, logger):
    """
    Samples: list of examples, each of the form--

    IF HAVE LABELS
    {
        "id": "WebQTest-12",
        "text": "who is governor of ohio 2011?",
        "mentions": [[19, 23], [7, 15]],
        "tokenized_text_ids": [2040, 2003, 3099, 1997, 4058, 2249, 1029],
        "tokenized_mention_idxs": [[4, 5], [2, 3]],
        "label_id": [10902, 28422],
        "wikidata_id": ["Q1397", "Q132050"],
        "entity": ["Ohio", "Governor"],
        "label": [list of wikipedia descriptions]
    }

    IF NO LABELS (JUST PREDICTION)
    {
        "id": "WebQTest-12",
        "text": "who is governor of ohio 2011?",
    }
    """
    if 'label_id' in samples[0]:
        # have labels
        tokens_data, tensor_data_tuple, _ = process_mention_data(
            samples=samples,
            tokenizer=tokenizer,
            max_context_length=biencoder_params["max_context_length"],
            max_cand_length=biencoder_params["max_cand_length"],
            silent=False,
            logger=logger,
            debug=biencoder_params["debug"],
            add_mention_bounds=(not biencoder_params.get("no_mention_bounds", False)),
            params=biencoder_params,
        )
    else:
        samples_text_tuple = []
        max_seq_len = 0
        for sample in samples:
            samples_text_tuple
            # truncate the end if the sequence is too long...
            encoded_sample = [101] + tokenizer.encode(sample['text'])[:biencoder_params["max_context_length"]-2] + [102]
            max_seq_len = max(len(encoded_sample), max_seq_len)
            samples_text_tuple.append(encoded_sample + [0 for _ in range(biencoder_params["max_context_length"] - len(encoded_sample))])

            # print(samples_text_tuple)

        tensor_data_tuple = [torch.tensor(samples_text_tuple)]
    tensor_data = TensorDataset(*tensor_data_tuple)
    sampler = SequentialSampler(tensor_data)
    dataloader = DataLoader(
        tensor_data, sampler=sampler, batch_size=biencoder_params["eval_batch_size"]
    )
    return dataloader


def _run_biencoder(
    args, biencoder, dataloader, candidate_encoding, samples,
    num_cand_mentions=50, num_cand_entities=10,
    device="cpu", sample_to_all_context_inputs=None,
    threshold=0.0, indexer=None,
):
    """
    Returns: tuple
        labels (List[int]) [(max_num_mentions_gold) x exs]: gold labels -- returns None if no labels
        nns (List[Array[int]]) [(# of pred mentions, cands_per_mention) x exs]: predicted entity IDs in each example
        dists (List[Array[float]]) [(# of pred mentions, cands_per_mention) x exs]: scores of each entity in nns
        pred_mention_bounds (List[Array[int]]) [(# of pred mentions, 2) x exs]: predicted mention boundaries in each examples
        mention_scores (List[Array[float]]) [(# of pred mentions,) x exs]: mention score logit
        cand_scores (List[Array[float]]) [(# of pred mentions, cands_per_mention) x exs]: candidate score logit
    """
    biencoder.model.eval()
    biencoder_model = biencoder.model
    if hasattr(biencoder.model, "module"):
        biencoder_model = biencoder.model.module

    context_inputs = []
    nns = []
    dists = []
    mention_dists = []
    pred_mention_bounds = []
    mention_scores = []
    cand_scores = []
    sample_idx = 0
    ctxt_idx = 0
    label_ids = None
    for step, batch in enumerate(tqdm(dataloader)):
        context_input = batch[0].to(device)
        mask_ctxt = context_input != biencoder.NULL_IDX
        with torch.no_grad():
            context_outs = biencoder.encode_context(
                context_input, num_cand_mentions=num_cand_mentions, topK_threshold=threshold,
            )
            embedding_ctxt = context_outs['mention_reps']
            left_align_mask = context_outs['mention_masks']
            chosen_mention_logits = context_outs['mention_logits']
            chosen_mention_bounds = context_outs['mention_bounds']

            '''
            GET TOP CANDIDATES PER MENTION
            '''
            # (all_pred_mentions_batch, embed_dim)
            embedding_ctxt = embedding_ctxt[left_align_mask]
            if indexer is None:
                try:
                    cand_logits, _, _ = biencoder.score_candidate(
                        context_input, None,
                        text_encs=embedding_ctxt,
                        cand_encs=candidate_encoding.to(device),
                    )
                    # DIM (all_pred_mentions_batch, num_cand_entities); (all_pred_mentions_batch, num_cand_entities)
                    top_cand_logits_shape, top_cand_indices_shape = cand_logits.topk(num_cand_entities, dim=-1, sorted=True)
                except:
                    # for memory savings, go through one chunk of candidates at a time
                    SPLIT_SIZE=1000000
                    done=False
                    while not done:
                        top_cand_logits_list = []
                        top_cand_indices_list = []
                        max_chunk = int(len(candidate_encoding) / SPLIT_SIZE)
                        for chunk_idx in range(max_chunk):
                            try:
                                # DIM (num_total_mentions, num_cand_entities); (num_total_mention, num_cand_entities)
                                top_cand_logits, top_cand_indices = embedding_ctxt.mm(candidate_encoding[chunk_idx*SPLIT_SIZE:(chunk_idx+1)*SPLIT_SIZE].to(device).t().contiguous()).topk(10, dim=-1, sorted=True)
                                top_cand_logits_list.append(top_cand_logits)
                                top_cand_indices_list.append(top_cand_indices + chunk_idx*SPLIT_SIZE)
                                if len((top_cand_indices_list[chunk_idx] < 0).nonzero()) > 0:
                                    import pdb
                                    pdb.set_trace()
                            except:
                                SPLIT_SIZE = int(SPLIT_SIZE/2)
                                break
                        if len(top_cand_indices_list) == max_chunk:
                            # DIM (num_total_mentions, num_cand_entities); (num_total_mentions, num_cand_entities) -->
                            #       top_top_cand_indices_shape indexes into top_cand_indices
                            top_cand_logits_shape, top_top_cand_indices_shape = torch.cat(
                                top_cand_logits_list, dim=-1).topk(num_cand_entities, dim=-1, sorted=True)
                            # make indices index into candidate_encoding
                            # DIM (num_total_mentions, max_chunk*num_cand_entities)
                            all_top_cand_indices = torch.cat(top_cand_indices_list, dim=-1)
                            # DIM (num_total_mentions, num_cand_entities)
                            top_cand_indices_shape = all_top_cand_indices.gather(-1, top_top_cand_indices_shape)
                            done = True
            else:
                # DIM (all_pred_mentions_batch, num_cand_entities); (all_pred_mentions_batch, num_cand_entities)
                top_cand_logits_shape, top_cand_indices_shape = indexer.search_knn(embedding_ctxt.cpu().numpy(), num_cand_entities)
                top_cand_logits_shape = torch.tensor(top_cand_logits_shape).to(embedding_ctxt.device)
                top_cand_indices_shape = torch.tensor(top_cand_indices_shape).to(embedding_ctxt.device)

            # DIM (bs, max_num_pred_mentions, num_cand_entities)
            top_cand_logits = torch.zeros(chosen_mention_logits.size(0), chosen_mention_logits.size(1), top_cand_logits_shape.size(-1)).to(
                top_cand_logits_shape.device, top_cand_logits_shape.dtype)
            top_cand_logits[left_align_mask] = top_cand_logits_shape
            top_cand_indices = torch.zeros(chosen_mention_logits.size(0), chosen_mention_logits.size(1), top_cand_indices_shape.size(-1)).to(
                top_cand_indices_shape.device, top_cand_indices_shape.dtype)
            top_cand_indices[left_align_mask] = top_cand_indices_shape

            '''
            COMPUTE FINAL SCORES FOR EACH CAND-MENTION PAIR + PRUNE USING IT
            '''
            # Has NAN for impossible mentions...
            # log p(entity && mb) = log [p(entity|mention bounds) * p(mention bounds)] = log p(e|mb) + log p(mb)
            # DIM (bs, max_num_pred_mentions, num_cand_entities)
            scores = torch.log_softmax(top_cand_logits, -1) + torch.sigmoid(chosen_mention_logits.unsqueeze(-1)).log()

            '''
            DON'T NEED TO RESORT BY NEW SCORE -- DISTANCE PRESERVING (largest entity score still be largest entity score)
            '''
    
            for idx in range(len(batch[0])):
                # [(seqlen) x exs] <= (bsz, seqlen)
                context_inputs.append(context_input[idx][mask_ctxt[idx]].data.cpu().numpy())
                # [(max_num_mentions, cands_per_mention) x exs] <= (bsz, max_num_mentions=num_cand_mentions, cands_per_mention)
                nns.append(top_cand_indices[idx][left_align_mask[idx]].data.cpu().numpy())
                # [(max_num_mentions, cands_per_mention) x exs] <= (bsz, max_num_mentions=num_cand_mentions, cands_per_mention)
                dists.append(scores[idx][left_align_mask[idx]].data.cpu().numpy())
                # [(max_num_mentions, 2) x exs] <= (bsz, max_num_mentions=num_cand_mentions, 2)
                pred_mention_bounds.append(chosen_mention_bounds[idx][left_align_mask[idx]].data.cpu().numpy())
                # [(max_num_mentions,) x exs] <= (bsz, max_num_mentions=num_cand_mentions)
                mention_scores.append(chosen_mention_logits[idx][left_align_mask[idx]].data.cpu().numpy())
                # [(max_num_mentions, cands_per_mention) x exs] <= (bsz, max_num_mentions=num_cand_mentions, cands_per_mention)
                cand_scores.append(top_cand_logits[idx][left_align_mask[idx]].data.cpu().numpy())

    return nns, dists, pred_mention_bounds, mention_scores, cand_scores


def get_predictions(
    args, dataloader, biencoder_params, samples, nns, dists, mention_scores, cand_scores,
    pred_mention_bounds, id2title, threshold=-2.9, mention_threshold=-0.6931,
):
    """
    Arguments:
        args, dataloader, biencoder_params, samples, nns, dists, pred_mention_bounds
    Returns:
        all_entity_preds,
        num_correct_weak, num_correct_strong, num_predicted, num_gold,
        num_correct_weak_from_input_window, num_correct_strong_from_input_window, num_gold_from_input_window
    """

    # save biencoder predictions and print precision/recalls
    num_correct_weak = 0
    num_correct_strong = 0
    num_predicted = 0
    num_gold = 0
    num_correct_weak_from_input_window = 0
    num_correct_strong_from_input_window = 0
    num_gold_from_input_window = 0
    all_entity_preds = []

    f = errors_f = None
    if getattr(args, 'save_preds_dir', None) is not None:
        save_biencoder_file = os.path.join(args.save_preds_dir, 'biencoder_outs.jsonl')
        f = open(save_biencoder_file, 'w')
        errors_f = open(os.path.join(args.save_preds_dir, 'biencoder_errors.jsonl'), 'w')

    # nns (List[Array[int]]) [(num_pred_mentions, cands_per_mention) x exs])
    # dists (List[Array[float]]) [(num_pred_mentions, cands_per_mention) x exs])
    # pred_mention_bounds (List[Array[int]]) [(num_pred_mentions, 2) x exs]
    # cand_scores (List[Array[float]]) [(num_pred_mentions, cands_per_mention) x exs])
    # mention_scores (List[Array[float]]) [(num_pred_mentions,) x exs])
    for batch_num, batch_data in enumerate(dataloader):
        batch_context = batch_data[0]
        if len(batch_data) > 1:
            _, batch_cands, batch_label_ids, batch_mention_idxs, batch_mention_idx_masks = batch_data
        for b in range(len(batch_context)):
            i = batch_num * biencoder_params['eval_batch_size'] + b
            sample = samples[i]
            input_context = batch_context[b][batch_context[b] != 0].tolist()  # filter out padding

            # (num_pred_mentions, cands_per_mention)
            scores = dists[i] if args.threshold_type == "joint" else cand_scores[i]
            cands_mask = (scores[:,0] == scores[:,0])
            pred_entity_list = nns[i][cands_mask]
            if len(pred_entity_list) > 0:
                e_id = pred_entity_list[0]
            distances = scores[cands_mask]
            # (num_pred_mentions, 2)
            entity_mention_bounds_idx = pred_mention_bounds[i][cands_mask]
            utterance = sample['text']

            if args.threshold_type == "joint":
                # THRESHOLDING
                assert utterance is not None
                top_mentions_mask = (distances[:,0] > threshold)
            elif args.threshold_type == "top_entity_by_mention":
                top_mentions_mask = (mention_scores[i] > mention_threshold)
            elif args.threshold_type == "thresholded_entity_by_mention":
                top_mentions_mask = (distances[:,0] > threshold) & (mention_scores[i] > mention_threshold)
    
            _, sort_idxs = torch.tensor(distances[:,0][top_mentions_mask]).sort(descending=True)
            # cands already sorted by score
            all_pred_entities = pred_entity_list[:,0][top_mentions_mask]
            e_mention_bounds = entity_mention_bounds_idx[top_mentions_mask]
            chosen_distances = distances[:,0][top_mentions_mask]
            if len(all_pred_entities) >= 2:
                all_pred_entities = all_pred_entities[sort_idxs]
                e_mention_bounds = e_mention_bounds[sort_idxs]
                chosen_distances = chosen_distances[sort_idxs]

            # prune mention overlaps
            e_mention_bounds_pruned = []
            all_pred_entities_pruned = []
            chosen_distances_pruned = []
            mention_masked_utterance = np.zeros(len(input_context))
            # ensure well-formed-ness, prune overlaps
            # greedily pick highest scoring, then prune all overlapping
            for idx, mb in enumerate(e_mention_bounds):
                mb[1] += 1  # prediction was inclusive, now make exclusive
                # check if in existing mentions
                if args.threshold_type != "top_entity_by_mention" and mention_masked_utterance[mb[0]:mb[1]].sum() >= 1:
                    continue
                e_mention_bounds_pruned.append(mb)
                all_pred_entities_pruned.append(all_pred_entities[idx])
                chosen_distances_pruned.append(float(chosen_distances[idx]))
                mention_masked_utterance[mb[0]:mb[1]] = 1

            input_context = input_context[1:-1]  # remove BOS and sep
            pred_triples = [(
                str(all_pred_entities_pruned[j]),
                int(e_mention_bounds_pruned[j][0]) - 1,  # -1 for BOS
                int(e_mention_bounds_pruned[j][1]) - 1,
            ) for j in range(len(all_pred_entities_pruned))]

            entity_results = {
                "id": sample["id"],
                "text": sample["text"],
                "scores": chosen_distances_pruned,
            }

            if 'label_id' in sample:
                # Get LABELS
                input_mention_idxs = batch_mention_idxs[b][batch_mention_idx_masks[b]].tolist()
                input_label_ids = batch_label_ids[b][batch_label_ids[b] != -1].tolist()
                    

                # assert len(input_label_ids) == len(input_mention_idxs)
                gold_mention_bounds = [
                    sample['text'][ment[0]-10:ment[0]] + "[" + sample['text'][ment[0]:ment[1]] + "]" + sample['text'][ment[1]:ment[1]+10]
                    for ment in sample['mentions']
                ]

                # GET ALIGNED MENTION_IDXS (input is slightly different to model) between ours and gold labels -- also have to account for BOS
                gold_input = sample['tokenized_text_ids']
                # return first instance of my_input in gold_input
                for my_input_start in range(len(gold_input)):
                    if (
                        gold_input[my_input_start] == input_context[0] and
                        gold_input[my_input_start:my_input_start+len(input_context)] == input_context
                    ):
                        break

                # add alignment factor (my_input_start) to predicted mention triples
                pred_triples = [(
                    triple[0],
                    triple[1] + my_input_start, triple[2] + my_input_start,
                ) for triple in pred_triples]
                gold_triples = [(
                    str(sample['label_id'][j]),
                    sample['tokenized_mention_idxs'][j][0], sample['tokenized_mention_idxs'][j][1],
                ) for j in range(len(sample['label_id']))]
                num_overlap_weak, num_overlap_strong = entity_linking_tp_with_overlap(gold_triples, pred_triples)
                num_correct_weak += num_overlap_weak
                num_correct_strong += num_overlap_strong
                num_predicted += len(all_pred_entities_pruned)
                num_gold += len(sample["label_id"])

                # compute number correct given the input window
                pred_input_window_triples = [(
                    str(all_pred_entities_pruned[j]),
                    int(e_mention_bounds_pruned[j][0]), int(e_mention_bounds_pruned[j][1]),
                ) for j in range(len(all_pred_entities_pruned))]
                gold_input_window_triples = [(
                    str(input_label_ids[j]),
                    input_mention_idxs[j][0], input_mention_idxs[j][1] + 1,
                ) for j in range(len(input_label_ids))]
                num_overlap_weak_window, num_overlap_strong_window = entity_linking_tp_with_overlap(gold_input_window_triples, pred_input_window_triples)
                num_correct_weak_from_input_window += num_overlap_weak_window
                num_correct_strong_from_input_window += num_overlap_strong_window
                num_gold_from_input_window += len(input_mention_idxs)

                entity_results.update({
                    "pred_tuples_string": [
                        [id2title[triple[0]], tokenizer.decode(sample['tokenized_text_ids'][triple[1]:triple[2]])]
                        for triple in pred_triples
                    ],
                    "gold_tuples_string": [
                        [id2title[triple[0]], tokenizer.decode(sample['tokenized_text_ids'][triple[1]:triple[2]])]
                        for triple in gold_triples
                    ],
                    "pred_triples": pred_triples,
                    "gold_triples": gold_triples,
                    "tokens": input_context,
                })

                if errors_f is not None and (num_overlap_weak != len(gold_triples) or num_overlap_weak != len(pred_triples)):
                    errors_f.write(json.dumps(entity_results) + "\n")
            else:
                entity_results.update({
                    "pred_tuples_string": [
                        [id2title[triple[0]], tokenizer.decode(input_context[triple[1]:triple[2]])]
                        for triple in pred_triples
                    ],
                    "pred_triples": pred_triples,
                    "tokens": input_context,
                })

            all_entity_preds.append(entity_results)
            if f is not None:
                f.write(
                    json.dumps(entity_results) + "\n"
                )
    
    if f is not None:
        f.close()
        errors_f.close()
      
    # print(
    #     all_entity_preds, num_correct_weak, num_correct_strong, num_predicted, num_gold,
    #     num_correct_weak_from_input_window, num_correct_strong_from_input_window, num_gold_from_input_window
    # )
    return (
        all_entity_preds, num_correct_weak, num_correct_strong, num_predicted, num_gold,
        num_correct_weak_from_input_window, num_correct_strong_from_input_window, num_gold_from_input_window
    )

def get_predictions_multiple(
    args, dataloader, biencoder_params, samples, nns, dists, mention_scores, cand_scores,
    pred_mention_bounds, id2title, threshold=-2.9, mention_threshold=-0.6931,
):
    """
    Arguments:
        args, dataloader, biencoder_params, samples, nns, dists, pred_mention_bounds
    Returns:
        all_entity_preds,
        num_correct_weak, num_correct_strong, num_predicted, num_gold,
        num_correct_weak_from_input_window, num_correct_strong_from_input_window, num_gold_from_input_window
    """

    # save biencoder predictions and print precision/recalls
    num_correct_weak = 0
    num_correct_strong = 0
    num_predicted = 0
    num_gold = 0
    num_correct_weak_from_input_window = 0
    num_correct_strong_from_input_window = 0
    num_gold_from_input_window = 0
    all_entity_preds = []

    f = errors_f = None
    if getattr(args, 'save_preds_dir', None) is not None:
        save_biencoder_file = os.path.join(args.save_preds_dir, 'biencoder_outs.jsonl')
        f = open(save_biencoder_file, 'w')
        errors_f = open(os.path.join(args.save_preds_dir, 'biencoder_errors.jsonl'), 'w')

    # nns (List[Array[int]]) [(num_pred_mentions, cands_per_mention) x exs])
    # dists (List[Array[float]]) [(num_pred_mentions, cands_per_mention) x exs])
    # pred_mention_bounds (List[Array[int]]) [(num_pred_mentions, 2) x exs]
    # cand_scores (List[Array[float]]) [(num_pred_mentions, cands_per_mention) x exs])
    # mention_scores (List[Array[float]]) [(num_pred_mentions,) x exs])
    for batch_num, batch_data in enumerate(dataloader):
        batch_context = batch_data[0]
        if len(batch_data) > 1:
            _, batch_cands, batch_label_ids, batch_mention_idxs, batch_mention_idx_masks = batch_data
        for b in range(len(batch_context)):
            i = batch_num * biencoder_params['eval_batch_size'] + b
            sample = samples[i]
            input_context = batch_context[b][batch_context[b] != 0].tolist()  # filter out padding

            # (num_pred_mentions, cands_per_mention)
            scores = dists[i] if args.threshold_type == "joint" else cand_scores[i]
            cands_mask = (scores[:,0] == scores[:,0])
            pred_entity_list = nns[i][cands_mask]
            if len(pred_entity_list) > 0:
                e_id = pred_entity_list[0]
            distances = scores[cands_mask]
            # (num_pred_mentions, 2)
            entity_mention_bounds_idx = pred_mention_bounds[i][cands_mask]
            utterance = sample['text']

            if args.threshold_type == "joint":
                # THRESHOLDING
                assert utterance is not None
                top_mentions_mask = (distances[:,0] > threshold)
            elif args.threshold_type == "top_entity_by_mention":
                top_mentions_mask = (mention_scores[i] > mention_threshold)
            elif args.threshold_type == "thresholded_entity_by_mention":
                top_mentions_mask = (distances[:,0] > threshold) & (mention_scores[i] > mention_threshold)

            # count the number of valid predictions (i.e offset the threshold)
            valid_top_mentions_mask = sum(top_mentions_mask)
            
            # If found valid predictions
            if valid_top_mentions_mask > 0:
                sorted_dist , sort_idxs = torch.tensor(distances).sort(descending=True)

                assert torch.equal(sorted_dist, torch.tensor(distances))
                # cands already sorted by score
                all_pred_entities = pred_entity_list
                e_mention_bounds = entity_mention_bounds_idx
                chosen_distances = distances

                # if len(all_pred_entities) >= 2:
                #     all_pred_entities = all_pred_entities[sort_idxs]
                #     e_mention_bounds = e_mention_bounds[sort_idxs]
                #     chosen_distances = chosen_distances[sort_idxs]

                # prune mention overlaps
                e_mention_bounds_pruned = []
                all_pred_entities_pruned = []
                chosen_distances_pruned = []
                mention_masked_utterance = np.zeros(len(input_context))
                # ensure well-formed-ness, prune overlaps
                # greedily pick highest scoring, then prune all overlapping
                for idx, mb in enumerate(e_mention_bounds):
                    mb[1] += 1  # prediction was inclusive, now make exclusive
                    # check if in existing mentions
                    if args.threshold_type != "top_entity_by_mention" and mention_masked_utterance[mb[0]:mb[1]].sum() >= 1:
                        continue
                    e_mention_bounds_pruned.append(mb)
                    all_pred_entities_pruned.append(all_pred_entities[idx])
                    chosen_distances_pruned.append(chosen_distances[idx])
                    mention_masked_utterance[mb[0]:mb[1]] = 1


                #print('###################')

                input_context = input_context[1:-1]  # remove BOS and sep
                pred_triples = [(
                    [str(p) for p in all_pred_entities_pruned[j]],
                    int(e_mention_bounds_pruned[j][0]) - 1,  # -1 for BOS
                    int(e_mention_bounds_pruned[j][1]) - 1,
                ) for j in range(len(all_pred_entities_pruned))]



                #print('###################')
                #print(pred_triples)

                entity_results = {
                    "id": sample["id"],
                    "text": sample["text"],
                    "scores": chosen_distances_pruned,
                }


                entity_results.update({
                    "pred_tuples_string": [
                        [[id2title[t] for t in triple[0]], tokenizer.decode(input_context[triple[1]:triple[2]])]
                        for triple in pred_triples
                    ],
                    "pred_triples": pred_triples,
                    "tokens": input_context,
                })

                all_entity_preds.append(entity_results)
                if f is not None:
                    f.write(
                        json.dumps(entity_results) + "\n"
                    )

            # If not found valid predictions
            else:
                None
                
    if f is not None:
        f.close()
        errors_f.close()
    return (
        all_entity_preds, num_correct_weak, num_correct_strong, num_predicted, num_gold,
        num_correct_weak_from_input_window, num_correct_strong_from_input_window, num_gold_from_input_window
    )
            
            

def display_metrics(
    num_correct, num_predicted, num_gold, prefix="",
):
    p = 0 if num_predicted == 0 else float(num_correct) / float(num_predicted)
    r = 0 if num_gold == 0 else float(num_correct) / float(num_gold)
    if p + r > 0:
        f1 = 2 * p * r / (p + r)
    else:
        f1 = 0
    print("{0}precision = {1} / {2} = {3}".format(prefix, num_correct, num_predicted, p))
    print("{0}recall = {1} / {2} = {3}".format(prefix, num_correct, num_gold, r))
    print("{0}f1 = {1}".format(prefix, f1))

def load_models(args, logger=1):
    # load biencoder model
    if logger: print("Loading biencoder model")
    try:
        with open(args.biencoder_config) as json_file:
            biencoder_params = json.load(json_file)
    except json.decoder.JSONDecodeError:
        with open(args.biencoder_config) as json_file:
            for line in json_file:
                line = line.replace("'", "\"")
                line = line.replace("True", "true")
                line = line.replace("False", "false")
                line = line.replace("None", "null")
                biencoder_params = json.loads(line)
                break
    biencoder_params["path_to_model"] = args.biencoder_model
    biencoder_params["cand_token_ids_path"] = args.cand_token_ids_path
    biencoder_params["eval_batch_size"] = getattr(args, 'eval_batch_size', 8)
    biencoder_params["no_cuda"] = (not getattr(args, 'use_cuda', False) or not torch.cuda.is_available())
    if biencoder_params["no_cuda"]:
        biencoder_params["data_parallel"] = False
    biencoder_params["load_cand_enc_only"] = False
    if getattr(args, 'max_context_length', None) is not None:
        biencoder_params["max_context_length"] = args.max_context_length
    biencoder = load_biencoder(biencoder_params)
    if biencoder_params["no_cuda"] and type(biencoder.model).__name__ == 'DataParallel':
        biencoder.model = biencoder.model.module
    elif not biencoder_params["no_cuda"] and type(biencoder.model).__name__ != 'DataParallel':
        biencoder.model = torch.nn.DataParallel(biencoder.model)

    # load candidate entities
    if logger: print("Loading candidate entities")

    (
        candidate_encoding,
        indexer,
        id2title,
        id2text,
        id2wikidata,
    ) = _load_candidates(
        args.entity_catalogue, args.entity_encoding,
        args.faiss_index, args.index_path, logger=logger,
    )

    return (
        biencoder,
        biencoder_params,
        candidate_encoding,
        indexer,
        id2title
    )

    


def run(
    args,
    biencoder,
    biencoder_params,
    candidate_encoding,
    indexer,
    id2title,
    test_data=None,
):

    stopping_condition = False
    threshold = float(args.threshold)
    if args.threshold_type == "top_entity_by_mention":
        assert args.mention_threshold is not None
        mention_threshold = float(args.mention_threshold)
    else:
        mention_threshold = threshold
  
    samples = test_data

    dataloader = _process_biencoder_dataloader(
        samples, biencoder.tokenizer, biencoder_params, None,
    )

    nns, dists, pred_mention_bounds, mention_scores, cand_scores = _run_biencoder(
        args, biencoder, dataloader, candidate_encoding, samples=samples,
        num_cand_mentions=args.num_cand_mentions, num_cand_entities=args.num_cand_entities,
        device="cuda",
        threshold=mention_threshold, indexer=indexer,
    )

    assert len(samples) == len(nns) == len(dists) == len(pred_mention_bounds) == len(cand_scores) == len(mention_scores)

    if args.prediction_type == "unique":
        (
                all_entity_preds, num_correct_weak, num_correct_strong, num_predicted, num_gold,
                num_correct_weak_from_input_window, num_correct_strong_from_input_window, num_gold_from_input_window,
            ) = get_predictions(
                args, dataloader, biencoder_params,
                samples, nns, dists, mention_scores, cand_scores,
                pred_mention_bounds, id2title, threshold=threshold,
                mention_threshold=mention_threshold,
            )
    elif args.prediction_type == "multiple":
        (
                all_entity_preds, num_correct_weak, num_correct_strong, num_predicted, num_gold,
                num_correct_weak_from_input_window, num_correct_strong_from_input_window, num_gold_from_input_window,
            ) = get_predictions_multiple(
                args, dataloader, biencoder_params,
                samples, nns, dists, mention_scores, cand_scores,
                pred_mention_bounds, id2title, threshold=threshold,
                mention_threshold=mention_threshold,
            )        
    print(f"Prediction type: {args.prediction_type}")
    if num_gold > 0:
        print("WEAK MATCHING")
        display_metrics(num_correct_weak, num_predicted, num_gold)
        print("Just entities within input window...")
        display_metrics(num_correct_weak_from_input_window, num_predicted, num_gold_from_input_window)
        print("*--------*")
        print("STRONG MATCHING")
        display_metrics(num_correct_strong, num_predicted, num_gold)
        print("Just entities within input window...")
        display_metrics(num_correct_strong_from_input_window, num_predicted, num_gold_from_input_window)
        print("*--------*")

    return all_entity_preds



