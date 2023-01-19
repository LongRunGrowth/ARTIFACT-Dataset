import numpy as np
import pandas as pd
from termcolor import colored, cprint


class ATTPrinter():
    def __init__(
        self,
        pred,
        aat,
        aat_desc_df,
        print_list,
        ann_type,
        correct_aat=None,
        tag_type=None,
        color="grey"
    ):
        self.tag_type = tag_type
        self.correct_aat = correct_aat
        self.pred = str(pred)
        self.ann_type = str(ann_type)
        self.color = str(color)
        # get elements and print
        self.get_linking_desc(aat, aat_desc_df)
        self.print_desc(aat, print_list)

    def get_linking_desc(self, aat, aat_desc_df):
        orig_df = aat_desc_df.copy()
        try:
            aat_desc_df = aat_desc_df[aat_desc_df["Preferred"] == True]
            self.name = aat_desc_df[aat_desc_df['AAT_ID']
                                    == aat].Name.values[0]
            full_description = aat_desc_df[aat_desc_df['AAT_ID']
                                           == aat].Description.values[0]
            first_dot = full_description.find(".")
            self.description = full_description[:first_dot + 1]
            self.parent = aat_desc_df[aat_desc_df['AAT_ID']
                                      == aat].Parent.values[0]
            self.facet = aat_desc_df[aat_desc_df['AAT_ID']
                                     == aat].Facet.values[0]
        except:
            self.name = orig_df[orig_df['AAT_ID'] == aat].Name.values[0]
            self.description = orig_df[orig_df['AAT_ID']
                                       == aat].Description.values[0]
            self.parent = orig_df[orig_df['AAT_ID'] == aat].Parent.values[0]
            self.facet = orig_df[orig_df['AAT_ID'] == aat].Facet.values[0]

    def print_desc(self, aat, elements_list):

        if self.ann_type == "golden":
            pred_colored = colored(self.pred, "blue", 'on_grey')
        else:
            pred_colored = colored(self.pred, "yellow", 'on_grey')

        cprint("  " + u"\u2022" +
               f" {pred_colored} " u"\u27F6" + f"  {self.name}", self.color)

        if 'aat' in elements_list:
            if self.ann_type == "golden":
                pred_aat_colored = colored(int(aat), 'blue', 'on_yellow')
                if self.tag_type == "seen":
                    cprint(f"   * AAT_ID: {pred_aat_colored}",
                           self.color, attrs=['underline'])
                elif self.tag_type == "unseen":
                    cprint(f"   * AAT_ID: {pred_aat_colored}", self.color)
            else:
                if self.correct_aat != None:
                    if self.correct_aat == "spurious":
                        pred_aat_colored = colored(int(aat), 'white', 'on_red')
                        cprint(f"   * AAT_ID: {pred_aat_colored}", self.color)
                    elif self.correct_aat:
                        pred_aat_colored = colored(
                            int(aat), 'white', 'on_green')
                        cprint(f"   * AAT_ID: {pred_aat_colored}", self.color)
                    else:
                        pred_aat_colored = colored(int(aat), 'white', 'on_red')
                        cprint(f"   * AAT_ID: {pred_aat_colored}", self.color)
                else:
                    cprint(f"   * AAT_ID: {int(aat)}", self.color)

        if self.ann_type == "golden":
            if 'description' in elements_list:
                cprint(f"   * Description: {self.description}", self.color)
        elif self.ann_type != "golden" and not self.correct_aat:
            cprint(f"   * Description: {self.description}", self.color)
        elif self.ann_type != "golden" and self.correct_aat == "spurious":
            cprint(f"   * Description: {self.description}", self.color)

        if 'facet' in elements_list:
            cprint(f"   * Facet: {self.facet}", self.color)
        if 'parent' in elements_list:
            cprint(f"   * Parent: {self.parent}", self.color)


class QualitativeAnalizer():
    def __init__(
        self,
        gold_standard_data,
        prediction_df,
        aat_df=None,
        model_name="ELQ Entity Linking",
        aat_print_list=['aat', 'description']  # facet, parent
    ):
        self.aat_df = aat_df
        self.aat_print_list = aat_print_list
        self.model_name = model_name
        self.total_correct = gold_standard_data.shape[0]
        
        # treat prediction data
        self.pred_df = self._adjust_prediction_df(prediction_df)
        # calculate matches (each case)
        self.calc_matches_df(prediction_df)

    def _adjust_df_types(self, df):
        df["text"] = df["text"].values.astype(str)
        df["text"] = df["text"].str.lower()
        df["chunk_text"] = df["chunk_text"].values.astype(str)
        df["chunk_text"] = df["chunk_text"].str.lower()
        df["chunk_start"] = df["chunk_start"].values.astype(str)
        df["chunk_end"] = df["chunk_end"].values.astype(str)
        df["aat"] = df["aat"].values.astype(float)
        return df

    def _adjust_prediction_df(self, prediction_df):
        prediction_df["text"] = prediction_df["text"].values.astype(str)
        prediction_df["text"] = prediction_df["text"].str.lower()
        prediction_df["chunk_text"] = prediction_df["chunk_text"].values.astype(str)
        prediction_df["chunk_text"] = prediction_df["chunk_text"].str.lower()
        prediction_df["chunk_start"] = prediction_df["chunk_start"].values.astype(str)
        prediction_df["chunk_end"] = prediction_df["chunk_end"].values.astype(str)
        prediction_df["aat"] = prediction_df["aat"].values.astype(float)

        if 'id' in prediction_df.columns:
            prediction_df.rename(columns={'id': 'unique_id', }, inplace=True)
        return prediction_df

    def calc_matches_df(self, prediction_df):
        keys_join = ['text', 'chunk_text',
                     'chunk_start', 'chunk_end', "unique_id"]
        joined_df = prediction_df.merge(
            self.test_data, on=keys_join, how='inner')
        total_merge = prediction_df.merge(
            self.test_data, on=keys_join, how='outer', indicator=True)
        total_merge.rename(
            columns={"aat_x": "pred_aat", "aat_y": "aat"}, inplace=True)

        self.total_merge_df = total_merge
        self.incorrect_pred_df = total_merge[total_merge['_merge']
                                             == 'left_only']
        self.missed_pred_df = total_merge[total_merge['_merge']
                                          == 'right_only']
        self.correct_pred_df = total_merge[total_merge['_merge'] == 'both']

    def color_pred(self, sub_df, color, ann_type, ret_list=True, print_ann=True):
        if ann_type != "golden":
            # or _merge == 'right_only'
            sub_df = sub_df.query("_merge=='both' or _merge == 'left_only'")

        if ann_type != "golden":
            pred_sub_df = sub_df.query(
                "_merge=='both' or _merge == 'left_only'")
            correct_list = []

            for _, row in pred_sub_df.iterrows():
                correct_list.append(row["_merge"] == "both")

        text = sub_df["text"].iloc[0]
        ann_words_list = list(sub_df.chunk_text.values)

        if ann_type == "golden":
            correct_list = [True]*len(ann_words_list)

        ann_pos = []
        for _, row in sub_df.iterrows():
            ann_pos.append(int(row["chunk_start"]))
            ann_pos.append(int(row["chunk_end"]))
        ann_pos.sort()

        if 0 not in ann_pos:
            ann_pos.append(0)

        ann_pos.sort()
        len_pos = len(ann_pos)

        colored_text = []
        idx_predicted = 0
        for idx in range(len_pos):
            if idx < len_pos - 1:
                text_chunk = text[ann_pos[idx]: ann_pos[idx + 1]]
            else:
                text_chunk = text[ann_pos[idx]:]

            if (text_chunk in ann_words_list) and (correct_list[idx_predicted] == True):
                colored_text.append(colored(text_chunk, color, 'on_grey'))
                idx_predicted += 1
            elif (text_chunk in ann_words_list) and (correct_list[idx_predicted] == False):
                colored_text.append(colored(text_chunk, color, 'on_red'))
                idx_predicted += 1
            else:
                colored_text.append(text_chunk)

        colored_text = ("").join(colored_text)

        if print_ann:
            if color == "blue":
                print(u"\u21E8", colored_text, "\n")
            else:
                print(u"\u27A1", colored_text, "\n")

        if type(self.aat_df) == pd.core.frame.DataFrame:
            if ann_type == "golden":
                pass
            else:
                # only tagged correctly
                sub_df = sub_df.query("_merge=='both'")
                sub_df_spurious = pred_sub_df.query(
                    "_merge=='left_only'").reset_index()  # only annotated by the system

            for idx, row in sub_df.iterrows():
                if ann_type == "golden":
                    ATTPrinter(row["chunk_text"], float(
                        row["aat"]), self.aat_df, self.aat_print_list, ann_type, tag_type=row["tag_type"])
                else:
                    if idx == 0:
                        cprint("AATs Tagged correctly",
                               'green', attrs=['bold'])
                    if row["aat"] == row["golden_aat"]:
                        correct_linking = True
                    else:
                        correct_linking = False

                    ATTPrinter(row["chunk_text"], float(
                        row["aat"]), self.aat_df, self.aat_print_list, ann_type, correct_aat=correct_linking)

            if ann_type != "golden":
                for idx_, row in sub_df_spurious.iterrows():
                    if idx_ == 0:
                        cprint("AATs Tagged incorrectly",
                               'red', attrs=['bold'])
                    ATTPrinter(row["chunk_text"], float(
                        row["aat"]), self.aat_df, self.aat_print_list, ann_type, correct_aat="spurious")

    def compare_tags(self, golden_subdf, pred_subf, ret_list=True, print_ann=True):
        if print_ann:
            cprint(
                f"Number of golden/predicted annotations: {golden_subdf.shape[0]} / {pred_subf.shape[0]}")
            print("")
            cprint(u"\u25A1" + f" Golden:")
            print("")
        colored_golden = self.color_pred(
            golden_subdf, color="blue", ann_type="golden", print_ann=print_ann)

        if print_ann:
            print("")
            cprint(u"\u25A0" + f" Predicted:")
            # omit 'chunk_text' as capital letters hurt matching
        full_df = pd.merge(pred_subf, golden_subdf, on=[
                           'unique_id', 'text', 'chunk_start', 'chunk_end'], how='outer', indicator=True)
        #cprint(full_df, "red")
        full_df.rename(
            columns={'aat_x': 'aat', 'aat_y': 'golden_aat'}, inplace=True)
        full_df.rename(columns={'chunk_text_x': 'chunk_text',
                       'chunk_text_y': 'golden_chunk_text'}, inplace=True)

        colored_pred = self.color_pred(
            full_df, color="yellow", ann_type="not_golden", print_ann=print_ann)

        if print_ann:
            print("")
            print("")

        if ret_list:
            return colored_golden, colored_pred

    def ansi_report_lines(self):
        incorrect_uniqu_ids = list(set(self.incorrect_pred_df.unique_id.values))
        correct_unique_ids = list(
            set(self.correct_pred_df.unique_id.values))
        intersection_unique_ids = set(
            correct_unique_ids).union(set(incorrect_uniqu_ids))

        report_lines = []
        title = f"Prediction examples for qualitative analysis (Tagging & Linking) | {self.model_name}"
        cprint(f"{title}", attrs=['bold'])
        print("")
        print("")

        for unique_id in intersection_unique_ids:
            golden_ann = self.test_data[self.test_data["unique_id"] == unique_id]
            pred_ann = self.pred_df[self.pred_df["unique_id"] == unique_id]
            if (golden_ann.shape[0] and pred_ann.shape[0]) > 0:
                cprint(u"\u25B2" + f" ID: {unique_id}", attrs=['bold'])
                colored_golden, colored_pred = self.compare_tags(golden_ann, pred_ann, print_ann=True)
                report_lines.append(colored_golden)
                report_lines.append(colored_pred)
                print("")
            else:
                cprint("test", "green")
        return report_lines
