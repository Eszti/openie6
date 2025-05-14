# '''
# Usage:
#    benchmark --gold=GOLD_OIE --out=OUTPUT_FILE --allennlp=ALLENNLP_OIE | --exactMatch
# Options:
#   --gold=GOLD_OIE              The gold reference Open IE file (by default, it should be under ./oie_corpus/all.oie).
#   --allennlp=ALLENNLP_OIE      Read from allennlp output format
#   --exactmatch                 Use exact match when judging whether an extraction is correct.
# '''
import argparse
import json
# import nltk
# nltk.download('stopwords')

import logging
import re
import string
from collections import defaultdict

import docopt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score

logging.basicConfig(level=logging.INFO)
import pandas as pd
from sklearn import metrics

# from benchmark.oie_readers.stanfordReader import StanfordReader
# from benchmark.oie_readers.ollieReader import OllieReader
# from benchmark.oie_readers.reVerbReader import ReVerbReader
# from benchmark.oie_readers.clausieReader import ClausieReader
# from benchmark.oie_readers.openieFourReader import OpenieFourReader
# from benchmark.oie_readers.propsReader import PropSReader
from oie_readers.allennlpReader import AllennlpReader
from oie_readers.goldReader import GoldReader
from matcher import Matcher


class Benchmark:
    ''' Compare the gold OIE dataset against a predicted equivalent '''

    def __init__(self, gold_fn):
        ''' Load gold Open IE, this will serve to compare against using the compare function '''
        gr = GoldReader()
        gr.read(gold_fn)
        self.gold = gr.oie

    def compare(self, predicted, matchingFunc, output_fn):
        ''' Compare gold against predicted using a specified matching function. Outputs PR curve to output_fn '''

        y_true, y_scores = [], []
        tp_ids = []
        tp_rules = defaultdict(list)
        fp_rules = defaultdict(list)
        total_gold, total_unmatched, total_predict = 0, 0, 0
        correctTotal, unmatchedCount = 0, 0
        predicted = Benchmark.normalizeDict(predicted)
        gold = Benchmark.normalizeDict(self.gold)
        for sent, goldExtractions in list(gold.items()):
            if sent not in predicted:
                predictedExtractions = []
                # The extractor didn't find any extractions for this sentence
                unmatchedCount += len(goldExtractions)
                # correctTotal += len(goldExtractions)
                # continue
            else:
                predictedExtractions = predicted[sent]
            total_gold += len(goldExtractions)
            total_predict += len(predictedExtractions)

            for goldEx in goldExtractions:
                correctTotal += 1
                found = False
                for predictedEx in predictedExtractions:
                    if matchingFunc(goldEx, predictedEx, ignoreStopwords=True, ignoreCase=True):
                        y_true.append(1)
                        y_scores.append(predictedEx.confidence)
                        predictedEx.matched.append(output_fn)
                        found = True
                        tp_ids.append(predictedEx.id_str)
                        break
                if not found:
                    total_unmatched += 1

            for predictedEx in [x for x in predictedExtractions if (output_fn not in x.matched)]:
                # Add false positives
                y_true.append(0)
                y_scores.append(predictedEx.confidence)

        nr_multiple_rules = 0
        for sent in predicted:
            for predictedEx in predicted[sent]:
                if "#" in predictedEx.rule:
                    nr_multiple_rules += 1
                if predictedEx.id_str in tp_ids:
                    tp_rules[predictedEx.rule].append(predictedEx.id_str)
                else:
                    fp_rules[predictedEx.rule].append(predictedEx.id_str)

        print('the number of unmatched is: ', total_unmatched)
        tp = total_gold - total_unmatched
        fn = total_unmatched
        fp = total_predict - tp

        print(f"TP: {tp}")
        print(f"FN: {fn}")
        print(f"FP: {fp}")
        print(f"total gold (TP + FN): {total_gold} - {tp + fn}")
        print(f"total predict (TP + FP): {total_predict} - {tp + fp}")
        print(f"total unmatched: {total_unmatched}")
        print(f"multiple rules: {nr_multiple_rules}")

        sorted_tp = sorted(tp_rules.items(), key=lambda x: len(x[1]), reverse=True)
        with open("tp_rules_full.txt", "w") as f:
            f.write(
                f"{json.dumps(sorted_tp, indent=2)}")
        with open("tp_rules_summary.txt", "w") as f:
            f.write(
                f"{json.dumps([f'{len(y[1])}: {y[0]}' for y in sorted_tp], indent=2)}")

        sorted_fp = sorted(fp_rules.items(), key=lambda x: len(x[1]), reverse=True)
        with open("fp_rules_full.txt", "w") as f:
            f.write(
                f"{json.dumps(sorted_fp, indent=2)}")
        with open("fp_rules_summary.txt", "w") as f:
            f.write(
                f"{json.dumps([f'{len(y[1])}: {y[0]}' for y in sorted_fp], indent=2)}")

        common_rules = set(tp_rules.keys()).intersection(set(fp_rules.keys()))
        with open("common_rules.txt", "w") as f:
            for rule in common_rules:
                f.write(f"{rule}: {len(tp_rules[rule])} (TP), {len(fp_rules[rule])} (FP)\n")
                # f.write(f"TP: {tp_rules[rule]}\n")
                # f.write(f"FP: {fp_rules[rule]}\n")
                # f.write("\n")

        try:
            recall = (total_gold - total_unmatched) / total_gold
            precision = (total_gold - total_unmatched) / total_predict
            my_F1 = 2.0 * recall * precision / (recall + precision)
            my_auc = roc_auc_score(y_true, y_scores)
            print("my F1: ", my_F1, "my AUC: ", my_auc)
            print("precision: ", precision, "recall", recall)
        except ZeroDivisionError:
            my_F1, my_auc = 0.0, 0.0

        recallMultiplier = (correctTotal - unmatchedCount) / float(correctTotal)

        # recall on y_true, y  (r')_scores computes |covered by extractor| / |True in what's covered by extractor|
        # to get to true recall we do r' * (|True in what's covered by extractor| / |True in gold|) = |true in what's covered| / |true in gold|
        p, r = Benchmark.prCurve(np.array(y_true), np.array(y_scores), recallMultiplier=recallMultiplier)
        # write PR to file
        with open(output_fn, 'w') as fout:
            fout.write('{0}\t{1}\n'.format("Precision", "Recall"))
            for cur_p, cur_r in sorted(zip(p, r), key=lambda cur_p_cur_r: cur_p_cur_r[1]):
                fout.write('{0}\t{1}\n'.format(cur_p, cur_r))

        data = np.array([p, r]).transpose()
        df = pd.DataFrame(data=data, columns=["p", "r"])
        df['f1'] = 2 * (df['r'] * df['p']) / (df['r'] + df['p'])
        best_F1 = df['f1'].max()
        print('max f1 is ' + str(df['f1'].max()))
        df = df[df['r'] > 0]
        r = tuple(list(df['r']))
        p = tuple(list(df['p']))
        try:
            auc = metrics.auc(df['r'].values, df['p'].values)
        except ValueError:
            auc = 0.0
        print('auc is ' + str(auc))

        return my_F1, my_auc, best_F1, auc

    @staticmethod
    def prCurve(y_true, y_scores, recallMultiplier):
        # Recall multiplier - accounts for the percentage examples unreached by
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        recall = recall * recallMultiplier
        return precision, recall

    # Helper functions:
    @staticmethod
    def normalizeDict(d):
        return dict([(Benchmark.normalizeKey(k), v) for k, v in list(d.items())])

    @staticmethod
    def normalizeKey(k):
        return Benchmark.removePunct(str(Benchmark.PTB_unescape(k.replace(' ', ''))))

    @staticmethod
    def PTB_escape(s):
        for u, e in Benchmark.PTB_ESCAPES:
            s = s.replace(u, e)
        return s

    @staticmethod
    def PTB_unescape(s):
        for u, e in Benchmark.PTB_ESCAPES:
            s = s.replace(e, u)
        return s

    @staticmethod
    def removePunct(s):
        return Benchmark.regex.sub('', s)

    # CONSTANTS
    regex = re.compile('[%s]' % re.escape(string.punctuation))

    # Penn treebank bracket escapes
    # Taken from: https://github.com/nlplab/brat/blob/master/server/src/gtbtokenize.py
    PTB_ESCAPES = [('(', '-LRB-'),
                   (')', '-RRB-'),
                   ('[', '-LSB-'),
                   (']', '-RSB-'),
                   ('{', '-LCB-'),
                   ('}', '-RCB-'), ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=str)
    parser.add_argument("--allennlp", type=str)
    parser.add_argument("--out", type=str)
    return parser


if __name__ == '__main__':
    parser = parse_args()
    args = parser.parse_args()
    logging.debug(args)

    predicted = AllennlpReader(threshold=None)
    predicted.read(args.allennlp)

    b = Benchmark(args.gold)
    out_filename = args.out

    logging.info("Writing PR curve of {} to {}".format(predicted.name, out_filename))
    b.compare(predicted=predicted.oie, matchingFunc=Matcher.exactMatch, output_fn=out_filename)
