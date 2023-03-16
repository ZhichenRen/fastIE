from fastNLP import Metric, Vocabulary
from typing import Dict


def calculate_metrics(pred_correct_cnt, pred_cnt, correct_cnt):
    """This function calculation metrics: precision, recall, f1-score.

    Arguments:
        pred_correct_cnt {int} -- the number of corrected prediction
        pred_cnt {int} -- the number of prediction
        correct_cnt {int} -- the number of truth

    Returns:
        tuple -- precision, recall, f1-score
    """

    tp, fp, fn = pred_correct_cnt, pred_cnt - \
        pred_correct_cnt, correct_cnt - pred_correct_cnt
    p = 0 if tp + fp == 0 else (tp / (tp + fp))
    r = 0 if tp + fn == 0 else (tp / (tp + fn))
    f = 0 if p + r == 0 else (2 * p * r / (p + r))
    return p, r, f


class REMetric(Metric):

    def __init__(self, tag_vocab: Dict[str, Vocabulary]):
        super(REMetric, self).__init__()
        keys = tag_vocab.keys()
        assert 'entity' in keys and 'relation' in keys and 'joint' in keys, 'tag vocab should contain key "entity", "relation", and "joint".'

        self.entity_vocab = tag_vocab['entity']
        self.relation_vocab = tag_vocab['relation']
        self.joint_vocab = tag_vocab['joint']
        self.reset_count()

    def reset_count(self):
        count = {}
        for k in self.joint_vocab.word2idx.keys():
            count[k] = {'pred_correct_cnt': 0, 'correct_cnt': 0, 'pred_cnt': 0}

        count['all_entity'] = {
            'pred_correct_cnt': 0,
            'correct_cnt': 0,
            'pred_cnt': 0
        }
        count['all_relation'] = {
            'pred_correct_cnt': 0,
            'correct_cnt': 0,
            'pred_cnt': 0
        }
        self.count = count

    def reset(self):
        self.reset_count()

    def update(self,
               ent_pred=None,
               ent_target=None,
               rel_pred=None,
               rel_target=None,
               evaluate_entity=True,
               evaluate_relation=True):
        # ！ spans here are right exclusive !!!
        for i in range(len(ent_pred)):
            if ent_pred is not None and ent_target is not None and evaluate_entity:
                for span_ent in ent_pred[i]:
                    ent = span_ent[1]
                    self.count[ent]['pred_cnt'] += 1
                    if span_ent in ent_target[i]:
                        self.count[ent]['pred_correct_cnt'] += 1

                for span_ent in ent_target[i]:
                    ent = span_ent[1]
                    self.count[ent]['correct_cnt'] += 1

            if rel_pred is not None and rel_target is not None and evaluate_relation:
                for spans_rel in rel_pred[i]:
                    span_s = spans_rel[0]
                    span_o = spans_rel[1]
                    rel = spans_rel[2]
                    self.count[rel]['pred_cnt'] += 1
                    if spans_rel in rel_target[i]:
                        # correct spans and relation type
                        subject_check = False
                        object_check = False
                        for span_ent in ent_pred[i]:
                            if span_ent[0] == span_s and span_ent in ent_pred[
                                    i]:
                                subject_check = True
                            if span_ent[0] == span_o and span_ent in ent_pred[
                                    i]:
                                object_check = True
                        if subject_check and object_check:
                            self.count[rel]['pred_correct_cnt'] += 1

                for spans_rel in rel_target[i]:
                    rel = spans_rel[2]
                    self.count[rel]['correct_cnt'] += 1

    def get_metric(self) -> dict:
        for ent in self.entity_vocab.word2idx.keys():
            self.count['all_entity']['pred_correct_cnt'] += self.count[ent][
                'pred_correct_cnt']
            self.count['all_entity']['correct_cnt'] += self.count[ent][
                'correct_cnt']
            self.count['all_entity']['pred_cnt'] += self.count[ent]['pred_cnt']
        for rel in self.relation_vocab.word2idx.keys():
            self.count['all_relation']['pred_correct_cnt'] += self.count[rel][
                'pred_correct_cnt']
            self.count['all_relation']['correct_cnt'] += self.count[rel][
                'correct_cnt']
            self.count['all_relation']['pred_cnt'] += self.count[rel][
                'pred_cnt']

        for type, metric in self.count.items():
            p, r, f = calculate_metrics(metric['pred_correct_cnt'],
                                        metric['pred_cnt'],
                                        metric['correct_cnt'])
            self.count[type]['precision'] = p
            self.count[type]['recall'] = r
            self.count[type]['F-1'] = f

        result = {}
        result['entity'] = self.count['all_entity']
        result['relation'] = self.count['all_relation']

        return result
