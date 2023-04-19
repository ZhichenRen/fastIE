from fastNLP import Metric, Vocabulary
from typing import Dict, Union
import random
from fastie.envs import logger


def calculate_metrics(metric: Dict[str, Union[float, int]]):
    pred_correct_cnt = metric['pred_correct_cnt']
    pred_cnt = metric['pred_cnt']
    correct_cnt = metric['correct_cnt']
    tp, fp, fn = pred_correct_cnt, pred_cnt - \
        pred_correct_cnt, correct_cnt - pred_correct_cnt
    p = 0 if tp + fp == 0 else (tp / (tp + fp))
    r = 0 if tp + fn == 0 else (tp / (tp + fn))
    f = 0 if p + r == 0 else (2 * p * r / (p + r))
    metric['precision'] = p
    metric['recall'] = r
    metric['F-1'] = f
    return metric


class REMetric(Metric):

    def __init__(self,
                 tag_vocab: Dict[str, Vocabulary],
                 evaluate_entity=True,
                 evaluate_relation=True,
                 use_strict=True):
        super(REMetric, self).__init__()
        keys = tag_vocab.keys()
        assert 'entity' in keys and 'relation' in keys and 'joint' in keys, 'tag vocab should contain key "entity", "relation", and "joint".'
        assert evaluate_entity or evaluate_relation, 'The metric should evaluate entity or relation or both!'
        self.evaluate_entity = evaluate_entity
        self.evaluate_relation = evaluate_relation
        self.use_strict = use_strict
        self.entity_vocab = tag_vocab['entity']
        self.relation_vocab = tag_vocab['relation']
        self.joint_vocab = tag_vocab['joint']
        self.reset_count()

    def reset_count(self):
        count = {}
        for k in self.joint_vocab.word2idx.keys():
            count[k] = {
                'pred_correct_cnt': 0,
                'strict_pred_correct_cnt': 0,
                'correct_cnt': 0,
                'pred_cnt': 0
            }

        count['all_entity'] = {
            'pred_correct_cnt': 0,
            'strict_pred_correct_cnt': 0,
            'correct_cnt': 0,
            'pred_cnt': 0
        }
        count['all_relation'] = {
            'pred_correct_cnt': 0,
            'strict_pred_correct_cnt': 0,
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
               rel_target=None):
        # ！ spans here are right exclusive !!!
        if self.evaluate_entity:
            assert ent_pred and ent_target, 'You should give the predict value and target value of entities!'
            assert len(ent_pred) == len(
                ent_target
            ), 'Different sentence number in entity predict value and entity target value!'
        if self.evaluate_relation:
            assert rel_pred and rel_target, 'You should give the predict value and target value of relations!'
            assert len(rel_pred) == len(
                rel_target
            ), 'Different sentence number in relation predict value and relation target value!'
            if self.use_strict:
                assert ent_pred and ent_target, 'You should give the predict value and target value of entities to calculate strict relation metrics!'
        if self.evaluate_entity and self.evaluate_relation:
            assert len(ent_pred) == len(
                rel_pred
            ), 'Different sentence number in entity predict value and relation predict value!'
        rand = random.random()
        if rand > 0.9:
            if ent_pred:
                logger.info(
                    '--------------------Entity Output Example--------------------'
                )
                logger.info(f'pred: {ent_pred[0]}, target: {ent_target[0]}')
            if rel_pred:
                logger.info(
                    '--------------------Relation Output Example--------------------'
                )
                logger.info(f'pred: {rel_pred[0]}, target: {rel_target[0]}')
        batch_size = len(ent_pred) if self.evaluate_entity else len(rel_pred)
        for i in range(batch_size):
            if ent_pred and ent_target and self.evaluate_entity:
                for span_ent in ent_pred[i]:
                    ent = span_ent[1]
                    self.count[ent]['pred_cnt'] += 1
                    if span_ent in ent_target[i]:
                        self.count[ent]['pred_correct_cnt'] += 1

                for span_ent in ent_target[i]:
                    ent = span_ent[1]
                    self.count[ent]['correct_cnt'] += 1

            if rel_pred and rel_target and self.evaluate_relation:
                for spans_rel in rel_pred[i]:
                    span_s = spans_rel[0]
                    span_o = spans_rel[1]
                    rel = spans_rel[2]
                    self.count[rel]['pred_cnt'] += 1
                    if spans_rel in rel_target[i]:
                        if self.use_strict and ent_pred and ent_target:
                            # correct spans and relation type
                            subject_check = False
                            object_check = False
                            for span_ent in ent_pred[i]:
                                if not subject_check and span_ent[
                                        0] == span_s and span_ent in ent_target[
                                            i]:
                                    subject_check = True
                                if not object_check and span_ent[
                                        0] == span_o and span_ent in ent_target[
                                            i]:
                                    object_check = True
                            if subject_check and object_check:
                                self.count[rel]['strict_pred_correct_cnt'] += 1
                        self.count[rel]['pred_correct_cnt'] += 1

                for spans_rel in rel_target[i]:
                    rel = spans_rel[2]
                    self.count[rel]['correct_cnt'] += 1

    def get_metric(self) -> dict:
        entity: Dict[str, Union[float, int]] = {
            'pred_correct_cnt': 0,
            'correct_cnt': 0,
            'pred_cnt': 0
        }
        for ent in self.entity_vocab.word2idx.keys():
            entity['pred_correct_cnt'] += self.count[ent]['pred_correct_cnt']
            entity['correct_cnt'] += self.count[ent]['correct_cnt']
            entity['pred_cnt'] += self.count[ent]['pred_cnt']
        relation: Dict[str, Union[float, int]] = {
            'pred_correct_cnt': 0,
            'correct_cnt': 0,
            'pred_cnt': 0
        }
        for rel in self.relation_vocab.word2idx.keys():
            relation['pred_correct_cnt'] += self.count[rel]['pred_correct_cnt']
            relation['correct_cnt'] += self.count[rel]['correct_cnt']
            relation['pred_cnt'] += self.count[rel]['pred_cnt']
        relation_strict: Dict[str, Union[float, int]] = {
            'pred_correct_cnt': 0,
            'correct_cnt': 0,
            'pred_cnt': 0
        }
        for rel in self.relation_vocab.word2idx.keys():
            relation_strict['correct_cnt'] += self.count[rel]['correct_cnt']
            relation_strict['pred_cnt'] += self.count[rel]['pred_cnt']
            relation_strict['pred_correct_cnt'] += self.count[rel][
                'strict_pred_correct_cnt']

        result = {}
        if self.evaluate_entity:
            result['entity'] = calculate_metrics(entity)
        if self.evaluate_relation:
            result['relation'] = calculate_metrics(relation)
            if self.use_strict:
                result['relation_strict'] = calculate_metrics(relation_strict)
        return result
