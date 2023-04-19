from fastNLP.io import Pipe, DataBundle
from fastNLP.transformers.torch import BertTokenizer
from fastNLP import DataSet, Instance, Vocabulary
from rich.progress import track
from typing import Dict, List, Tuple
from fastie.utils.utils import add_cross_sentence
from fastie.envs import logger


class PUREEntityPipe(Pipe):

    def __init__(self,
                 tag_vocab: Dict[str, Vocabulary],
                 tokenizer='bert-base-uncased',
                 max_span_len=8,
                 add_cross_sent=True,
                 cross_sent_window=200):
        super(PUREEntityPipe, self).__init__()
        assert 'entity' in tag_vocab.keys(
        ), "The key entity doesn't exist in tag_vocab!"
        self.tag_vocab = tag_vocab
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        self.max_span_len = max_span_len
        self.cross_sent_window = cross_sent_window
        self.add_cross_sent = add_cross_sent

    def process(self, data_bundle: DataBundle) -> DataBundle:
        new_databundle = DataBundle()
        for dataset_name in data_bundle.get_dataset_names():
            new_dataset = DataSet()
            dataset = data_bundle.get_dataset(dataset_name)
            if self.add_cross_sent:
                cross_sent_instances = add_cross_sentence(
                    dataset, self.cross_sent_window)

            for i in track(
                    range(len(dataset)),
                    description=f'Processing {dataset_name} for PURE Entity Task ...'):
                tokens = dataset.tokens[i]
                tokens_len = len(tokens)

                if self.add_cross_sent:
                    sent_start = cross_sent_instances[i]['sent_start']
                    cross_sent_tokens = cross_sent_instances[i][
                        'cross_sent_tokens']
                else:
                    sent_start = 0
                    cross_sent_tokens = dataset.tokens[i]
                entity_mentions = dataset.entity_mentions[
                    i] if dataset.has_field('entity_mentions') else []
                ners = {ner[0]: ner[1] for ner in entity_mentions}
                relation_mentions = dataset.relation_mentions[
                    i] if dataset.has_field('relation_mentions') else []

                origin_spans = []
                tokenized_spans = []
                spans_labels = []

                for span_s in range(0, tokens_len):
                    for span_e in range(
                            span_s, min(span_s + self.max_span_len,
                                        tokens_len)):
                        # both start and end are inclusive
                        tokenized_spans.append([
                            span_s + sent_start, span_e + sent_start,
                            span_e - span_s + 1
                        ])
                        origin_spans.append([span_s, span_e])
                        if (span_s, span_e + 1) in ners:
                            spans_labels.append(
                                self.tag_vocab['entity'].to_index(
                                    ners[(span_s, span_e + 1)]))
                        else:
                            spans_labels.append(0)

                # tokenize here
                token2start = []
                token2end = []
                input_ids = [self.tokenizer.cls_token]
                for idx, token in enumerate(cross_sent_tokens):
                    # idx: word index in cross sentence
                    split_tokens = self.tokenizer.tokenize(token)
                    token2start.append(len(input_ids))
                    input_ids.extend(split_tokens)
                    token2end.append(len(input_ids) - 1)
                input_ids.append(self.tokenizer.sep_token)

                for span in tokenized_spans:
                    assert span[0] >= 0 and span[0] < len(
                        token2start
                    ) and span[1] >= 0 and span[1] < len(
                        token2end
                    ), f'token2start len: {len(token2start)}, token2end len: {len(token2end)}, span[0]: {span[0]}, span[1]: {span[1]}, sentence: {tokens}, cross sentence: {cross_sent_tokens}, sent start: {sent_start}'

                tokenized_spans = [[
                    token2start[span[0]], token2end[span[1]], span[2]
                ] for span in tokenized_spans]

                new_instance = Instance(
                    doc_key=dataset.doc_key[i],
                    sent_id=dataset.sent_id[i],
                    tokens=dataset.tokens[i],
                    input_ids=self.tokenizer.convert_tokens_to_ids(input_ids),
                    spans=tokenized_spans,
                    spans_mask=[True for _ in range(len(tokenized_spans))],
                    attention_mask=[True for _ in range(len(input_ids))],
                    labels=spans_labels,
                    span2ent=entity_mentions,
                    span2rel=relation_mentions,
                    origin_spans=origin_spans)
                new_dataset.append(new_instance)
            new_databundle.set_dataset(new_dataset, dataset_name)
        new_databundle.set_pad('span2ent', pad_val=None)
        new_databundle.set_pad('span2rel', pad_val=None)
        new_databundle.set_pad('origin_spans', pad_val=None)
        new_databundle.set_pad('tokens', pad_val=None)
        new_databundle.set_pad('sent_id', pad_val=None)
        new_databundle.set_pad('doc_key', pad_val=None)
        return new_databundle

    def process_from_file(self, paths: str) -> DataBundle:
        return super().process_from_file(paths)


def add_special_tokens(tokenizer: BertTokenizer, entity_vocab: Vocabulary):
    new_tokens = ['<SUBJ_START>', '<SUBJ_END>', '<OBJ_START>', '<OBJ_END>']
    for label in entity_vocab.word2idx.keys():
        new_tokens.append('<SUBJ_START=%s>' % label)
        new_tokens.append('<SUBJ_END=%s>' % label)
        new_tokens.append('<OBJ_START=%s>' % label)
        new_tokens.append('<OBJ_END=%s>' % label)
        new_tokens.append('<SUBJ=%s>' % label)
        new_tokens.append('<OBJ=%s>' % label)
    tokenizer.add_tokens(new_tokens, special_tokens=True)
    return tokenizer


def get_attention_mask(tokens_type):
    # * The return value mask have shape tokens_len * tokens_len
    attention_mask = []
    for from_token in tokens_type:
        attention_mask_line = []
        for to_token in tokens_type:
            if to_token <= 1:
                attention_mask_line.append(1)
            elif from_token == to_token:
                attention_mask_line.append(1)
            else:
                attention_mask_line.append(0)
        attention_mask.append(attention_mask_line)
    return attention_mask


class PURERelationPipe(Pipe):

    def __init__(self,
                 tag_vocab: Dict[str, Vocabulary],
                 tokenizer: BertTokenizer,
                 max_span_len: int = 8,
                 add_cross_sent: bool = True,
                 cross_sent_window: int = 200,
                 training: bool = True):
        super(PURERelationPipe, self).__init__()
        assert 'entity' in tag_vocab.keys() and 'relation' in tag_vocab.keys(
        ), "Key 'entity' or 'relation' not in tag_vocab!"
        self.tag_vocab = tag_vocab
        self.tokenizer = tokenizer
        self.max_span_len = max_span_len
        self.cross_sent_window = cross_sent_window
        self.add_cross_sent = add_cross_sent

    def process(self, data_bundle: DataBundle) -> DataBundle:
        '''
        返回的databundle需要包含以下字段:
        tokens: 未经处理的原句子
        input_ids: 经过cross sentence与tokenize的tokens list
        attention_mask:
        labels:
        sub_index: 插入的subject特殊token在input_ids中的位置
        obj_index: 插入的object特殊token在input_ids中的位置
        origin_subect_index: subject在原句子中的位置
        origin_object_index: object在原句子中的位置
        span2ent: 句子中实体的（位置，标签）键值对（字典或元组）
        span2rel: 句子中关系的（位置，标签）键值对（字典或元组）
        '''
        new_databundle = DataBundle()
        self.tokenizer = add_special_tokens(self.tokenizer,
                                            self.tag_vocab['entity'])
        for dataset_name in data_bundle.get_dataset_names():
            dataset = data_bundle.get_dataset(dataset_name)

            new_dataset = DataSet()

            if self.add_cross_sent:
                cross_sent_instances = add_cross_sentence(
                    dataset, self.cross_sent_window)

            for i in track(
                    range(len(dataset)),
                    description=f'Processing {dataset_name} for PURE Relation Task ...'):
                tokens = dataset.tokens[i]

                if self.add_cross_sent:
                    sent_start = cross_sent_instances[i]['sent_start']
                    cross_sent_tokens = cross_sent_instances[i][
                        'cross_sent_tokens']
                else:
                    sent_start = 0
                    cross_sent_tokens = dataset.tokens[i]
                relation_mentions = dataset.relation_mentions[
                    i] if dataset.has_field('relation_mentions') else []
                rels = {((rel[0][0], rel[0][1]), (rel[1][0], rel[1][1])):
                        rel[2]
                        for rel in relation_mentions}
                span2ent = dataset.entity_mentions[i] if dataset.has_field(
                    'entity_mentions') else []
                ent_pred = dataset.ent_pred[i] if dataset.has_field(
                    'ent_pred') else span2ent

                # traverse all span pairs here
                # may split one dataset instance into several instances
                for subject in ent_pred:
                    for object in ent_pred:
                        if subject == object:
                            continue
                        subject_pos = subject[0]
                        object_pos = object[0]
                        instance = {}
                        tokens = dataset.tokens[i]
                        origin_subject_index = [subject_pos]
                        origin_object_index = [object_pos]
                        if ((subject_pos, object_pos) in rels):
                            labels = self.tag_vocab['relation'].to_index(
                                rels[(subject_pos, object_pos)])
                            rel_type = rels[(subject_pos, object_pos)]
                            span2rel = [(subject_pos, object_pos, rel_type)]
                        else:
                            labels = 0
                            span2rel = []

                        input_ids = [self.tokenizer.cls_token_id]
                        subject_index = [0, 0]
                        object_index = [0, 0]
                        for idx, word in enumerate(cross_sent_tokens):
                            if idx == subject_pos[0] + sent_start:
                                subject_index[0] = len(input_ids)
                                input_ids.append(
                                    self.tokenizer.convert_tokens_to_ids(
                                        f'<SUBJ_START={subject[1]}>'))
                            if idx == object_pos[0] + sent_start:
                                object_index[0] = len(input_ids)
                                input_ids.append(
                                    self.tokenizer.convert_tokens_to_ids(
                                        f'<OBJ_START={object[1]}>'))
                            input_ids.extend(
                                self.tokenizer.convert_tokens_to_ids(
                                    self.tokenizer.tokenize(word)))
                            if idx == subject_pos[1] + sent_start:
                                subject_index[1] = len(input_ids)
                                input_ids.append(
                                    self.tokenizer.convert_tokens_to_ids(
                                        f'<SUBJ_END={subject[1]}>'))
                                subject_index[1] = idx
                            if idx == object_pos[1] + sent_start:
                                object_index[1] = len(input_ids)
                                input_ids.append(
                                    self.tokenizer.convert_tokens_to_ids(
                                        f'<OBJ_END={object[1]}>'))

                        input_ids.append(self.tokenizer.sep_token_id)
                        instance_ent_pred = [subject, object]
                        split_span2ent = []
                        for gold in span2ent:
                            if gold[0] == subject[0]:
                                split_span2ent.append(gold)
                            if gold[0] == object[0]:
                                split_span2ent.append(gold)
                        instance = Instance(
                            tokens=tokens,
                            attention_mask=[True for _ in input_ids],
                            input_ids=input_ids,
                            origin_subject_index=origin_subject_index,
                            origin_object_index=origin_object_index,
                            labels=labels,
                            span2ent=split_span2ent,
                            span2rel=span2rel,
                            ent_pred=instance_ent_pred,
                            subject_index=subject_index,
                            object_index=object_index,
                        )
                        new_dataset.append(instance)

            new_databundle.set_dataset(dataset=new_dataset, name=dataset_name)

        new_databundle.set_pad('tokens', pad_val=None)
        new_databundle.set_pad('origin_subject_index', pad_val=None)
        new_databundle.set_pad('origin_object_index', pad_val=None)
        new_databundle.set_pad('span2ent', pad_val=None)
        new_databundle.set_pad('span2rel', pad_val=None)
        new_databundle.set_pad('ent_pred', pad_val=None)
        return new_databundle

    def process_from_file(self, paths: str) -> DataBundle:
        return super().process_from_file(paths)


class PURERelationApproxPipe(Pipe):

    def __init__(self,
                 tag_vocab: Dict[str, Vocabulary],
                 tokenizer: BertTokenizer,
                 max_span_len: int = 8,
                 add_cross_sent: bool = True,
                 cross_sent_window: int = 200,
                 max_sequence_len: int = 512):
        super(PURERelationApproxPipe, self).__init__()
        assert 'entity' in tag_vocab.keys() and 'relation' in tag_vocab.keys(
        ), "Key 'entity' or 'relation' not in tag_vocab!"
        self.tag_vocab = tag_vocab
        self.tokenizer = tokenizer
        self.max_span_len = max_span_len
        self.cross_sent_window = cross_sent_window
        self.add_cross_sent = add_cross_sent
        self.max_sequence_len = max_sequence_len

    def process(self, data_bundle: DataBundle) -> DataBundle:
        '''
        返回的databundle需要包含以下字段:
        tokens: 未经处理的原句子
        input_ids: 经过cross sentence与tokenize的tokens list
        position_ids: 每个token的position id，长度与input_ids对应
        attention_mask:
        labels:
        sub_index: 插入的subject特殊token在input_ids中的位置
        obj_index: 插入的object特殊token在input_ids中的位置
        origin_subect_index: subject在原句子中的位置
        origin_object_index: object在原句子中的位置
        span2ent: 句子中实体的（位置，标签）键值对（字典或元组）
        span2rel: 句子中关系的（位置，标签）键值对（字典或元组
        '''
        new_databundle = DataBundle()
        self.tokenizer = add_special_tokens(self.tokenizer,
                                            self.tag_vocab['entity'])
        for dataset_name in data_bundle.get_dataset_names():
            dataset = data_bundle.get_dataset(dataset_name)
            new_dataset = DataSet()

            if self.add_cross_sent:
                cross_sent_instances = add_cross_sentence(
                    dataset, self.cross_sent_window)

            for i in track(
                    range(len(dataset)),
                    description=f'Processing {dataset_name} for PURE Relation Task...'):
                tokens = dataset.tokens[i]
                tokens_len = len(tokens)
                if self.cross_sent_window != 0 and tokens_len > self.cross_sent_window:
                    logger.info('Long sentence: {}'.format(tokens))

                # ? different model use different cross sentence strategy here
                # * use method in pure here

                if self.add_cross_sent:
                    sent_start = cross_sent_instances[i]['sent_start']
                    cross_sent_tokens = cross_sent_instances[i][
                        'cross_sent_tokens']
                else:
                    sent_start = 0
                    cross_sent_tokens = dataset.tokens[i]
                relation_mentions = dataset.relation_mentions[
                    i] if dataset.has_field('relation_mentions') else []
                rels = {((rel[0][0], rel[0][1]), (rel[1][0], rel[1][1])):
                        rel[2]
                        for rel in relation_mentions}

                tokenized_start = []
                tokenized_end = []
                input_ids = [self.tokenizer.cls_token_id]
                for word in cross_sent_tokens:
                    tokenized_start.append(len(input_ids))
                    input_ids.extend(
                        self.tokenizer.encode(word, add_special_tokens=False))
                    tokenized_end.append(len(input_ids) - 1)
                input_ids.append(self.tokenizer.sep_token_id)
                origin_len = len(input_ids)

                position_ids = list(range(origin_len))
                origin_subject_index: List[List] = []
                origin_object_index: List[List] = []
                subject_index: List[List] = []
                object_index: List[List] = []
                labels: List = []
                span2rel: List[Tuple] = []
                span2ent = dataset.entity_mentions[i] if dataset.has_field(
                    'entity_mentions') else []
                ent_pred = dataset.ent_pred[i] if dataset.has_field(
                    'ent_pred') else span2ent
                tokens_type = [1 for _ in range(origin_len)]

                if len(input_ids) + 4 > self.max_sequence_len:
                    continue

                # traverse all span pairs here
                # may split one dataset instance into several instances
                for subject in ent_pred:
                    for object in ent_pred:
                        if subject == object:
                            continue

                        if len(input_ids) + 4 > self.max_sequence_len:
                            # * if tokenized sequence is too long, add it to dataset and create a new instance for the rest span pairs
                            # ! if the original tokenized sequence (without span special tokens) is larger than max sequence length, discard this sentence

                            assert len(input_ids) == len(
                                position_ids
                            ), f"The length of input_ids and position_ids didn't match, got {len(input_ids)} and {len(position_ids)}"
                            assert len(origin_subject_index) == len(
                                origin_object_index
                            ) and len(origin_object_index) == len(
                                subject_index
                            ) and len(subject_index) == len(
                                object_index
                            ) and len(object_index) == len(
                                labels
                            ), f"The length of origin subject index, origin object index, subject index, object index, and labels didn't match, got {len(origin_subject_index)}, {len(origin_object_index)}, {len(subject_index)}, {len(object_index)}, {len(labels)}"
                            instance = {}
                            instance['input_ids'] = input_ids
                            instance['attention_mask'] = get_attention_mask(
                                tokens_type)
                            instance[
                                'origin_subject_index'] = origin_subject_index
                            instance[
                                'origin_object_index'] = origin_object_index
                            instance['span_pair_mask'] = [
                                True for _ in subject_index
                            ]
                            if len(subject_index) == 0:
                                subject_index.append([])
                                object_index.append([])
                            instance['subject_index'] = subject_index
                            instance['object_index'] = object_index
                            instance['tokens'] = dataset.tokens[i]
                            instance['position_ids'] = position_ids
                            instance['labels'] = labels
                            instance['span2ent'] = span2ent
                            instance['ent_pred'] = ent_pred
                            instance['span2rel'] = span2rel
                            new_dataset.append(Instance(**instance))

                            input_ids = input_ids[:origin_len]
                            position_ids = list(range(origin_len))
                            tokens_type = [1 for _ in input_ids]
                            origin_subject_index = []
                            origin_object_index = []
                            subject_index = []
                            object_index = []
                            labels = []
                            span2rel = []

                        subject_pos = subject[0]
                        object_pos = object[0]

                        origin_subject_index.append(subject_pos)
                        origin_object_index.append(object_pos)

                        if ((subject_pos, object_pos) in rels):
                            labels.append(self.tag_vocab['relation'].to_index(
                                rels[(subject_pos, object_pos)]))
                            rel_type = rels[(subject_pos, object_pos)]
                            span2rel.append(
                                (subject_pos, object_pos, rel_type))
                        else:
                            labels.append(0)

                        levitated_id = len(position_ids)
                        subject_index.append([levitated_id, levitated_id + 1])
                        object_index.append(
                            [levitated_id + 2, levitated_id + 3])

                        input_ids.append(
                            self.tokenizer.convert_tokens_to_ids(
                                f'<SUBJ_START={subject[1]}>'))
                        input_ids.append(
                            self.tokenizer.convert_tokens_to_ids(
                                f'<SUBJ_END={subject[1]}>'))
                        input_ids.append(
                            self.tokenizer.convert_tokens_to_ids(
                                f'<OBJ_START={object[1]}>'))
                        input_ids.append(
                            self.tokenizer.convert_tokens_to_ids(
                                f'<OBJ_END={object[1]}>'))

                        position_ids += [
                            tokenized_start[subject_pos[0] + sent_start],
                            tokenized_end[subject_pos[1] + sent_start - 1],
                            tokenized_start[object_pos[0] + sent_start],
                            tokenized_end[object_pos[1] + sent_start - 1]
                        ]
                        tokens_type += [tokens_type[-1] + 1] * 4

                assert len(input_ids) == len(
                    position_ids
                ), f"The length of input_ids and position_ids didn't match, got {len(input_ids)} and {len(position_ids)}"
                assert len(origin_subject_index) == len(
                    origin_object_index
                ) and len(origin_object_index) == len(subject_index) and len(
                    subject_index
                ) == len(object_index) and len(object_index) == len(
                    labels
                ), f"The length of origin subject index, origin object index, subject index, object index, and labels didn't match, got {len(origin_subject_index)}, {len(origin_object_index)}, {len(subject_index)}, {len(object_index)}, {len(labels)}"
                if len(subject_index) == 0:
                    subject_index.append([])
                    object_index.append([])
                instance = Instance(
                    input_ids=input_ids,
                    attention_mask=get_attention_mask(tokens_type),
                    origin_subject_index=origin_subject_index,
                    origin_object_index=origin_object_index,
                    span_pair_mask=[True for _ in origin_subject_index],
                    subject_index=subject_index,
                    object_index=object_index,
                    tokens=tokens,
                    position_ids=position_ids,
                    labels=labels,
                    span2ent=span2ent,
                    span2rel=span2rel,
                    ent_pred=ent_pred)
                new_dataset.append(instance)

            new_databundle.set_dataset(dataset=new_dataset, name=dataset_name)

        new_databundle.set_pad('tokens', pad_val=None)
        new_databundle.set_pad('origin_subject_index', pad_val=None)
        new_databundle.set_pad('origin_object_index', pad_val=None)
        new_databundle.set_pad('span2rel', pad_val=None)
        new_databundle.set_pad('span2ent', pad_val=None)
        return new_databundle

    def process_from_file(self, paths: str) -> DataBundle:
        return super().process_from_file(paths)
