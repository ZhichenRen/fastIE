from fastNLP.io import Pipe, DataBundle
from fastNLP.transformers.torch import BertTokenizer
from fastNLP import DataSet, Vocabulary, Instance
from rich.progress import track
from typing import Dict, Tuple
from .handshake_tagger import HandshakingTaggingScheme


def is_in_bound(span: Tuple[int, int], start, end):
    """判断一个左闭右开Span是否在一个左闭右开区间内.

    Args:
        span (Tuple[int, int]): _description_
        start (_type_): _description_
        end (_type_): _description_

    Returns:
        _type_: _description_
    """
    return span[0] >= start and span[1] <= end


def split_sentence(tokenizer: BertTokenizer,
                   instances: DataSet,
                   max_seq_len: int = 100,
                   slide_window: int = 50):
    new_instances = DataSet()
    for instance in track(instances, description='spliting sentences ...'):
        tokens_len = len(instance['tokens'])
        sent_start = 0
        sent_id = 0
        entity_mentions = instance['entity_mentions'] if instances.has_field(
            'entity_mentions') else []
        relation_mentions = instance[
            'relation_mentions'] if instances.has_field(
                'relation_mentions') else []
        while sent_start < tokens_len:
            # 这里需要保证tokenize后的input_ids长度小于max_seq_len
            input_ids_len = 1
            sent_end = sent_start
            while input_ids_len <= max_seq_len - 1 and sent_end < tokens_len:
                input_ids_len += len(
                    tokenizer.tokenize(instance['tokens'][sent_end]))
                sent_end += 1
            if input_ids_len > max_seq_len - 1:
                # 长度超出限制，舍弃最后一个word
                sent_end -= 1
            tokens = instance['tokens'][sent_start:sent_end]
            split_entity_mentions = []
            split_relation_mentions = []
            for entity in entity_mentions:
                span = entity[0]
                if is_in_bound(span, sent_start, sent_end):
                    split_entity_mentions.append(
                        ((span[0] - sent_start, span[1] - sent_start),
                         entity[1]))

            for relation in relation_mentions:
                span_s = relation[0]
                span_o = relation[1]
                if is_in_bound(span_s, sent_start, sent_end) and is_in_bound(
                        span_o, sent_start, sent_end):
                    split_relation_mentions.append(
                        ((span_s[0] - sent_start, span_s[1] - sent_start),
                         (span_o[0] - sent_start,
                          span_o[1] - sent_start), relation[2]))
            new_instances.append(
                Instance(tokens=tokens,
                         sent_id=sent_id,
                         doc_key=instance['doc_key'],
                         entity_mentions=split_entity_mentions,
                         relation_mentions=split_relation_mentions))

            sent_id += 1
            sent_start += slide_window
    return new_instances


class TPLinkerPipe(Pipe):

    def __init__(self,
                 tag_vocab: Dict[str, Vocabulary],
                 train_tagger: HandshakingTaggingScheme,
                 valid_tagger: HandshakingTaggingScheme,
                 tokenizer: str = 'bert-base-uncased',
                 slide_window: int = 50) -> None:
        super().__init__()
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(
            tokenizer)
        self.tag_vocab = tag_vocab
        self.train_tagger = train_tagger
        self.valid_tagger = valid_tagger
        self.slide_window = slide_window

    def process(self, data_bundle: DataBundle) -> DataBundle:
        """_summary_

        Args:
            data_bundle (DataBundle): 一个采用了标准FastIE关系抽取格式的DataBundle

        Returns:
            DataBundle: 含有TPLinker模型所需要的字段的DataBundle，具体包含字段如下
            tokens:
            input_ids:
            attention_mask:
            token_type_ids:
            entity_shaking_tag:
            rel_head_shaking_tag:
            rel_tail_shaking_tag:
            span2rel:
        """
        new_databundle = DataBundle()
        for dataset_name in data_bundle.get_dataset_names():
            dataset = data_bundle.get_dataset(dataset_name)
            new_dataset = DataSet()
            if dataset_name == 'train':
                tagger = self.train_tagger
            else:
                tagger = self.valid_tagger
            max_seq_len = tagger.matrix_size
            dataset = split_sentence(self.tokenizer,
                                     dataset,
                                     max_seq_len,
                                     slide_window=self.slide_window)
            for instance in track(
                    dataset, description=f'Pre-processing {dataset_name}'):
                tokens = instance['tokens']

                # * turn tokens into input_ids, calculate attention_mask
                input_ids = [self.tokenizer.cls_token_id]
                tokens_start = []
                tokens_end = []
                wordpiece2token = {0: 0}
                for token_index, token in enumerate(tokens):
                    tokens_start.append(len(input_ids))
                    wordpiece_index_start = len(input_ids)
                    input_ids.extend(
                        self.tokenizer.encode(token, add_special_tokens=False))
                    tokens_end.append(len(input_ids) - 1)
                    wordpiece_index_end = len(input_ids)
                    for wordpiece_index in range(wordpiece_index_start,
                                                 wordpiece_index_end):
                        wordpiece2token[wordpiece_index] = token_index
                input_ids.append(self.tokenizer.sep_token_id)
                wordpiece2token[len(input_ids)] = len(tokens) - 1
                assert (
                    len(input_ids) <= max_seq_len
                ), f'Expect max token length {max_seq_len}, got {len(input_ids)}'
                attention_mask = [1 for _ in input_ids]
                pad_len = max_seq_len - len(input_ids)
                input_ids.extend(
                    [self.tokenizer.pad_token_id for _ in range(pad_len)])
                attention_mask.extend([0 for _ in range(pad_len)])

                token_type_ids = [1 for _ in input_ids]
                span2ent = instance[
                    'entity_mentions'] if 'entity_mentions' in instance.keys(
                    ) else []
                span2rel = instance[
                    'relation_mentions'] if 'relation_mentions' in instance.keys(
                    ) else []

                # * HandshakeTagger transfer entity and SPO list to TPLinker Handshake Tag and vice versa
                # * convert entity and relation to tokenized index first
                wordpiece_ents = [((tokens_start[ent[0][0]],
                                    tokens_end[ent[0][1] - 1]), ent[1])
                                  for ent in instance['entity_mentions']]
                wordpiece_rels = [((tokens_start[rel[0][0]],
                                    tokens_end[rel[0][1] - 1]),
                                   (tokens_start[rel[1][0]],
                                    tokens_end[rel[1][1] - 1]), rel[2])
                                  for rel in instance['relation_mentions']]

                # This will be used in handshake_tagger
                instance['relation_mentions'] = wordpiece_rels
                instance['entity_mentions'] = wordpiece_ents

                ent_spots, rel_head_spots, rel_tail_spots = tagger.get_spots(
                    instance)
                entity_shaking_tag = tagger.sharing_spots2shaking_tag(
                    ent_spots)
                rel_head_shaking_tag = tagger.spots2shaking_tag(rel_head_spots)
                rel_tail_shaking_tag = tagger.spots2shaking_tag(rel_tail_spots)

                new_dataset.append(
                    Instance(tokens=tokens,
                             input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             entity_shaking_tag=entity_shaking_tag,
                             rel_head_shaking_tag=rel_head_shaking_tag,
                             rel_tail_shaking_tag=rel_tail_shaking_tag,
                             span2ent=span2ent,
                             span2rel=span2rel,
                             wordpiece2token=wordpiece2token))
            new_databundle.set_dataset(new_dataset, dataset_name)
        new_databundle.set_pad('wordpiece2token', pad_val=None)
        new_databundle.set_pad('tokens', pad_val=None)
        new_databundle.set_pad('span2ent', pad_val=None)
        new_databundle.set_pad('span2rel', pad_val=None)
        return new_databundle

    def process_from_file(self, paths: str) -> DataBundle:
        return super().process_from_file(paths)
