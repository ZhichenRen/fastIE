"""JsonLinesNER dataset for FastIE."""
__all__ = ['JsonLinesRE', 'JsonLinesREConfig']
import json
import os

from fastie.dataset.BaseDataset import BaseDataset, DATASET, BaseDatasetConfig

from fastNLP import DataSet, Instance
from fastNLP.io import Loader, DataBundle

from dataclasses import dataclass, field


@dataclass
class JsonLinesREConfig(BaseDatasetConfig):
    """JsonLinesRE 数据集配置类."""
    folder: str = field(
        default='',
        metadata=dict(help='The folder where the data set resides. '
                      'We will automatically read the possible train.jsonl, '
                      'dev.jsonl, test.jsonl and infer.jsonl in it. ',
                      existence=True))
    right_inclusive: bool = field(
        default=True,
        metadata=dict(
            help='When data is in the format of start and end, '
            'whether each span contains the token corresponding to end. ',
            existence=True))


@DATASET.register_module('jsonlines-re')
class JsonLinesRE(BaseDataset):
    """JsonLinesNER dataset for FastIE. Each row has a NER sample in json
    format:

    .. code-block:: json
        {
            "tokens": ["I", "love", "FastIE", "."],
            "entity_mentions": [
                {
                    "entity_index": [2],
                    "entity_type": "MISC"
                },
        }

    or:

    .. code-block:: json
        {
            "tokens": ["I", "love", "FastIE", "."],
            "entity_mentions": [
                {
                    "start": 2,
                    "end": 3,
                    "entity_type": "MISC"
                },
        }

    :param folder: The folder where the data set resides.
    :param right_inclusive: When data is in the format of start and end,
        whether each span contains the token corresponding to end.
    :param cache: Whether to cache the dataset.
    :param refresh_cache: Whether to refresh the cache.
    """
    _config = JsonLinesREConfig()
    _help = 'JsonLinesNER dataset for FastIE. Each row has a NER sample in json format. '

    def __init__(self,
                 folder: str = '',
                 right_inclusive: bool = False,
                 cache: bool = False,
                 refresh_cache: bool = False,
                 **kwargs):
        BaseDataset.__init__(self,
                             cache=cache,
                             refresh_cache=refresh_cache,
                             **kwargs)
        self.folder = folder
        self.right_inclusive = right_inclusive

    def run(self) -> DataBundle:

        class UnifiedRELoader(Loader):

            def __init__(self) -> None:
                super(UnifiedRELoader, self).__init__()

            def _load(self, path):
                ds = DataSet()
                with open(path, 'r', encoding='utf-8') as fin:
                    for line in fin:
                        raw_sentences = json.loads(line.strip())
                        if (len(raw_sentences) > 0):
                            sent_start = 0
                            for sent_id, (sents, ners, rels) in enumerate(
                                    zip(raw_sentences['sentences'],
                                        raw_sentences['ner'],
                                        raw_sentences['relations'])):
                                processed_ners = []
                                processed_rels = []
                                for ner in ners:
                                    ner[0] -= sent_start
                                    ner[1] -= sent_start
                                    processed_ners.append(
                                        ((ner[0], ner[1]), ner[2]))

                                for rel in rels:
                                    rel[0] -= sent_start
                                    rel[1] -= sent_start
                                    rel[2] -= sent_start
                                    rel[3] -= sent_start
                                    processed_rels.append(
                                        ((rel[0], rel[1]), (rel[2], rel[3]),
                                         rel[4]))
                                ds.append(
                                    Instance(
                                        sent_id=sent_id,
                                        tokens=sents,
                                        tokens_len=len(sents),
                                        entity_mentions=processed_ners,
                                        relation_mentions=processed_rels,
                                        doc_key=raw_sentences['doc_key'],
                                        clusters=raw_sentences['clusters']))

                                sent_start += len(sents)
                    return ds

        data_bundle = UnifiedRELoader().load({
            file: os.path.join(self.folder, f'{file}.jsonl')
            for file in ('train', 'dev', 'test', 'infer')
            if os.path.exists(os.path.join(self.folder, f'{file}.jsonl'))
        })
        return data_bundle
