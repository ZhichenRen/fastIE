"""JsonLinesNER dataset for FastIE."""
__all__ = ['JsonLinesRE', 'JsonLinesREConfig']
import json
import os

from fastie.dataset.base_dataset import BaseDataset, DATASET, BaseDatasetConfig

from fastNLP import DataSet, Instance
from fastNLP.io import Loader, DataBundle

from dataclasses import dataclass, field
from typing import List


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
    """JsonLinesRE dataset for FastIE. Each row has a sample in json
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
            "sentences": [["I", "love", "FastIE", "."]],
            "ner": [[2, 3, "Task"], [4, 5, "Task"]],
            "relations": [[2, 3, 4, 5, "PART-OF"]]
        }

    :param folder: The folder where the data set resides.
    :param right_inclusive: When data is in the format of start and end,
        whether each span contains the token corresponding to end.
    :param cache: Whether to cache the dataset.
    :param refresh_cache: Whether to refresh the cache.
    """
    _config = JsonLinesREConfig()
    _help = 'JsonLinesRE dataset for FastIE. Each row has a sample in json format. '

    def __init__(self,
                 folder: str = '',
                 right_inclusive: bool = False,
                 symmetric_label: List[str] = [],
                 cache: bool = False,
                 refresh_cache: bool = False,
                 **kwargs):
        BaseDataset.__init__(self,
                             cache=cache,
                             refresh_cache=refresh_cache,
                             **kwargs)
        self.folder = folder
        self.right_inclusive = right_inclusive
        self.symmetric_label = symmetric_label

    def run(self) -> DataBundle:

        class UnifiedRELoader(Loader):

            def __init__(self, symmetric_label: List[str]) -> None:
                super(UnifiedRELoader, self).__init__()
                self.symmetric_label = symmetric_label

            def _load(self, path):
                ds = DataSet()
                with open(path, 'r', encoding='utf-8') as fin:
                    doc_id = 0
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
                                    assert ner[0] >= 0 and ner[1] < len(
                                        sents
                                    ), f'sentence len: {len(sents[sent_id])}, ner[0]: {ner[0]}, ner[1]: {ner[1]}'
                                    processed_ners.append(
                                        ((ner[0], ner[1] + 1), ner[2]))

                                for rel in rels:
                                    rel[0] -= sent_start
                                    rel[1] -= sent_start
                                    rel[2] -= sent_start
                                    rel[3] -= sent_start
                                    processed_rels.append(
                                        ((rel[0], rel[1] + 1),
                                         (rel[2], rel[3] + 1), rel[4]))
                                    if rel[4] in self.symmetric_label:
                                        processed_rels.append(
                                            ((rel[2], rel[3] + 1),
                                             (rel[0], rel[1] + 1), rel[4]))
                                doc_key = raw_sentences[
                                    'doc_key'] if 'doc_key' in raw_sentences else doc_id
                                ds.append(
                                    Instance(sent_id=sent_id,
                                             tokens=sents,
                                             entity_mentions=processed_ners,
                                             relation_mentions=processed_rels,
                                             doc_key=doc_key))

                                sent_start += len(sents)
                        doc_id += 1
                return ds

        data_bundle = UnifiedRELoader(
            symmetric_label=self.symmetric_label).load({
                file: os.path.join(self.folder, f'{file}.jsonl')
                for file in ('train', 'dev', 'test', 'infer')
                if os.path.exists(os.path.join(self.folder, f'{file}.jsonl'))
            })
        return data_bundle
