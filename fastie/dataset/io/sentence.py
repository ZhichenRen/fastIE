"""Sentence dataset for inference."""
__all__ = ['SentenceConfig', 'Sentence']

from dataclasses import dataclass, field
from typing import Union, Sequence, Optional

from fastNLP import DataSet, Instance, Vocabulary
from fastNLP.io import DataBundle

from fastie.dataset.base_dataset import DATASET, BaseDataset, BaseDatasetConfig


@dataclass
class SentenceConfig(BaseDatasetConfig):
    sentence: str = field(default='',
                          metadata=dict(help='Input a sequence as a dataset.',
                                        existence=True,
                                        nargs='+',
                                        multi_method='space-join'))


@DATASET.register_module('sentence')
class Sentence(BaseDataset):
    """Sentence dataset for inference (Only for inference).

    :param sentence: Input a sequence or sentences as a dataset (Use Spaces to separate tokens).
        For examples:

        .. code-block:: python
            data_bundle = Sentence(sentence='I love FastIE .').run()
            data_bundle = Sentence(sentence=['I love FastIE .', 'I love fastNLP .']).run()
    """
    _config = SentenceConfig()
    _help = 'Input a sequence or sentences as a dataset. (Only for inference). '

    def __init__(self,
                 sentence: Optional[Union[Sequence[str], str]] = None,
                 cache: bool = False,
                 refresh_cache: bool = False,
                 tag_vocab: Optional[Union[Vocabulary, dict]] = None,
                 **kwargs):
        super(Sentence, self).__init__(cache=cache,
                                       refresh_cache=refresh_cache,
                                       tag_vocab=tag_vocab,
                                       **kwargs)
        self.sentence = sentence

    def run(self):
        dataset = DataSet()
        sentences = [self.sentence] if isinstance(self.sentence,
                                                  str) else self.sentence
        for sentence in sentences:
            dataset.append(
                Instance(tokens=sentence.split(' '), doc_key=0, sent_id=0))
        data_bundle = DataBundle(datasets={'infer': dataset})
        return data_bundle
