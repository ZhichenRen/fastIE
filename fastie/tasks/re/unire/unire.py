"""UniRE."""
__all__ = ['UniRETask', 'UniREConfig']

from dataclasses import dataclass, field
from typing import Optional, Dict
from fastNLP import Vocabulary
from fastNLP.io import DataBundle

from fastie.tasks.BaseTask import RE
from fastie.tasks.re.BaseRETask import BaseRETask, BaseRETaskConfig
from fastie.metrics import REMetric


@dataclass
class UniREConfig(BaseRETaskConfig):
    """UniRE 所需参数."""
    pretrained_model_name_or_path: str = field(
        default='bert-base-uncased',
        metadata=dict(
            help='name of transformer model (see '
            'https://huggingface.co/transformers/pretrained_models.html for '
            'options).',
            existence='train'))
    lr: float = field(default=5e-5,
                      metadata=dict(help='learning rate', existence='train'))


@RE.register_module('unire')
class UniRETask(BaseRETask):
    """实现 UNIRE: A Unified Label Space for Entity Relation Extraction
    论文中的关系抽取模型.

    :param pretrained_model_name_or_path: transformers 预训练 BERT 模型名字或路径.
        (see https://huggingface.co/models for options).
    :param lr: 学习率
    """

    _config = UniREConfig()
    _help = 'Use UniRE model to extract entities and relations from sentence'

    def __init__(self,
                 pretrained_model_name_or_path: str = 'bert-base-uncased',
                 lr: float = 5e-5,
                 **kwargs):
        super().__init__(**kwargs)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.lr = lr

    def on_dataset_preprocess(self, data_bundle: DataBundle,
                              tag_vocab: Dict[str, Vocabulary],
                              state_dict: Optional[dict]) -> DataBundle:
        return super().on_dataset_preprocess(data_bundle, tag_vocab,
                                             state_dict)

    def on_setup_model(self, data_bundle: DataBundle,
                       tag_vocab: Dict[str, Vocabulary],
                       state_dict: Optional[dict]):
        return super().on_setup_model(data_bundle, tag_vocab, state_dict)

    def on_setup_optimizers(self, model, data_bundle: DataBundle,
                            tag_vocab: Dict[str, Vocabulary],
                            state_dict: Optional[dict]):
        return {'re': REMetric(tag_vocab)}
