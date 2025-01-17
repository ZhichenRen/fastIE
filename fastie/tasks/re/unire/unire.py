"""UniRETask."""
__all__ = ['UniRETask', 'UniREConfig']

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Sequence, Union
from fastNLP import Vocabulary, Callback
from fastNLP.io import DataBundle
from fastNLP.core.callbacks import TorchWarmupCallback

from fastie.tasks.base_task import RE
from fastie.tasks.re.base_re_task import BaseRETask, BaseRETaskConfig
from fastie.metrics import REMetric
from torch.optim import AdamW
from .unire_model import UniRE
from .unire_pipe import UniREPipe


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
                 load_model: str = '',
                 save_model_folder: str = '',
                 batch_size: int = 32,
                 epochs: int = 200,
                 monitor: str = 'F-1#relation_strict',
                 is_large_better: bool = True,
                 topk: int = 0,
                 topk_folder: str = '',
                 fp16: bool = False,
                 evaluate_every: int = -1,
                 device: Union[int, Sequence[int], str] = 'cpu',
                 pretrained_model_name_or_path: str = 'bert-base-uncased',
                 lr: float = 5e-5,
                 bert_lr: float = 5e-5,
                 warmup_rate: float = 0.2,
                 early_stop: int = 30,
                 adam_beta_1: float = 0.9,
                 adam_beta_2: float = 0.9,
                 adam_epsilon: float = 1e-12,
                 adam_weight_decay_rate: float = 1e-5,
                 adam_bert_weight_decay_rate: float = 1e-5,
                 bert_lr_decay_rate: float = 0.9,
                 dropout: float = 0.4,
                 logit_dropout: float = 0.2,
                 bert_dropout: float = 0.0,
                 mlp_hidden_size: int = 150,
                 separate_threshold: float = 1.4,
                 max_sent_len: int = 512,
                 max_wordpiece_len: int = 512,
                 max_span_len: int = 10,
                 add_cross_sent: bool = True,
                 cross_sent_len: int = 200,
                 symmetric_label: List[str] = [],
                 **kwargs):
        super().__init__(load_model, save_model_folder, batch_size, epochs,
                         monitor, is_large_better, topk, topk_folder, fp16,
                         evaluate_every, device, **kwargs)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.lr = lr
        self.bert_lr = bert_lr
        self.warmup_rate = warmup_rate
        self.early_stop = early_stop
        self.adam_beta_1 = adam_beta_1
        self.adam_beta_2 = adam_beta_2
        self.adam_epsilon = adam_epsilon
        self.adam_weight_decay_rate = adam_weight_decay_rate
        self.adam_bert_weight_decay_rate = adam_bert_weight_decay_rate
        self.bert_lr_decay_rate = bert_lr_decay_rate
        self.dropout = dropout
        self.logit_dropout = logit_dropout
        self.bert_dropout = bert_dropout
        self.mlp_hidden_size = mlp_hidden_size
        self.seperate_threshold = separate_threshold
        self.max_sent_len = max_sent_len
        self.max_wordpiece_len = max_wordpiece_len
        self.max_span_len = max_span_len
        self.add_cross_sent = add_cross_sent
        self.cross_sent_len = cross_sent_len
        self.symmetric_label = symmetric_label

    def on_dataset_preprocess(self, data_bundle: DataBundle,
                              tag_vocab: Dict[str, Vocabulary],
                              state_dict: Optional[dict]) -> DataBundle:
        unire_pipe = UniREPipe(tag_vocab=tag_vocab,
                               tokenizer=self.pretrained_model_name_or_path,
                               add_cross_sent=self.add_cross_sent,
                               cross_sent_len=self.cross_sent_len)
        return unire_pipe.process(data_bundle)

    def on_setup_model(self, data_bundle: DataBundle,
                       tag_vocab: Dict[str, Vocabulary],
                       state_dict: Optional[dict]):
        model = UniRE(tag_vocab=tag_vocab,
                      bert_model_name=self.pretrained_model_name_or_path,
                      max_span_length=self.max_span_len,
                      separate_threshold=self.seperate_threshold,
                      mlp_hidden_size=self.mlp_hidden_size,
                      dropout=self.dropout,
                      logit_dropout=self.logit_dropout,
                      bert_dropout=self.bert_dropout)
        return model

    def on_setup_optimizers(self, model, data_bundle: DataBundle,
                            tag_vocab: Dict[str, Vocabulary],
                            state_dict: Optional[dict]):
        parameters = [(name, param)
                      for name, param in model.named_parameters()
                      if param.requires_grad]
        no_decay: List[str] = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_layer_lr = {}
        base_lr = self.bert_lr
        for i in range(11, -1, -1):
            bert_layer_lr['.' + str(i) + '.'] = base_lr
            base_lr *= self.bert_lr_decay_rate

        optimizer_grouped_parameters = []
        for name, param in parameters:
            params = {'params': [param], 'lr': self.lr}
            if any(item in name for item in no_decay):
                params['weight_decay_rate'] = 0.0
            else:
                if 'bert' in name:
                    params[
                        'weight_decay_rate'] = self.adam_bert_weight_decay_rate
                else:
                    params['weight_decay_rate'] = self.adam_weight_decay_rate

            for bert_layer_name, lr in bert_layer_lr.items():
                if bert_layer_name in name:
                    params['lr'] = lr
                    break

            optimizer_grouped_parameters.append(params)

        optimizers = AdamW(optimizer_grouped_parameters,
                           betas=(self.adam_beta_1, self.adam_beta_2),
                           lr=self.lr,
                           eps=self.adam_epsilon,
                           weight_decay=self.adam_weight_decay_rate)
        return optimizers

    def on_setup_metrics(self, model, data_bundle: DataBundle,
                         tag_vocab: Dict[str, Vocabulary],
                         state_dict: Optional[dict]) -> dict:
        return {'re': REMetric(tag_vocab)}

    def on_get_state_dict(self, model, data_bundle: DataBundle,
                          tag_vocab: Dict[str, Vocabulary]) -> dict:
        state_dict = super().on_get_state_dict(model, data_bundle, tag_vocab)
        state_dict[
            'pretrained_model_name_or_path'] = self.pretrained_model_name_or_path
        return state_dict

    def on_setup_callbacks(
            self, model, data_bundle: DataBundle, tag_vocab: Dict[str,
                                                                  Vocabulary],
            state_dict: Optional[dict]) -> Union[Callback, Sequence[Callback]]:
        callbacks = [TorchWarmupCallback(self.warmup_rate)]
        return callbacks
