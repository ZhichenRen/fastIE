"""PUREEntityTask."""
__all__ = ['PUREEntityTask', 'PUREEntityConfig']

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Sequence, Union
from fastNLP import Vocabulary, Callback
from fastNLP.io import DataBundle
from fastNLP.core.callbacks import TorchWarmupCallback

from fastie.tasks.base_task import RE
from fastie.tasks.re.base_re_task import BaseRETask, BaseRETaskConfig
from fastie.metrics import REMetric
from torch.optim import AdamW
from .pure_model import BertForEntity
from .pure_pipe import PUREEntityPipe


@dataclass
class PUREEntityConfig(BaseRETaskConfig):
    """PUREEntityTask 所需参数."""
    pretrained_model_name_or_path: str = field(
        default='bert-base-uncased',
        metadata=dict(
            help='name of transformer model (see '
            'https://huggingface.co/transformers/pretrained_models.html for '
            'options).',
            existence='train'))
    lr: float = field(default=5e-5,
                      metadata=dict(help='learning rate', existence='train'))


@RE.register_module('pure-entity')
class PUREEntityTask(BaseRETask):

    def __init__(self,
                 load_model: str = '',
                 save_model_folder: str = '',
                 batch_size: int = 32,
                 epochs: int = 200,
                 monitor: str = 'F-1#entity',
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
                 adam_beta_1: float = 0.9,
                 adam_beta_2: float = 0.9,
                 adam_epsilon: float = 1e-12,
                 adam_weight_decay_rate: float = 1e-5,
                 adam_bert_weight_decay_rate: float = 1e-5,
                 bert_lr_decay_rate: float = 0.9,
                 hidden_size: int = 768,
                 mlp_hidden_size: int = 150,
                 width_embedding_dim: int = 150,
                 max_span_length: int = 10,
                 bert_dropout: float = 0.1,
                 dropout: float = 0.2,
                 add_cross_sent: bool = True,
                 cross_sent_window: int = 200,
                 **kwargs):
        super().__init__(load_model, save_model_folder, batch_size, epochs,
                         monitor, is_large_better, topk, topk_folder, fp16,
                         evaluate_every, device)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.lr = lr
        self.bert_lr = bert_lr
        self.warmup_rate = warmup_rate
        self.adam_beta_1 = adam_beta_1
        self.adam_beta_2 = adam_beta_2
        self.adam_epsilon = adam_epsilon
        self.adam_weight_decay_rate = adam_weight_decay_rate
        self.adam_bert_weight_decay_rate = adam_bert_weight_decay_rate
        self.bert_lr_decay_rate = bert_lr_decay_rate
        self.hidden_size = hidden_size
        self.mlp_hidden_size = mlp_hidden_size
        self.width_embedding_dim = width_embedding_dim
        self.max_span_length = max_span_length
        self.bert_dropout = bert_dropout
        self.dropout = dropout
        self.add_cross_sent = add_cross_sent
        self.cross_sent_window = cross_sent_window

    def on_dataset_preprocess(self, data_bundle: DataBundle,
                              tag_vocab: Dict[str, Vocabulary],
                              state_dict: Optional[dict]) -> DataBundle:
        pipe = PUREEntityPipe(tag_vocab, self.pretrained_model_name_or_path,
                              self.max_span_length, self.add_cross_sent,
                              self.cross_sent_window)
        return pipe.process(data_bundle)

    def on_setup_model(self, data_bundle: DataBundle,
                       tag_vocab: Dict[str, Vocabulary],
                       state_dict: Optional[dict]):
        model = BertForEntity(tag_vocab, self.pretrained_model_name_or_path,
                              self.hidden_size, self.mlp_hidden_size,
                              self.width_embedding_dim, self.max_span_length,
                              self.bert_dropout, self.dropout)
        return model

    def on_setup_optimizers(self, model, data_bundle: DataBundle,
                            tag_vocab: Dict[str, Vocabulary],
                            state_dict: Optional[dict]):
        parameters = [(name, param)
                      for name, param in model.named_parameters()
                      if param.requires_grad]
        no_decay: List[str] = []
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

    def on_setup_callbacks(
            self, model, data_bundle: DataBundle, tag_vocab: Dict[str,
                                                                  Vocabulary],
            state_dict: Optional[dict]) -> Union[Callback, Sequence[Callback]]:
        callbacks = [TorchWarmupCallback(self.warmup_rate)]
        return callbacks

    def on_setup_metrics(self, model, data_bundle: DataBundle,
                         tag_vocab: Dict[str, Vocabulary],
                         state_dict: Optional[dict]) -> dict:
        re_metric = REMetric(tag_vocab, evaluate_relation=False)
        return {'re': re_metric}
