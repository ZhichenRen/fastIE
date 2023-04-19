"""TPLinkerTask."""
__all__ = ['TPLinkerTask', 'TPLinkerConfig']

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Sequence, Union
from fastNLP import Vocabulary, Callback
from fastNLP.io import DataBundle
from fastNLP.core.callbacks import TorchWarmupCallback

from fastie.tasks.base_task import RE
from fastie.tasks.re.base_re_task import BaseRETask, BaseRETaskConfig
from fastie.metrics import REMetric
from fastie.envs import logger
from torch.optim import AdamW
from .handshake_tagger import HandshakingTaggingScheme
from .tplinker_pipe import TPLinkerPipe
from .tplinker_model import TPLinker


@dataclass
class TPLinkerConfig(BaseRETaskConfig):
    """TPLinker 所需参数."""
    pretrained_model_name_or_path: str = field(
        default='bert-base-uncased',
        metadata=dict(
            help='name of transformer model (see '
            'https://huggingface.co/transformers/pretrained_models.html for '
            'options).',
            existence='train'))
    lr: float = field(default=5e-5,
                      metadata=dict(help='learning rate', existence='train'))


@RE.register_module('tplinker')
class TPLinkerTask(BaseRETask):

    def __init__(self,
                 load_model: str = '',
                 save_model_folder: str = '',
                 batch_size: int = 32,
                 epochs: int = 100,
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
                 train_max_len: int = 100,
                 valid_max_len: int = 200,
                 slide_window: int = 20,
                 shaking_type: str = 'cat',
                 kernel_encoder_type: str = 'lstm',
                 **kwargs):
        super().__init__(load_model, save_model_folder, batch_size, epochs,
                         monitor, is_large_better, topk, topk_folder, fp16,
                         evaluate_every, device)
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
        self.train_max_len = train_max_len
        self.valid_max_len = valid_max_len
        self.slide_window = slide_window
        self.shaking_type = shaking_type
        self.kernel_encoder_type = kernel_encoder_type

    def on_dataset_preprocess(self, data_bundle: DataBundle,
                              tag_vocab: Dict[str, Vocabulary],
                              state_dict: Optional[dict]) -> DataBundle:
        assert state_dict is not None, 'Error! Missing state_dict!'
        logger.info('Creating train tagger ...')
        train_tagger = HandshakingTaggingScheme(tag_vocab, self.train_max_len)
        logger.info('Creating valid tagger ...')
        valid_tagger = HandshakingTaggingScheme(tag_vocab, self.valid_max_len)
        state_dict['train_tagger'] = train_tagger
        state_dict['valid_tagger'] = valid_tagger
        logger.info('Start pre-processing ...')
        pipe = TPLinkerPipe(tag_vocab, train_tagger, valid_tagger,
                            self.pretrained_model_name_or_path,
                            self.slide_window)
        return pipe.process(data_bundle)

    def on_setup_model(self, data_bundle: DataBundle,
                       tag_vocab: Dict[str, Vocabulary],
                       state_dict: Optional[dict]):
        assert state_dict is not None, 'Error! Missing state_dict!'
        assert 'valid_tagger' in state_dict.keys(
        ), 'Error! valid handshaking tagger missing!'
        model = TPLinker(tag_vocab, state_dict['valid_tagger'],
                         self.pretrained_model_name_or_path, self.shaking_type,
                         self.kernel_encoder_type)
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
        re_metric = REMetric(tag_vocab)
        return {'re': re_metric}
