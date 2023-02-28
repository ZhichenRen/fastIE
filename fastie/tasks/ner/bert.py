# -*- coding: UTF-8 -*- 
from dataclasses import dataclass, field
from functools import reduce
from io import BytesIO
from typing import Union, Sequence, Optional

import numpy as np
import torch
import torch.nn.functional as F
from fastNLP import prepare_dataloader, Instance, Vocabulary
from fastNLP.core.metrics import Accuracy
from fastNLP.io import DataBundle
from fastNLP.transformers.torch.models.bert import BertModel, BertConfig, \
    BertTokenizer
from torch import nn

from fastie.envs import get_flag
from fastie.tasks.BaseTask import BaseTask, BaseTaskConfig, NER


class Model(nn.Module):

    def __init__(self,
                 pretrained_model_name_or_path: Optional[str] = None,
                 num_labels: int = 9,
                 tag_vocab: Optional[Vocabulary] = None,
                 **kwargs):
        super(Model, self).__init__()
        if pretrained_model_name_or_path is not None:
            self.bert = BertModel.from_pretrained(
                pretrained_model_name_or_path)
        else:
            self.bert = BertModel(BertConfig(**kwargs))
        self.num_labels = num_labels
        self.classificationHead = nn.Linear(self._get_bert_embedding_dim(),
                                            num_labels)
        # 为了推理过程中能输出人类可读的结果，把 tag2label 也传进来
        self.tag_vocab = tag_vocab
        print(1)

    def _get_bert_embedding_dim(self):
        with torch.no_grad():
            temp = torch.zeros(1, 1).int().to(self.bert.device)
            return self.bert(temp).last_hidden_state.shape[-1]

    def forward(self, input_ids, attention_mask):
        features = self.bert(input_ids=input_ids,
                             attention_mask=attention_mask).last_hidden_state
        features = self.classificationHead(features)
        return dict(features=features)

    def train_step(self, input_ids, attention_mask, offset_mask,
                   entity_mentions):
        features = self.forward(input_ids, attention_mask)['features']
        loss = 0
        for b in range(features.shape[0]):
            logits = F.softmax(
                features[b][offset_mask[b].nonzero(), :].squeeze(1), dim=1)
            for entity_mention in entity_mentions[b]:
                target = torch.zeros(self.num_labels).to(features.device)
                target[entity_mention[1]] = 1
                for i in entity_mention[0]:
                    loss += F.binary_cross_entropy(logits[i], target)
        return dict(loss=loss)

    def evaluate_step(self, input_ids, attention_mask, offset_mask,
                      entity_mentions):
        features = self.forward(input_ids, attention_mask)['features']
        pred_list = []
        target_list = []
        max_len = 0
        for b in range(features.shape[0]):
            logits = F.softmax(
                features[b][offset_mask[b].nonzero(), :].squeeze(1), dim=1)
            pred = logits.argmax(dim=1).to(features.device)
            target = torch.zeros(pred.shape[0]).to(features.device)
            if pred.shape[0] > max_len: \
                    max_len = pred.shape[0]
            for entity_mention in entity_mentions[b]:
                for i in entity_mention[0]:
                    target[i] = entity_mention[1]
            pred_list.append(pred)
            target_list.append(target)
        pred = torch.stack(
            [F.pad(pred, (0, max_len - pred.shape[0])) for pred in pred_list])
        target = torch.stack([
            F.pad(target, (0, max_len - target.shape[0]))
            for target in target_list
        ])
        return dict(pred=pred, target=target)

    def inference_step(self, tokens, input_ids, attention_mask, offset_mask):
        features = self.forward(input_ids, attention_mask)['features']
        pred_list = []
        for b in range(features.shape[0]):
            logits = F.softmax(
                features[b][offset_mask[b].nonzero(), :].squeeze(1), dim=1)
            pred = logits.argmax(dim=1).to(features.device)
            pred_dict = {}
            pred_dict['tokens'] = tokens[b]
            pred_dict['entity_mentions'] = []
            for i in range(pred.shape[0]):
                # if pred[i] != 0:
                pred_dict['entity_mentions'].append(
                    ([i], self.tag_vocab.idx2word[int(pred[i])],
                     round(float(logits[i].max()), 3)))
            pred_list.append(pred_dict)
        # 推理的结果一定是可 json 化的，建议 List[Dict]，和输入的数据集的格式一致
        # 这里的结果是用户可读的，所以建议把 idx2label 存起来
        # 怎么存可以看一下下面 233 行
        return dict(pred=pred_list)


@dataclass
class BertNERConfig(BaseTaskConfig):
    pretrained_model_name_or_path: str = field(
        default='bert-base-uncased',
        metadata=dict(
            help='name of transformer model (see '
            'https://huggingface.co/transformers/pretrained_models.html for '
            'options).',
            existence=True))
    num_labels: int = field(default=9,
                            metadata=dict(
                                help='Number of label categories to predict.',
                                existence=True))


@NER.register_module('bert')
class BertNER(BaseTask):
    """用预训练模型 Bert 对 token 进行向量表征，然后通过 classification head 对每个 token 进行分类。"""
    # 必须在这里定义自己 config
    _config = BertNERConfig()
    # 帮助信息，会显示在命令行分组的帮助信息中
    _help = 'Use pre-trained BERT and a classification head to classify tokens'

    def __init__(
            self,
            pretrained_model_name_or_path: str = 'bert-base-uncased',
            batch_size: int = 32,
            lr: float = 2e-5,
            # 以下是父类的参数，也要复制过来，可以查看一下 BaseTask 参数
            cuda: Union[bool, int, Sequence[int]] = False,
            load_model: str = '',
            **kwargs):
        # 必须要把父类 （BaseTask）的参数也复制过来，否则用户没有父类的代码提示；
        # 在这里进行父类的初始化；
        # 父类的参数我们不需要进行任何操作，比如这里的 cuda 和 load_model，我们无视就可以了。
        super(BertNER, self).__init__(cuda=cuda,
                                      load_model=load_model,
                                      **kwargs)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        # __init__ 值用来初始化所有属性变量，不要进行任何耗时操作
        # 这里初始化的属性优先级是要比 global_config 低的
        self.batch_size = batch_size
        self.lr = lr
        self.load_model = load_model

        # 存储模型时要额外存的隐属性
        # 只存 model 权重的不用这些东西，save 和 load 函数也不用写
        # 建议保存简单的数据结构，比如这里的 tag2idx 保存的是个 dict 而不是 Vocabulary
        # 注意额外的参数，如果不想暴露给用户的话，变量名要以下划线开头
        self._tag2idx = None
        self._num_labels = None
        self._model_dict = None

    def run(self, data_bundle: DataBundle):
        # 注意，接收的 data_bundle 可能有用来 infer 的，也就是说没有 label 信息，
        # 预处理的时候要注意
        self.tokenizer = BertTokenizer.from_pretrained(
            # 对自己进行 getattr 优先级是首先从 global_config 取，
            # global_config 中没有这个值的话再从自己的 __dict__ 里面取
            self.pretrained_model_name_or_path)
        # _tag_vocab 为自动生成的 tag vocab
        _tag_vocab = None
        # 如果存在标注数据
        # 则自动创建 tag 到 id 的映射
        if 'train' in data_bundle.datasets.keys() \
                or 'dev' in data_bundle.datasets.keys() \
                or 'test' in data_bundle.datasets.keys():

            _tag_vocab: Vocabulary = Vocabulary(padding=None, unknown=None)

            def construct_vocab(instance: Instance):
                # 当然，用来 infer 的数据集是无法构建的，这里判断一下
                if 'entity_mentions' in instance.keys():
                    for entity_mention in instance['entity_mentions']:
                        _tag_vocab.add(entity_mention[1])
                return instance

            data_bundle.apply_more(construct_vocab)
            self._num_labels = len(list(_tag_vocab.word2idx.keys()))

        # 下面是检验的过程，需要检验三种情况
        # 1. 没 load，也没法自动生成，证明没加载任何模型就要推理，直接报错
        # 2. load 了，没法自动生成，这是 infer 的正常流程
        # 3. load 了，也能自动生成，证明用训练集 infer 或者接着 train，这里要检验一下
        #   3.1 如果是 train，那么就要检验一下 tag_vocab 是否一致，不一致要判断是否是生成过程中的随机性导致的
        #   3.2 如果不是 train，那么就不用检验了，直接用加载的 self._tag2idx 就好
        if self._tag2idx is None and _tag_vocab is None:
            # 两个都是 None，无法推理
            print('Unable to find a tag to id mapping. '
                  'Make sure that the model you load is a fastie model. ')
            exit(1)
        if self._tag2idx is not None and _tag_vocab is not None:
            # 两个都有
            if get_flag() == 'train':  # 证明想用我们训练好的 checkpoint 继续在自己的数据上训练
                # 如果不是 train 的话可能只是想用我们的模型做推理，只不过没特意做 infer 数据集，想在 train 上推理
                # 那就没必要报错了，直接用加载的 self._tag2idx 就好
                if self._tag2idx != _tag_vocab.word2idx:
                    # 这里可以分情况再讨论一下
                    # 如果 tag 都一样，只是 id 不一样，可能只是生成过程中的随机性导致的
                    if set(self._tag2idx.keys()) == _tag_vocab.word2idx.keys():
                        _tag_vocab._word2idx.update(self._tag2idx)
                        _tag_vocab._idx2word.update({
                            value: key
                            for key, value in self._tag2idx.items()
                        })
                    else:
                        # tag 文本都对不上
                        # 就直接不加载模型了
                        self._tag2idx = None
                        self._num_labels = None
                        self._model_dict = None
        if self._tag2idx is not None and _tag_vocab is None:
            # 正常的推理流程须走这里
            _tag_vocab = Vocabulary(padding=None, unknown=None)
            _tag_vocab._word2idx.update(self._tag2idx)
            _tag_vocab._idx2word.update(
                {value: key
                 for key, value in self._tag2idx.items()})

        # 将 token 转换为 id，由于 bpe 会改变 token 的长度，所以除了 input_ids，
        # attention_mask 意外还返回了 offset_mask，input_ids 里面哪些是原来的 token
        def tokenize(instance: Instance):
            result_dict = {}
            input_ids_list, attention_mask_list, offset_mask_list = [], [], []
            for token in instance['tokens']:
                tokenized_token = self.tokenizer([token],
                                                 is_split_into_words=True,
                                                 return_tensors='np',
                                                 return_attention_mask=True,
                                                 return_token_type_ids=False,
                                                 add_special_tokens=False)
                token_offset_mask = np.zeros(
                    tokenized_token['input_ids'].shape, dtype=int)
                token_offset_mask[0, 0] = 1
                input_ids_list.append(tokenized_token['input_ids'])
                attention_mask_list.append(tokenized_token['attention_mask'])
                offset_mask_list.append(token_offset_mask)
            input_ids = reduce(lambda x, y: np.concatenate((x, y), axis=1),
                               input_ids_list)[0, :]
            attention_mask = reduce(
                lambda x, y: np.concatenate((x, y), axis=1),
                attention_mask_list)[0, :]
            offset_mask = reduce(lambda x, y: np.concatenate((x, y), axis=1),
                                 offset_mask_list)[0, :]
            result_dict['input_ids'] = input_ids
            result_dict['attention_mask'] = attention_mask
            result_dict['offset_mask'] = offset_mask
            # 顺便把 tag 转化为 id
            if 'entity_mentions' in instance.keys():
                for i in range(len(instance['entity_mentions'])):
                    instance['entity_mentions'][i] = (
                        instance['entity_mentions'][i][0],
                        _tag_vocab.to_index(instance['entity_mentions'][i][1]))
                result_dict['entity_mentions'] = instance['entity_mentions']
            return result_dict

        data_bundle.apply_more(tokenize)
        # 为了 infer 的时候能够使用，我们把 tag_vocab 存起来
        self.model = Model(self.pretrained_model_name_or_path,
                           self._num_labels, _tag_vocab)
        if self._model_dict is not None:
            self.model.load_state_dict(self._model_dict)
        # Vocabulary 这种复杂的数据类型是不会存到配置文件中的，所以把 dict 类型的 word2idx
        # 存起来对自己进行 setattr 会同时把这个变量存到 global_config 中，便于后续导出
        # 前提条件：1. 是简单的数据类型，可以见 fastie.envs.type_dict，
        # 2. 变量名不以下划线开头
        # 因此想要保存起来的东西就直接存到 self 中吧。
        self.tag_vocab = _tag_vocab._word2idx
        # 此外，global_config 中已经存在的变量，不允许修改
        # 因此建议想要存储的变量不要修改
        # 比如下面两句不行的：
        # self.var = 1 // 存到了 global_config 中
        # self.var = 2 // 没有任何作用
        metrics = {'accuracy': Accuracy()}
        # 使用 get_flag 判断现在要进行的事情，可能的取值有 `train`, `eval`, `infer`,
        # `interact`
        if get_flag() == 'train':
            train_dataloader = prepare_dataloader(
                data_bundle.get_dataset('train'),
                batch_size=self.batch_size,
                shuffle=True)
            # 用户不一定需要验证集，所以这里要判断一下
            if 'dev' in data_bundle.datasets.keys():
                evaluate_dataloader = prepare_dataloader(
                    data_bundle.get_dataset('dev'),
                    batch_size=self.batch_size,
                    shuffle=True)
                parameters = dict(model=self.model,
                                  optimizers=torch.optim.Adam(
                                      self.model.parameters(), lr=self.lr),
                                  train_dataloader=train_dataloader,
                                  evaluate_dataloaders=evaluate_dataloader,
                                  metrics=metrics)
            else:
                parameters = dict(model=self.model,
                                  optimizers=torch.optim.Adam(
                                      self.model.parameters(), lr=self.lr),
                                  train_dataloader=train_dataloader)
        elif get_flag() == 'eval':
            evaluate_dataloader = prepare_dataloader(
                data_bundle.get_dataset('test'),
                batch_size=self.batch_size,
                shuffle=True)
            parameters = dict(model=self.model,
                              evaluate_dataloaders=evaluate_dataloader,
                              metrics=metrics)
        elif get_flag() == 'infer' or get_flag() == 'interact':
            # 注意：infer 和 eval 其实并没有区别，只是把 evaluate_dataloaders
            # 换成推理的数据集了；
            # 我们不需要管怎么推理的，只要在模型里面写好 inference_step 就可以了
            # 目前推理和交互对于任务来说没有任何区别
            infer_dataloader = prepare_dataloader(
                data_bundle.get_dataset('infer'),
                batch_size=self.batch_size,
                shuffle=True)
            parameters = dict(model=self.model,
                              evaluate_dataloaders=infer_dataloader,
                              driver='torch')
        return parameters

    # 自己有额外的数据需要存的话就要额外在自己的 task 类加上 state_dict 和 load_state_dict 方法
    # 否则只会存和读取 model 的 state_dict
    def state_dict(self) -> dict:
        return {
            'model': self.model.state_dict(),
            'tag2idx': self._tag2idx,
            'num_labels': self._num_labels
        }

    # 因为 checkpoint 里面的额外变量会影响到模型结构
    # 比如这里的 num_labels 会影响全连接层的输出维度
    # 因此没有自动进行的可行性，把这些因变量输入进来后自己判断吧
    # 比如 self._num_labels 为 None 的话就去找 train 数据集自动判断
    # 不为 None 检验一下和自动生成的是否一致
    # 注意额外的参数，如果不想暴露给用户的话，变量名要以下划线开头
    # 注意：task 里面的 load_state_dict 调用时机是 __init__ 阶段
    def load_state_dict(self, state_dict: dict):
        self._tag2idx = state_dict['tag2idx']
        self._num_labels = state_dict['num_labels']
        self._model_dict = state_dict['model']
