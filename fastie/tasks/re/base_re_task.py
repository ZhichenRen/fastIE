"""Base class for RE tasks."""
__all__ = ['BaseRETask', 'BaseRETaskConfig']
from fastie.tasks.base_task import BaseTask, BaseTaskConfig
from fastie.utils.utils import generate_tag_vocab, check_loaded_tag_vocab
from fastie.envs import logger, get_flag
from fastNLP import Vocabulary
from fastNLP.io import DataBundle
import abc
from typing import Dict, Union, Sequence, Optional


class BaseRETaskConfig(BaseTaskConfig):
    """RE 任务所需参数."""
    pass


class BaseRETask(BaseTask, metaclass=abc.ABCMeta):
    """FastIE RE 任务基类.

    :param load_model: 模型文件的路径或者模型名
    :param save_model_folder: ``topk`` 或 ``load_best_model`` 保存模型的文件夹
    :param batch_size: batch size
    :param epochs: 训练的轮数
    :param monitor: 根据哪个 ``metric`` 选择  ``topk`` 和 ``load_best_model``；
        如果不设置，则默认使用结果中的第一个 ``metric``
    :param is_large_better: ``metric`` 中 ``monitor`` 监控的指标是否越大越好
    :param topk: 将 ``metric`` 中 ``monitor`` 监控的指标最好的 k 个模型保存到
        ``save_model_folder`` 中
    :param load_best_model: 是否在训练结束后将 ``metric`` 中 ``monitor`` 监控的指标最
        好的模型保存到 ``save_model_folder`` 中，并自动加载到 ``task`` 中
    :param fp16: 是否使用混合精度训练
    :param evaluate_every: 训练过程中检验的频率,
        ``topk`` 和 ``load_best_model`` 将在所有的检验中选择:
            * 为 ``0`` 时则训练过程中不进行检验
            * 如果为正数，则每 ``evaluate_every`` 个 batch 进行一次检验
            * 如果为负数，则每 ``evaluate_every`` 个 epoch 进行一次检验
    :param device: 指定具体训练时使用的设备
        device 的可选输入如下所示:
            * *str*: 例如 ``'cpu'``, ``'cuda'``, ``'cuda:0'``, ``'cuda:1'``,
            `'gpu:0'`` 等；
            * *int*: 将使用 ``device_id`` 为该值的 ``gpu`` 进行训练；如果值为 -1，那么
            默认使用全部的显卡；
            * *list(int)*: 如果多于 1 个device，应当通过该种方式进行设定，将会启用分布式程序。
    """

    _config = BaseRETaskConfig()
    _help = 'Base class for RE tasks. '

    def __init__(self,
                 load_model: str = '',
                 save_model_folder: str = '',
                 batch_size: int = 32,
                 epochs: int = 200,
                 monitor: str = 'F-1#relation',
                 is_large_better: bool = True,
                 topk: int = 0,
                 topk_folder: str = '',
                 fp16: bool = False,
                 evaluate_every: int = -1,
                 device: Union[int, Sequence[int], str] = 'cpu',
                 use_strict: bool = True,
                 **kwargs):
        super().__init__(load_model, save_model_folder, batch_size, epochs,
                         monitor, is_large_better, topk, topk_folder, fp16,
                         evaluate_every, device, **kwargs)
        self.use_strict = use_strict

    def on_generate_and_check_tag_vocab(self,
                                        data_bundle: DataBundle,
                                        state_dict: Optional[dict]) \
            -> Dict[str, Vocabulary]:
        """根据数据集中每个样本 `sample['entity_motions'][i][1]` 生成标签词典。 如果加载模型得到的
        ``state_dict`` 中存在 ``tag_vocab``，则检查是否与根据 ``data_bundle`` 生成的 tag_vocab
        一致 (优先使用加载得到的 tag_vocab)。

        :param data_bundle: 原始数据集，
            可能包含 ``train``、``dev``、``test``、``infer`` 四种，需要分类处理。
        :param state_dict: 加载模型得到的 ``state_dict``，可能为 ``None``
        :return: 标签词典，可能为 ``None``
        """
        tag_vocab = {}
        if state_dict is not None and 'tag_vocab' in state_dict:
            tag_vocab = state_dict['tag_vocab']
        generated_tag_vocab = {}
        if get_flag() == 'train':
            generated_tag_vocab = generate_tag_vocab(data_bundle,
                                                     unknown='None')
        else:
            if 'entity' not in tag_vocab.keys() or 'relation' not in tag_vocab.keys() or 'joint' not in tag_vocab.keys():
                generated_tag_vocab = generate_tag_vocab(data_bundle,
                                                         unknown='None')
        for key, value in tag_vocab.items():
            if key not in generated_tag_vocab.keys():
                generated_tag_vocab[key] = check_loaded_tag_vocab(value,
                                                                  None)[1]
            else:
                logger.info(f'Checking tag vocab {key}...')
                signal, generated_tag_vocab[key] = check_loaded_tag_vocab(
                    value, generated_tag_vocab[key])
                if signal == -1:
                    logger.warning(
                        f'It is detected that the loaded ``{key}`` vocabulary '
                        f'conflicts with the generated ``{key}`` vocabulary, '
                        f'so the model loading may fail. ')
                logger.info('Check finished!')
        return generated_tag_vocab
