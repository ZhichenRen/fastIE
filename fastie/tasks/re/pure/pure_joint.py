"""PUREJointTask."""
__all__ = ['PUREJointTask', 'PUREJointConfig']

from fastie.tasks import SequentialTask
from fastie.tasks.re.base_re_task import BaseRETask, BaseRETaskConfig
from fastie.controller import Trainer, Evaluator, Inference
from fastNLP.io import DataBundle
from fastNLP import DataSet
from .pure_entity import PUREEntityTask
from .pure_relation import PURERelationTask
from fastie.tasks.base_task import RE
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PUREJointConfig(BaseRETaskConfig):
    """PUREJointTask 所需参数."""
    pretrained_model_name_or_path: str = field(
        default='bert-base-uncased',
        metadata=dict(
            help='name of transformer model (see '
            'https://huggingface.co/transformers/pretrained_models.html for '
            'options).',
            existence='train'))
    lr: float = field(default=5e-5,
                      metadata=dict(help='learning rate', existence='train'))


@RE.register_module('pure')
class PUREJointTask(SequentialTask):

    def __init__(self,
                 entity_model_path='',
                 relation_model_path='',
                 entity_task: Optional[BaseRETask] = None,
                 relation_task: Optional[BaseRETask] = None,
                 **kwargs):
        super().__init__()
        self.entity_model_path = entity_model_path
        self.relation_model_path = relation_model_path
        self.entity_task = entity_task
        self.relation_task = relation_task
        self.kwargs = kwargs

    def _build_tasks(self):
        if self.entity_task is None and self.relation_task is None:
            self.entity_task = PUREEntityTask(
                load_model=self.entity_model_path, **self.kwargs)
            self.relation_task = PURERelationTask(
                load_model=self.relation_model_path, **self.kwargs)

    def on_train(self, data_bundle: DataBundle):
        self._build_tasks()
        assert self.entity_task and self.relation_task, 'Error! Either entity_task or relation_task is None!'
        data_bundle.set_dataset(data_bundle.get_dataset('dev'), 'infer')
        entity_params = self.entity_task.run(data_bundle)
        Trainer().run(entity_params)
        instances = Inference().run(entity_params)
        data_bundle.set_dataset(DataSet(instances), 'dev')
        relation_params = self.relation_task.run(data_bundle)
        Trainer().run(relation_params)
        yield self.relation_task._on_get_state_dict_cache

    def on_eval(self, data_bundle: DataBundle):
        self._build_tasks()
        assert self.entity_task and self.relation_task, 'Error! Either entity_task or relation_task is None!'
        data_bundle.set_dataset(data_bundle.get_dataset('test'), 'infer')
        entity_params = self.entity_task.run(data_bundle)
        ent_result = Evaluator().run(entity_params)
        instances = Inference().run(entity_params)
        data_bundle.set_dataset(DataSet(instances), 'test')
        relation_params = self.relation_task.run(data_bundle)
        rel_result = Evaluator().run(relation_params)
        result = {'entity': ent_result, 'relation': rel_result}
        yield result

    def on_infer(self, data_bundle: DataBundle):
        self._build_tasks()
        assert self.entity_task and self.relation_task, 'Error! Either entity_task or relation_task is None!'
        entity_params = self.entity_task.run(data_bundle)
        instances = Inference().run(entity_params)
        data_bundle.set_dataset(DataSet(instances), 'infer')
        relation_params = self.relation_task.run(data_bundle)
        result = Inference().run(relation_params)
        yield result
