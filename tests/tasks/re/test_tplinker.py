from fastie.tasks.re import TPLinkerTask
from tests.dummy import dummy_re_dataset
from tests.utils import UnifiedTaskTest


class TestTPLinker(UnifiedTaskTest):

    def setup_class(self):
        super().setup_class(self,
                            task_cls=TPLinkerTask,
                            data_bundle=dummy_re_dataset(),
                            extra_parameters={})
