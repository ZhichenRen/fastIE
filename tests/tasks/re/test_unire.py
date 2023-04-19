from fastie.tasks.re import UniRETask
from tests.dummy import dummy_re_dataset
from tests.utils import UnifiedTaskTest


class TestUniRE(UnifiedTaskTest):

    def setup_class(self):
        super().setup_class(self,
                            task_cls=UniRETask,
                            data_bundle=dummy_re_dataset(),
                            extra_parameters={'symmetric_label': ['PER-SOC']})
