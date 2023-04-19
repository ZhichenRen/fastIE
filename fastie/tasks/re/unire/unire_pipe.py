from fastNLP.io import Pipe, DataBundle
from fastNLP.transformers.torch import BertTokenizer
from fastNLP import DataSet, Vocabulary, Instance
from fastie.utils.utils import add_cross_sentence
from rich.progress import track
from typing import Dict


class UniREPipe(Pipe):

    def __init__(self,
                 tag_vocab: Dict[str, Vocabulary],
                 tokenizer: str = 'bert-base-uncased',
                 add_cross_sent: bool = True,
                 cross_sent_len: int = 200) -> None:
        super().__init__()
        self.tag_vocab = tag_vocab
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(
            tokenizer)
        self.add_cross_sent = add_cross_sent
        self.cross_sent_len = cross_sent_len

    def process(self, data_bundle: DataBundle) -> DataBundle:
        """_summary_

        Args:
            data_bundle (DataBundle): _description_

        Returns:
            DataBundle: 包含以下字段
            input_ids: bert输入
            attention_mask: bert输入
            tokens: 原句子中的token
            tokens_len: 每个句子的长度
            tokens_index: 每个句子中的token在cross sentence tokens中的位置
            joint_label_matrix: 联合任务概率矩阵，具体含义见论文，大小为[tokens_len, tokens_len]
            joint_label_matrix_mask: 概率矩阵的mask
            span2ent: [((start, end), label), ...]
            span2rel: [((s_start, s_end), (o_start, o_end), label), ...]
        """
        new_databundle = DataBundle()
        for dataset_name in data_bundle.get_dataset_names():
            dataset = data_bundle.get_dataset(dataset_name)
            new_dataset = DataSet()

            if self.add_cross_sent:
                cross_sent_instances = add_cross_sentence(
                    dataset, self.cross_sent_len)

            for i in track(range(len(dataset)),
                           description=f'Processing {dataset_name} ...'):
                tokens_len = len(dataset.tokens[i])
                if self.add_cross_sent:
                    sent_start = cross_sent_instances[i]['sent_start']
                    sent_end = sent_start + tokens_len
                    cross_sent_tokens = cross_sent_instances[i][
                        'cross_sent_tokens']
                else:
                    sent_start = 0
                    sent_end = sent_start + tokens_len
                    cross_sent_tokens = dataset.tokens[i]

                input_ids = [self.tokenizer.cls_token_id]
                tokens_start = []
                tokens_end = []
                for token in cross_sent_tokens:
                    tokens_start.append(len(input_ids))
                    input_ids.extend(
                        self.tokenizer.encode(token, add_special_tokens=False))
                    tokens_end.append(len(input_ids) - 1)
                input_ids.append(self.tokenizer.sep_token_id)
                attention_mask = [True for _ in input_ids]

                tokens_index = tokens_start[sent_start:sent_end]

                span2ent = dataset.entity_mentions[i] if dataset.has_field(
                    'entity_mentions') else []
                span2rel = dataset.relation_mentions[i] if dataset.has_field(
                    'relation_mentions') else []

                none_id = self.tag_vocab['joint'].to_index('None')
                joint_label_matrix = [[none_id for _ in range(tokens_len)]
                                      for _ in range(tokens_len)]
                joint_label_matrix_mask = [[True for _ in range(tokens_len)]
                                           for _ in range(tokens_len)]

                for ent in span2ent:
                    ent_label_idx = self.tag_vocab['joint'].to_index(ent[1])
                    assert 0 <= ent[0][0] and tokens_len >= ent[0][
                        1], f'Index {ent[0]} out of bound {tokens_len}'
                    for row in range(ent[0][0], ent[0][1]):
                        for col in range(ent[0][0], ent[0][1]):
                            joint_label_matrix[row][col] = ent_label_idx

                for rel in span2rel:
                    rel_label_idx = self.tag_vocab['joint'].to_index(rel[2])
                    assert 0 <= rel[0][0] and tokens_len >= rel[0][
                        1], f'Index {rel[0]} out of bound {tokens_len}'
                    assert 0 <= rel[0][0] and tokens_len >= rel[0][
                        1], f'Index {rel[1]} out of bound {tokens_len}'
                    for row in range(rel[0][0], rel[0][1]):
                        for col in range(rel[1][0], rel[1][1]):
                            joint_label_matrix[row][col] = rel_label_idx

                new_dataset.append(
                    Instance(input_ids=input_ids,
                             attention_mask=attention_mask,
                             tokens=dataset.tokens[i],
                             tokens_len=tokens_len,
                             tokens_index=tokens_index,
                             joint_label_matrix=joint_label_matrix,
                             joint_label_matrix_mask=joint_label_matrix_mask,
                             span2ent=span2ent,
                             span2rel=span2rel))

            new_databundle.set_dataset(new_dataset, dataset_name)
        new_databundle.set_pad('span2ent', pad_val=None)
        new_databundle.set_pad('span2rel', pad_val=None)
        new_databundle.set_pad('tokens', pad_val=None)

        return new_databundle

    def process_from_file(self, paths: str) -> DataBundle:
        return super().process_from_file(paths)
