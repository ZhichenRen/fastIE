import torch
import torch.nn as nn
import numpy as np
from fastNLP.transformers.torch import BertModel, BertTokenizer
from fastNLP import Vocabulary, Instance
from typing import Dict, List

from fastie.utils.nn_utils import batched_index_select
from fastie.modules import BertLinear


class UniRE(nn.Module):

    def __init__(self,
                 tag_vocab: Dict[str, Vocabulary],
                 symmetric_label: List[str] = [],
                 bert_model_name: str = 'bert-base-uncased',
                 bert_output_size: int = 768,
                 max_span_length: int = 10,
                 separate_threshold: float = 1.4,
                 mlp_hidden_size: int = 150,
                 dropout: float = 0.4,
                 logit_dropout: float = 0.2,
                 bert_dropout: float = 0):

        super().__init__()
        keys = tag_vocab.keys()
        assert 'entity' in keys and 'relation' in keys and 'joint' in keys, 'tag vocab should contain key "entity", "relation", and "joint".'
        self.tag_vocab = tag_vocab
        self.max_span_length = max_span_length
        self.activation = nn.GELU()
        self.separate_threshold = separate_threshold

        self.bert_model = BertModel.from_pretrained(bert_model_name)
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)

        self.head_mlp = BertLinear(bert_output_size, mlp_hidden_size,
                                   self.activation, dropout)
        self.tail_mlp = BertLinear(bert_output_size, mlp_hidden_size,
                                   self.activation, dropout)

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = lambda x: x

        self.layer_norm = nn.LayerNorm(mlp_hidden_size)

        self.U = nn.Parameter(
            torch.FloatTensor(len(tag_vocab['joint'].word2idx),
                              mlp_hidden_size + 1, mlp_hidden_size + 1))
        self.U.data.zero_()

        if logit_dropout > 0:
            self.logit_dropout = nn.Dropout(p=logit_dropout)
        else:
            self.logit_dropout = lambda x: x

        if bert_dropout > 0:
            self.bert_dropout = nn.Dropout(p=bert_dropout)
        else:
            self.bert_dropout = lambda x: x

        self.none_idx = self.tag_vocab['joint']['None']
        for ent in tag_vocab['entity'].word2idx.keys():
            if ent not in symmetric_label:
                symmetric_label.append(ent)
        symmetric_label_idx = [
            tag_vocab['joint'].to_index(label) for label in symmetric_label
            if label != 'None'
        ]
        self.symmetric_label = torch.LongTensor(symmetric_label_idx)
        self.ent_label = torch.LongTensor([
            tag_vocab['joint'].to_index(label)
            for label in tag_vocab['entity'].word2idx.keys() if label != 'None'
        ])
        self.rel_label = torch.LongTensor([
            tag_vocab['joint'].to_index(label)
            for label in tag_vocab['relation'].word2idx.keys()
            if label != 'None'
        ])
        self.element_loss = nn.CrossEntropyLoss()

    def forward(self,
                tokens_len,
                input_ids,
                attention_mask,
                tokens_index,
                joint_label_matrix_mask,
                joint_label_matrix=None):
        results = {}
        output = self.bert_model(input_ids=input_ids,
                                 attention_mask=attention_mask)

        bert_output = self.bert_dropout(output[0])

        # bert_output here is [batch_size, wordpiece_tokens_len, embedding_len], should change to [batch_size, tokens_len, embedding_len]

        bert_output = batched_index_select(bert_output, tokens_index)

        batch_seq_tokens_head_repr = self.head_mlp(bert_output)
        batch_seq_tokens_head_repr = torch.cat([
            batch_seq_tokens_head_repr,
            torch.ones_like(batch_seq_tokens_head_repr[..., :1])
        ],
                                               dim=-1)
        batch_seq_tokens_tail_repr = self.tail_mlp(bert_output)
        batch_seq_tokens_tail_repr = torch.cat([
            batch_seq_tokens_tail_repr,
            torch.ones_like(batch_seq_tokens_tail_repr[..., :1])
        ],
                                               dim=-1)

        batch_joint_score = torch.einsum('bxi, oij, byj -> boxy',
                                         batch_seq_tokens_head_repr, self.U,
                                         batch_seq_tokens_tail_repr).permute(
                                             0, 2, 3, 1)

        batch_normalized_joint_score = torch.softmax(
            batch_joint_score,
            dim=-1) * joint_label_matrix_mask.unsqueeze(-1).float()

        if joint_label_matrix is None:
            results['joint_label_preds'] = torch.argmax(
                batch_normalized_joint_score, dim=-1)

            separate_position_preds, ent_preds, rel_preds = self.soft_joint_decoding(
                batch_normalized_joint_score, tokens_len)

            # results['all_separate_position_preds'] = separate_position_preds
            results['ent_pred'] = ent_preds
            results['rel_pred'] = rel_preds

            return results

        results['element_loss'] = self.element_loss(
            self.logit_dropout(batch_joint_score[joint_label_matrix_mask]),
            joint_label_matrix[joint_label_matrix_mask])

        batch_rel_normalized_joint_score = torch.max(
            batch_normalized_joint_score[..., self.rel_label], dim=-1).values
        batch_diag_ent_normalized_joint_score = torch.max(
            batch_normalized_joint_score[...,
                                         self.ent_label].diagonal(0, 1, 2),
            dim=1).values.unsqueeze(-1).expand_as(
                batch_rel_normalized_joint_score)

        results['implication_loss'] = (
            torch.relu(batch_rel_normalized_joint_score -
                       batch_diag_ent_normalized_joint_score).sum(dim=2) +
            torch.relu(
                batch_rel_normalized_joint_score.transpose(1, 2) -
                batch_diag_ent_normalized_joint_score).sum(dim=2)
        )[joint_label_matrix_mask[..., 0]].mean()

        batch_symmetric_normalized_joint_score = batch_normalized_joint_score[
            ..., self.symmetric_label]

        results['symmetric_loss'] = torch.abs(
            batch_symmetric_normalized_joint_score -
            batch_symmetric_normalized_joint_score.transpose(1, 2)).sum(
                dim=-1)[joint_label_matrix_mask].mean()

        return results

    def soft_joint_decoding(self, batch_normalized_joint_score,
                            batch_seq_tokens_lens):
        """soft_joint_decoding extracts entity and relation at the same time,
        and consider the interconnection of entity and relation.

        Args:
            batch_normalized_joint_score (tensor): batch normalized joint score
            batch_seq_tokens_lens (list): batch sequence length

        Returns:
            tuple: predicted entity and relation
        """

        separate_position_preds = []
        ent_preds = []
        rel_preds = []

        batch_normalized_joint_score = batch_normalized_joint_score.cpu(
        ).detach().numpy()
        symmetric_label = self.symmetric_label.cpu().numpy()
        ent_label = self.ent_label.cpu().numpy()
        rel_label = self.rel_label.cpu().numpy()

        for idx, seq_len in enumerate(batch_seq_tokens_lens):
            ent_pred = []
            rel_pred = []
            joint_score = batch_normalized_joint_score[
                idx][:seq_len, :seq_len, :]
            # set value for symmetric labels
            # i.e. score[i,j,s_l] = score[j,i,s_l] = (score_old[i,j,s_l] + score_old[j,i,s_l]) / 2
            joint_score[..., symmetric_label] = (
                joint_score[..., symmetric_label] +
                joint_score[..., symmetric_label].transpose((1, 0, 2))) / 2

            # calculate vector representation of rows and columns to find out entity boundary
            joint_score_feature = joint_score.reshape(seq_len, -1)
            transposed_joint_score_feature = joint_score.transpose(
                (1, 0, 2)).reshape(seq_len, -1)
            separate_pos = (
                (np.linalg.norm(joint_score_feature[0:seq_len - 1] -
                                joint_score_feature[1:seq_len],
                                axis=1) +
                 np.linalg.norm(transposed_joint_score_feature[0:seq_len - 1] -
                                transposed_joint_score_feature[1:seq_len],
                                axis=1)) * 0.5 >
                self.separate_threshold).nonzero()[0]
            separate_position_preds.append(
                [pos.item() for pos in separate_pos])
            if len(separate_pos) > 0:
                spans = [(0, separate_pos[0].item() + 1),
                         (separate_pos[-1].item() + 1, seq_len)
                         ] + [(separate_pos[idx].item() + 1,
                               separate_pos[idx + 1].item() + 1)
                              for idx in range(len(separate_pos) - 1)]
            else:
                spans = [(0, seq_len.item())]

            ents = []
            for span in spans:
                score = np.mean(joint_score[span[0]:span[1],
                                            span[0]:span[1], :],
                                axis=(0, 1))
                if not (np.max(score[ent_label]) <= score[self.none_idx]):
                    pred = ent_label[np.argmax(score[ent_label])].item()
                    pred_label = self.tag_vocab['joint'].to_word(pred)
                    ents.append(span)
                    ent_pred.append((span, pred_label))

            for ent1 in ents:
                for ent2 in ents:
                    if ent1 == ent2:
                        continue
                    score = np.mean(joint_score[ent1[0]:ent1[1],
                                                ent2[0]:ent2[1], :],
                                    axis=(0, 1))
                    if not (np.max(score[rel_label]) <= score[self.none_idx]):
                        pred = rel_label[np.argmax(score[rel_label])].item()
                        pred_label = self.tag_vocab['joint'].to_word(pred)
                        rel_pred.append((ent1, ent2, pred_label))

            ent_preds.append(ent_pred)
            rel_preds.append(rel_pred)

        return separate_position_preds, ent_preds, rel_preds

    def train_step(self, tokens_len, input_ids, attention_mask, tokens_index,
                   joint_label_matrix, joint_label_matrix_mask):
        result = self.forward(tokens_len=tokens_len,
                              input_ids=input_ids,
                              attention_mask=attention_mask,
                              tokens_index=tokens_index,
                              joint_label_matrix=joint_label_matrix,
                              joint_label_matrix_mask=joint_label_matrix_mask)

        result['loss'] = result['element_loss'] + result[
            'implication_loss'] + result['symmetric_loss']
        return result

    def evaluate_step(self, tokens_len, input_ids, attention_mask,
                      tokens_index, span2ent, span2rel,
                      joint_label_matrix_mask):
        result = self.forward(tokens_len=tokens_len,
                              input_ids=input_ids,
                              attention_mask=attention_mask,
                              tokens_index=tokens_index,
                              joint_label_matrix_mask=joint_label_matrix_mask)

        result['ent_target'] = span2ent
        result['rel_target'] = span2rel
        return result

    def infer_step(self, tokens, tokens_len, input_ids, attention_mask,
                   tokens_index, joint_label_matrix_mask):
        output = self.forward(tokens_len=tokens_len,
                              input_ids=input_ids,
                              attention_mask=attention_mask,
                              tokens_index=tokens_index,
                              joint_label_matrix_mask=joint_label_matrix_mask)
        instances = []
        for i in range(len(tokens)):
            instances.append(
                Instance(tokens=tokens[i],
                         entity_mentions=output['ent_pred'][i],
                         relation_mentions=output['rel_pred'][i]))
        return dict(pred=instances)
