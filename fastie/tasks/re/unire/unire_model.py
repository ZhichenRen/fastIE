import torch
import torch.nn as nn
import numpy as np
from fastNLP.transformers.torch import BertModel, BertTokenizer
from utils.nn_utils import batched_index_select
from modules import BertLinear


class UniRE(nn.Module):

    def __init__(self,
                 ent_rel_file,
                 bert_model_name='bert-base-uncased',
                 bert_output_size=768,
                 max_span_length=10,
                 separate_threshold=1.4,
                 mlp_hidden_size=150,
                 dropout=0.4,
                 logit_dropout=0.2,
                 bert_dropout=0):

        super().__init__()
        self.max_span_length = max_span_length
        self.activation = nn.GELU()
        self.separate_threshold = separate_threshold

        self.bert_model = BertModel.from_pretrained(bert_model_name)
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.encoder_output_size = bert_output_size

        self.head_mlp = nn.Linear(self.encoder_output_size, mlp_hidden_size)
        self.head_mlp.weight.data.normal_(mean=0.0, std=0.02)
        self.head_mlp.bias.data.zero_()
        self.tail_mlp = nn.Linear(self.encoder_output_size, mlp_hidden_size)
        self.tail_mlp.weight.data.normal_(mean=0.0, std=0.02)
        self.tail_mlp.bias.data.zero_()

        self.head_mlp = BertLinear(self.encoder_output_size, mlp_hidden_size,
                                   self.activation, dropout)
        self.tail_mlp = BertLinear(self.encoder_output_size, mlp_hidden_size,
                                   self.activation, dropout)

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = lambda x: x

        self.layer_norm = nn.LayerNorm(mlp_hidden_size)

        self.U = nn.Parameter(
            torch.FloatTensor(len(ent_rel_file['type2id']),
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

        self.none_idx = ent_rel_file['type2id']['None']

        self.symmetric_label = torch.LongTensor(ent_rel_file['symmetric'])
        self.asymmetric_label = torch.LongTensor(ent_rel_file['asymmetric'])
        self.ent_label = torch.LongTensor(ent_rel_file['entity'])
        self.rel_label = torch.LongTensor(ent_rel_file['relation'])

        self.ent_rel_file = ent_rel_file
        self.element_loss = nn.CrossEntropyLoss()

    def forward(self,
                tokens_len,
                wordpiece_tokens,
                wordpiece_segment_ids,
                wordpiece_tokens_index,
                joint_label_matrix=None,
                joint_label_matrix_mask=None,
                training=True):

        results = {}
        output = self.bert_model(input_ids=wordpiece_tokens,
                                 attention_mask=wordpiece_segment_ids)

        bert_output = self.bert_dropout(output[0])

        # bert_output here is [batch_size, wordpiece_tokens_len, embedding_len], should change to [batch_size, tokens_len, embedding_len]

        bert_output = batched_index_select(bert_output, wordpiece_tokens_index)

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

        if not training:
            results['joint_label_preds'] = torch.argmax(
                batch_normalized_joint_score, dim=-1)

            separate_position_preds, ent_preds, rel_preds = self.soft_joint_decoding(
                batch_normalized_joint_score, tokens_len)

            results['all_separate_position_preds'] = separate_position_preds
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
            ent_pred = {}
            rel_pred = {}
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
                if not (np.max(score[ent_label]) < score[self.none_idx]):
                    pred = ent_label[np.argmax(score[ent_label])].item()
                    pred_label = self.ent_rel_file['id2type'][pred]
                    ents.append(span)
                    ent_pred[span] = pred_label

            for ent1 in ents:
                for ent2 in ents:
                    if ent1 == ent2:
                        continue
                    score = np.mean(joint_score[ent1[0]:ent1[1],
                                                ent2[0]:ent2[1], :],
                                    axis=(0, 1))
                    if not (np.max(score[rel_label]) < score[self.none_idx]):
                        pred = rel_label[np.argmax(score[rel_label])].item()
                        pred_label = self.ent_rel_file['id2type'][pred]
                        rel_pred[(ent1, ent2)] = pred_label

            ent_preds.append(ent_pred)
            rel_preds.append(rel_pred)

        return separate_position_preds, ent_preds, rel_preds

    def train_step(self, tokens_len, wordpiece_tokens, wordpiece_segment_ids,
                   wordpiece_tokens_index, joint_label_matrix,
                   joint_label_matrix_mask):
        result = self.forward(tokens_len, wordpiece_tokens,
                              wordpiece_segment_ids, wordpiece_tokens_index,
                              joint_label_matrix, joint_label_matrix_mask)

        result['loss'] = result['element_loss'] + result[
            'implication_loss'] + result['symmetric_loss']
        return result

    def evaluate_step(self, tokens_len, wordpiece_tokens,
                      wordpiece_segment_ids, wordpiece_tokens_index,
                      joint_label_matrix, joint_label_matrix_mask, span2ent,
                      span2rel):
        result = self.forward(tokens_len,
                              wordpiece_tokens,
                              wordpiece_segment_ids,
                              wordpiece_tokens_index,
                              joint_label_matrix,
                              joint_label_matrix_mask,
                              training=False)
        result['ent_target'] = span2ent
        result['rel_target'] = span2rel
        result['ent_rel_file'] = self.ent_rel_file
        return result

    def inference_step(self, tokens, tokens_len, wordpiece_tokens,
                       wordpiece_segment_ids, wordpiece_tokens_index):
        output = self.forward(tokens_len,
                              wordpiece_tokens,
                              wordpiece_segment_ids,
                              wordpiece_tokens_index,
                              training=False)
        result = []
        for i in range(len(tokens)):
            result_i = {'entity': [], 'relation': []}
            for span, ent in output[i]['ent_pred']:
                result_i['entity'].append(
                    ((' ').join(tokens[span[0]:span[1]]), ent))

            for spans, rel in output[i]['rel_pred']:
                result_i['relation'].append(
                    (' '.join(tokens[spans[0][0], spans[0][1]]),
                     ' '.join(tokens[spans[1][0], spans[1][1]]), rel))

            result.append(result_i)
        return dict(pred=result)
