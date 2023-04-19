import torch
import torch.nn as nn
from typing import Dict
from fastNLP import Vocabulary, Instance
from fastNLP.transformers.torch import BertModel
from .handshaking_kernel import HandshakingKernel
from .handshake_tagger import HandshakingTaggingScheme


class TPLinker(nn.Module):

    def __init__(self,
                 tag_vocab: Dict[str, Vocabulary],
                 valid_tagger: HandshakingTaggingScheme,
                 bert_model_name: str = 'bert-base-uncased',
                 shaking_type: str = 'cat',
                 kernel_encoder_type: str = 'lstm'):
        super().__init__()
        assert 'entity' in tag_vocab.keys(
        ), 'key "entity" is not in tag_vocab!'
        assert 'relation' in tag_vocab.keys(
        ), 'key "relation" is not in tag_vocab!'
        self.tag_vocab = tag_vocab
        self.rel_size = len(tag_vocab['relation'].word2idx) - 1
        self.valid_tagger = valid_tagger
        self.bert = BertModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size

        # Modified
        entity_num = len(tag_vocab['entity'].word2idx)
        self.ent_fc = nn.Linear(hidden_size, entity_num)
        self.head_rel_fc_list = [
            nn.Linear(hidden_size, 3) for _ in range(self.rel_size)
        ]
        self.tail_rel_fc_list = [
            nn.Linear(hidden_size, 3) for _ in range(self.rel_size)
        ]

        for ind, fc in enumerate(self.head_rel_fc_list):
            self.register_parameter('weight_4_head_rel{}'.format(ind),
                                    fc.weight)
            self.register_parameter('bias_4_head_rel{}'.format(ind), fc.bias)
        for ind, fc in enumerate(self.tail_rel_fc_list):
            self.register_parameter('weight_4_tail_rel{}'.format(ind),
                                    fc.weight)
            self.register_parameter('bias_4_tail_rel{}'.format(ind), fc.bias)

        # handshaking kernel
        # 用于将bert的输出转换为handshake格式，由entity与各relation的分类器进行分类
        self.handshaking_kernel = HandshakingKernel(hidden_size, shaking_type,
                                                    kernel_encoder_type)

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                entity_shaking_tag=None,
                rel_head_shaking_tag=None,
                rel_tail_shaking_tag=None):
        output = self.bert(input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids)
        seq_output = output[0]
        handshaking_output = self.handshaking_kernel(seq_output)
        shaking_hiddens4ent = handshaking_output
        shaking_hiddens4rel = handshaking_output
        ent_shaking_outputs = self.ent_fc(shaking_hiddens4ent)

        rel_head_shaking_outputs_list = []
        for fc in self.head_rel_fc_list:
            rel_head_shaking_outputs_list.append(fc(shaking_hiddens4rel))

        rel_tail_shaking_outputs_list = []
        for fc in self.tail_rel_fc_list:
            rel_tail_shaking_outputs_list.append(fc(shaking_hiddens4rel))

        rel_head_shaking_outputs = torch.stack(rel_head_shaking_outputs_list,
                                               dim=1)
        rel_tail_shaking_outputs = torch.stack(rel_tail_shaking_outputs_list,
                                               dim=1)

        if entity_shaking_tag is not None and rel_head_shaking_tag is not None and rel_tail_shaking_tag is not None:
            loss = nn.CrossEntropyLoss()
            ent_loss = loss(
                ent_shaking_outputs.view(-1, ent_shaking_outputs.size(-1)),
                entity_shaking_tag.view(-1))
            rel_head_loss = loss(
                rel_head_shaking_outputs.view(
                    -1, rel_head_shaking_outputs.size(-1)),
                rel_head_shaking_tag.view(-1))
            rel_tail_loss = loss(
                rel_tail_shaking_outputs.view(
                    -1, rel_tail_shaking_outputs.size(-1)),
                rel_tail_shaking_tag.view(-1))

            return ent_loss, rel_head_loss, rel_tail_loss
        return ent_shaking_outputs, rel_head_shaking_outputs, rel_tail_shaking_outputs

    def train_step(self, input_ids, attention_mask, token_type_ids,
                   entity_shaking_tag, rel_head_shaking_tag,
                   rel_tail_shaking_tag):
        ent_loss, rel_head_loss, rel_tail_loss = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            entity_shaking_tag=entity_shaking_tag,
            rel_head_shaking_tag=rel_head_shaking_tag,
            rel_tail_shaking_tag=rel_tail_shaking_tag)

        loss = ent_loss + 0.2 * rel_head_loss + 0.2 * rel_tail_loss
        return {'loss': loss}

    def evaluate_step(self, input_ids, attention_mask, token_type_ids,
                      span2ent, span2rel, wordpiece2token):
        ent_shaking_outputs, rel_head_shaking_outputs, rel_tail_shaking_outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)
        ent_shaking_tag = torch.argmax(ent_shaking_outputs, dim=-1)
        rel_head_shaking_tag = torch.argmax(rel_head_shaking_outputs, dim=-1)
        rel_tail_shaking_tag = torch.argmax(rel_tail_shaking_outputs, dim=-1)
        ent_pred = []
        rel_pred = []
        for idx, tags in enumerate(
                zip(ent_shaking_tag, rel_head_shaking_tag,
                    rel_tail_shaking_tag)):
            entity_wordpiece_results, relation_wordpiece_results = self.valid_tagger.decode_fr_shaking_tag(
                tags[0], tags[1], tags[2])
            sent_ent_pred = []
            sent_rel_pred = []
            for result in relation_wordpiece_results:
                subject_wordpiece = result[0]
                object_wordpiece = result[1]
                label = result[2]
                sent_wordpiece2token = wordpiece2token[idx]
                if not subject_wordpiece[
                        0] in sent_wordpiece2token or not subject_wordpiece[
                            1] in sent_wordpiece2token or not object_wordpiece[
                                0] in sent_wordpiece2token or not object_wordpiece[
                                    1] in sent_wordpiece2token:
                    continue
                subject_token = (sent_wordpiece2token[subject_wordpiece[0]],
                                 sent_wordpiece2token[subject_wordpiece[1]] +
                                 1)
                object_token = (sent_wordpiece2token[object_wordpiece[0]],
                                sent_wordpiece2token[object_wordpiece[1]] + 1)
                sent_rel_pred.append((subject_token, object_token, label))
            rel_pred.append(sent_rel_pred)
            for result in entity_wordpiece_results:
                subject_wordpiece = result[0]
                label = result[1]
                sent_wordpiece2token = wordpiece2token[idx]
                if not subject_wordpiece[
                        0] in sent_wordpiece2token or not subject_wordpiece[
                            1] in sent_wordpiece2token:
                    continue
                subject_token = (sent_wordpiece2token[subject_wordpiece[0]],
                                 sent_wordpiece2token[subject_wordpiece[1]] +
                                 1)
                sent_ent_pred.append((subject_token, label))
            ent_pred.append(sent_ent_pred)
        return {
            'ent_pred': ent_pred,
            'ent_target': span2ent,
            'rel_pred': rel_pred,
            'rel_target': span2rel
        }

    def infer_step(self, tokens, input_ids, attention_mask, token_type_ids,
                   wordpiece2token):
        # TODO finish this function
        ent_shaking_outputs, rel_head_shaking_outputs, rel_tail_shaking_outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)
        ent_shaking_tag = torch.argmax(ent_shaking_outputs, dim=-1)
        rel_head_shaking_tag = torch.argmax(rel_head_shaking_outputs, dim=-1)
        rel_tail_shaking_tag = torch.argmax(rel_tail_shaking_outputs, dim=-1)
        ent_pred = []
        rel_pred = []
        instances = []
        for idx, tags in enumerate(
                zip(ent_shaking_tag, rel_head_shaking_tag,
                    rel_tail_shaking_tag)):
            entity_wordpiece_results, relation_wordpiece_results = self.valid_tagger.decode_fr_shaking_tag(
                tags[0], tags[1], tags[2])
            sent_ent_pred = []
            sent_rel_pred = []
            for result in relation_wordpiece_results:
                subject_wordpiece = result[0]
                object_wordpiece = result[1]
                label = result[2]
                sent_wordpiece2token = wordpiece2token[idx]
                if not subject_wordpiece[
                        0] in sent_wordpiece2token or not subject_wordpiece[
                            1] in sent_wordpiece2token or not object_wordpiece[
                                0] in sent_wordpiece2token or not object_wordpiece[
                                    1] in sent_wordpiece2token:
                    continue
                subject_token = (sent_wordpiece2token[subject_wordpiece[0]],
                                 sent_wordpiece2token[subject_wordpiece[1]] +
                                 1)
                object_token = (sent_wordpiece2token[object_wordpiece[0]],
                                sent_wordpiece2token[object_wordpiece[1]] + 1)
                sent_rel_pred.append((subject_token, object_token, label))
            rel_pred.append(sent_rel_pred)
            for result in entity_wordpiece_results:
                subject_wordpiece = result[0]
                label = result[1]
                sent_wordpiece2token = wordpiece2token[idx]
                if not subject_wordpiece[
                        0] in sent_wordpiece2token or not subject_wordpiece[
                            1] in sent_wordpiece2token:
                    continue
                subject_token = (sent_wordpiece2token[subject_wordpiece[0]],
                                 sent_wordpiece2token[subject_wordpiece[1]] +
                                 1)
                sent_ent_pred.append((subject_token, label))
            ent_pred.append(sent_ent_pred)
            instances.append(
                Instance(tokens=tokens[idx],
                         entity_mentions=ent_pred,
                         relation_mentions=rel_pred))
        return {'pred': instances}
