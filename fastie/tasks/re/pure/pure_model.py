import torch
from torch import nn
from fastNLP.transformers.torch import BertModel
from torch.nn import LayerNorm, CrossEntropyLoss
from fastie.utils.nn_utils import batched_index_select
from fastNLP import Vocabulary, Instance
from typing import Dict
from fastNLP.transformers.torch import BertTokenizer


class BertForEntity(nn.Module):

    def __init__(self,
                 tag_vocab: Dict[str, Vocabulary],
                 bert_model_name: str = 'bert-base-uncased',
                 hidden_size: int = 768,
                 mlp_hidden_size: int = 150,
                 width_embedding_dim: int = 150,
                 max_span_length: int = 10,
                 bert_dropout: float = 0.1,
                 dropout: float = 0.2):
        super().__init__()
        assert 'entity' in tag_vocab.keys(
        ), "The key entity doesn't exist in tag_vocab!"
        num_ner_labels = len(tag_vocab['entity'].word2idx)
        self.tag_vocab = tag_vocab
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(
            bert_model_name)
        self.bert_dropout = nn.Dropout(bert_dropout)
        self.width_embedding = nn.Embedding(max_span_length + 1,
                                            width_embedding_dim)

        self.ner_classifier = nn.Sequential(
            nn.Linear(hidden_size * 2 + width_embedding_dim, mlp_hidden_size),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, num_ner_labels))

    def _get_span_embeddings(self,
                             input_ids,
                             spans,
                             token_type_ids=None,
                             attention_mask=None):
        output = self.bert(input_ids=input_ids,
                           token_type_ids=token_type_ids,
                           attention_mask=attention_mask)

        sequence_output = self.bert_dropout(output[0])
        """
        spans: [batch_size, num_spans, 3]; 0: left_ned, 1: right_end, 2: width
        spans_mask: (batch_size, num_spans, )
        """
        spans_start = spans[:, :, 0].view(spans.size(0), -1)
        spans_start_embedding = batched_index_select(sequence_output,
                                                     spans_start)
        spans_end = spans[:, :, 1].view(spans.size(0), -1)
        spans_end_embedding = batched_index_select(sequence_output, spans_end)

        spans_width = spans[:, :, 2].view(spans.size(0), -1)
        spans_width_embedding = self.width_embedding(spans_width)

        # Concatenate embeddings of left/right points and the width embedding
        spans_embedding = torch.cat(
            (spans_start_embedding, spans_end_embedding,
             spans_width_embedding),
            dim=-1)
        """
        spans_embedding: (batch_size, num_spans, hidden_size*2+embedding_dim)
        """
        return spans_embedding

    def forward(self,
                input_ids,
                spans,
                spans_mask,
                spans_ner_label=None,
                token_type_ids=None,
                attention_mask=None):
        spans_embedding = self._get_span_embeddings(
            input_ids,
            spans,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask)
        ffnn_hidden = []
        hidden = spans_embedding
        for layer in self.ner_classifier:
            hidden = layer(hidden)
            ffnn_hidden.append(hidden)
        logits = ffnn_hidden[-1]

        if spans_ner_label is not None:
            loss_fct = CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = spans_mask.view(-1) == 1
                active_logits = logits.view(-1, logits.shape[-1])
                active_labels = torch.where(
                    active_loss, spans_ner_label.view(-1),
                    torch.tensor(
                        loss_fct.ignore_index).type_as(spans_ner_label))
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, logits.shape[-1]),
                                spans_ner_label.view(-1))
            return loss, logits, spans_embedding
        else:
            return logits, spans_embedding, spans_embedding

    def decode(self, spans, spans_mask, logits, origin_spans):
        ent_pred = []
        for idx, batch_logits in enumerate(logits):
            batch_logits = batch_logits[spans_mask[idx] == 1]
            batch_logits = torch.softmax(batch_logits, dim=-1)
            batch_labels = torch.argmax(batch_logits, dim=-1)

            ents = []

            batch_spans = spans[idx][spans_mask[idx] == 1]
            none_id = self.tag_vocab['entity'].to_index('None')
            for span_id, span in enumerate(batch_spans):
                if batch_labels[span_id].item() != none_id:
                    start = origin_spans[idx][span_id][0]
                    end = origin_spans[idx][span_id][1]
                    ents.append(
                        ((start, end + 1), self.tag_vocab['entity'].to_word(
                            batch_labels[span_id].item())))

            ent_pred.append(ents)
        return ent_pred

    def train_step(self, input_ids, spans, spans_mask, labels, attention_mask):
        loss, logits, spans_embedding = self.forward(
            input_ids=input_ids,
            spans=spans,
            spans_mask=spans_mask,
            spans_ner_label=labels,
            attention_mask=attention_mask)
        return {'loss': loss}

    def evaluate_step(self, input_ids, spans, spans_mask, attention_mask,
                      span2ent, origin_spans):
        logits, _, _ = self.forward(input_ids=input_ids,
                                    spans=spans,
                                    spans_mask=spans_mask,
                                    attention_mask=attention_mask)

        ent_pred = self.decode(spans=spans,
                               spans_mask=spans_mask,
                               logits=logits,
                               origin_spans=origin_spans)

        result = {}
        result['ent_target'] = span2ent
        result['ent_pred'] = ent_pred

        return result

    def infer_step(self,
                   tokens,
                   sent_id,
                   doc_key,
                   input_ids,
                   spans,
                   spans_mask,
                   attention_mask,
                   origin_spans,
                   span2ent=None,
                   span2rel=None):
        logits, _, _ = self.forward(input_ids=input_ids,
                                    spans=spans,
                                    spans_mask=spans_mask,
                                    attention_mask=attention_mask)
        ent_idx_pred = self.decode(spans=spans,
                                   spans_mask=spans_mask,
                                   logits=logits,
                                   origin_spans=origin_spans)
        # import pdb
        # breakpoint()
        ent_pred = []
        instances = []
        for id, ents in enumerate(ent_idx_pred):
            sent_ent_pred = []
            for ent in ents:
                ent_text = self.tokenizer.decode(
                    self.tokenizer.encode(tokens[id][ent[0][0]:ent[0][1]],
                                          add_special_tokens=False))
                sent_ent_pred.append((ent_text, ent[1]))
            ent_pred.append(sent_ent_pred)
            # TODO add golden entity label here
            entity_mentions = span2ent[id] if span2ent else []
            relation_mentions = span2rel[id] if span2rel else []
            instance = Instance(tokens=tokens[id],
                                doc_key=doc_key[id],
                                entity_mentions=entity_mentions,
                                relation_mentions=relation_mentions,
                                sent_id=sent_id[id],
                                ent_pred=ent_idx_pred[id])
            instances.append(instance)
        result = {}
        result['pred'] = instances
        return result


class BertForRelation(nn.Module):

    def __init__(self,
                 tag_vocab: Dict[str, Vocabulary],
                 tokenizer: BertTokenizer,
                 bert_model_name='bert-base-uncased',
                 hidden_size=768,
                 bert_dropout=0.1):
        super(BertForRelation, self).__init__()
        assert 'entity' in tag_vocab.keys() and 'relation' in tag_vocab.keys(
        ), "Key 'entity' or 'relation' not in tag_vocab!"
        self.tag_vocab = tag_vocab
        self.num_labels = len(tag_vocab['relation'].word2idx)
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert.resize_token_embeddings(len(tokenizer))
        self.dropout = nn.Dropout(bert_dropout)
        self.layer_norm = LayerNorm(hidden_size * 2)
        self.classifier = nn.Linear(hidden_size * 2, self.num_labels)

    def forward(self,
                input_ids,
                attention_mask=None,
                labels=None,
                subject_index=None,
                object_index=None,
                position_ids=None,
                span_pair_mask=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=False,
                            output_attentions=False,
                            position_ids=position_ids)
        sequence_output = outputs[0]
        sub_output = batched_index_select(sequence_output, subject_index[...,
                                                                         0])
        obj_output = batched_index_select(sequence_output, object_index[...,
                                                                        0])
        rep = torch.cat((sub_output, obj_output), dim=-1)
        rep = self.layer_norm(rep)
        rep = self.dropout(rep)
        logits = self.classifier(rep)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            if span_pair_mask is not None:
                active_loss = (span_pair_mask.view(-1) == 1)
                active_logits = logits.view(-1, logits.shape[-1])
                active_labels = torch.where(
                    active_loss, labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels))
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels),
                                labels.view(-1))
            return loss
        else:
            return logits

    def decode_rel(self,
                   logits,
                   origin_subject_index,
                   origin_object_index,
                   span_pair_mask=None):
        rel_pred = []
        for batch_idx, batch_logits in enumerate(logits):
            if span_pair_mask is not None:
                batch_logits = batch_logits[span_pair_mask[batch_idx]]
            else:
                assert batch_logits.dim() == 1
                batch_logits = torch.unsqueeze(batch_logits, 0)
            relations = torch.argmax(batch_logits, dim=-1)

            batch_pred = []
            for spans_idx, relation in enumerate(relations):
                if relation.item() != 0:
                    rel_type = self.tag_vocab['relation'].to_word(
                        relation.item())
                    batch_pred.append(
                        (origin_subject_index[batch_idx][spans_idx],
                         origin_object_index[batch_idx][spans_idx], rel_type))
            rel_pred.append(batch_pred)
        return rel_pred

    def train_step(self,
                   input_ids,
                   labels,
                   subject_index,
                   object_index,
                   attention_mask=None,
                   position_ids=None,
                   span_pair_mask=None):
        loss = self.forward(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            subject_index=subject_index,
                            object_index=object_index,
                            position_ids=position_ids,
                            span_pair_mask=span_pair_mask)
        return {'loss': loss}

    def evaluate_step(self,
                      input_ids,
                      subject_index,
                      object_index,
                      origin_subject_index,
                      origin_object_index,
                      span2ent,
                      span2rel,
                      ent_pred,
                      position_ids=None,
                      attention_mask=None,
                      span_pair_mask=None):
        result = {
            'ent_pred': ent_pred,
            'ent_target': span2ent,
            'rel_target': span2rel
        }
        if not torch.is_tensor(subject_index) or not torch.is_tensor(
                object_index):
            result['rel_pred'] = []
            return result
        logits = self.forward(input_ids=input_ids,
                              attention_mask=attention_mask,
                              subject_index=subject_index,
                              object_index=object_index,
                              position_ids=position_ids)
        rel_pred = self.decode_rel(logits, origin_subject_index,
                                   origin_object_index, span_pair_mask)
        result['rel_pred'] = rel_pred
        return result

    def infer_step(self,
                   tokens,
                   input_ids,
                   subject_index,
                   object_index,
                   origin_subject_index,
                   origin_object_index,
                   ent_pred,
                   position_ids=None,
                   attention_mask=None,
                   span_pair_mask=None):
        if not torch.is_tensor(subject_index) or not torch.is_tensor(
                object_index):
            instances = []
            for id, token in enumerate(tokens):
                instance = Instance(tokens=token,
                                    entity_mentions=ent_pred[id],
                                    relation_mentions=[])
                instances.append(instance)
            result = {}
            result['pred'] = instances
            return result
        logits = self.forward(input_ids=input_ids,
                              attention_mask=attention_mask,
                              subject_index=subject_index,
                              object_index=object_index,
                              position_ids=position_ids)
        rel_idx_pred = self.decode_rel(logits, origin_subject_index,
                                       origin_object_index, span_pair_mask)
        instances = []
        for id, rels in enumerate(rel_idx_pred):
            instance = Instance(tokens=tokens[id],
                                entity_mentions=ent_pred[id],
                                relation_mentions=rel_idx_pred[id])
            instances.append(instance)
        result = {}
        result['pred'] = instances
        # result['instances'] = instances
        return result
