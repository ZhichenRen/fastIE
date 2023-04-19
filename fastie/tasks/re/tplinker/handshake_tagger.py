from typing import Dict
from fastNLP import Vocabulary
import torch
from fastie.envs import logger


class HandshakingTaggingScheme(object):
    """将实体与关系三元组转换为TPLinker模型需要的HandshakingTaggingScheme.

    Args:
        object (_type_): _description_
    """

    def __init__(self,
                 tag_vocab: Dict[str, Vocabulary],
                 max_seq_len: int = 100):
        super(HandshakingTaggingScheme, self).__init__()
        assert 'entity' in tag_vocab.keys(), "key 'entity' not in tag_vocab!"
        assert 'relation' in tag_vocab.keys(
        ), "key 'relation' not in tag_vocab!"
        self.rel2id = {
            rel: id - 1
            for rel, id in tag_vocab['relation'].word2idx.items()
            if rel != 'None'
        }
        self.id2rel = {id: rel for rel, id in self.rel2id.items()}

        # self.tag2id_ent = {
        #     "O": 0,
        #     "ENT-H2T": 1, # entity head to entity tail
        # }
        # self.id2tag_ent = {id_:tag for tag, id_ in self.tag2id_ent.items()}
        # Modified
        self.tag2id_ent = tag_vocab['entity'].word2idx
        self.id2tag_ent = tag_vocab['entity'].idx2word

        self.tag2id_head_rel = {
            'O': 0,
            'REL-SH2OH': 1,  # subject head to object head
            'REL-OH2SH': 2,  # object head to subject head
        }
        self.id2tag_head_rel = {
            id_: tag
            for tag, id_ in self.tag2id_head_rel.items()
        }

        self.tag2id_tail_rel = {
            'O': 0,
            'REL-ST2OT': 1,  # subject tail to object tail
            'REL-OT2ST': 2,  # object tail to subject tail
        }
        self.id2tag_tail_rel = {
            id_: tag
            for tag, id_ in self.tag2id_tail_rel.items()
        }

        # mapping shaking sequence and matrix
        self.matrix_size = max_seq_len
        # 拉平后的上三角矩阵，可以根据下标查找矩阵位置
        # e.g. [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        self.shaking_ind2matrix_ind = [
            (ind, end_ind) for ind in range(self.matrix_size)
            for end_ind in list(range(self.matrix_size))[ind:]
        ]
        # 根据矩阵位置查找下标
        self.matrix_ind2shaking_ind = [[0 for i in range(self.matrix_size)]
                                       for j in range(self.matrix_size)]
        for shaking_ind, matrix_ind in enumerate(self.shaking_ind2matrix_ind):
            self.matrix_ind2shaking_ind[matrix_ind[0]][
                matrix_ind[1]] = shaking_ind

    def get_spots(self, instance):
        """entity spot and tail_rel spot: (span_pos1, span_pos2, tag_id)
        head_rel spot: (rel_id, span_pos1, span_pos2, tag_id)"""
        ent_matrix_spots, head_rel_matrix_spots, tail_rel_matrix_spots = [], [], []
        ents = {ent[0]: ent[1] for ent in instance['entity_mentions']}
        for rel in instance['relation_mentions']:
            subj_tok_span = rel[0]
            obj_tok_span = rel[1]
            # ent_matrix_spots.append((subj_tok_span[0], subj_tok_span[1], self.tag2id_ent["ENT-H2T"]))
            # ent_matrix_spots.append((obj_tok_span[0], obj_tok_span[1], self.tag2id_ent["ENT-H2T"]))
            # Modified
            ent_matrix_spots.append((subj_tok_span[0], subj_tok_span[1],
                                     self.tag2id_ent[ents[subj_tok_span]]))
            ent_matrix_spots.append((obj_tok_span[0], obj_tok_span[1],
                                     self.tag2id_ent[ents[obj_tok_span]]))

            if subj_tok_span[0] <= obj_tok_span[0]:
                head_rel_matrix_spots.append(
                    (self.rel2id[rel[2]], subj_tok_span[0], obj_tok_span[0],
                     self.tag2id_head_rel['REL-SH2OH']))
            else:
                head_rel_matrix_spots.append(
                    (self.rel2id[rel[2]], obj_tok_span[0], subj_tok_span[0],
                     self.tag2id_head_rel['REL-OH2SH']))

            if subj_tok_span[1] <= obj_tok_span[1]:
                tail_rel_matrix_spots.append(
                    (self.rel2id[rel[2]], subj_tok_span[1], obj_tok_span[1],
                     self.tag2id_tail_rel['REL-ST2OT']))
            else:
                tail_rel_matrix_spots.append(
                    (self.rel2id[rel[2]], obj_tok_span[1], subj_tok_span[1],
                     self.tag2id_tail_rel['REL-OT2ST']))

        return ent_matrix_spots, head_rel_matrix_spots, tail_rel_matrix_spots

    def sharing_spots2shaking_tag(self, spots):
        '''
        convert spots to shaking seq tag
        spots: [(start_ind, end_ind, tag_id), ], for entiy
        return:
            shake_seq_tag: (shaking_seq_len, )
        '''
        shaking_seq_len = self.matrix_size * (self.matrix_size + 1) // 2
        shaking_seq_tag = torch.zeros(shaking_seq_len).long()
        for sp in spots:
            shaking_ind = self.matrix_ind2shaking_ind[sp[0]][sp[1]]
            shaking_seq_tag[shaking_ind] = sp[2]
        return shaking_seq_tag

    def spots2shaking_tag(self, spots):
        '''
        convert spots to shaking seq tag
        spots: [(rel_id, start_ind, end_ind, tag_id), ], for head relation and tail relation
        return:
            shake_seq_tag: (rel_size, shaking_seq_len, )
        '''
        shaking_seq_len = self.matrix_size * (self.matrix_size + 1) // 2
        shaking_seq_tag = torch.zeros(len(self.rel2id), shaking_seq_len).long()
        for sp in spots:
            shaking_ind = self.matrix_ind2shaking_ind[sp[1]][sp[2]]
            shaking_seq_tag[sp[0]][shaking_ind] = sp[3]
        return shaking_seq_tag

    def sharing_spots2shaking_tag4batch(self, batch_spots):
        '''
        convert spots to batch shaking seq tag
        因长序列的stack是费时操作，所以写这个函数用作生成批量shaking tag
        如果每个样本生成一条shaking tag再stack，一个32的batch耗时1s，太昂贵
        spots: [(start_ind, end_ind, tag_id), ], for entiy
        return:
            batch_shake_seq_tag: (batch_size, shaking_seq_len)
        '''
        shaking_seq_len = self.matrix_size * (self.matrix_size + 1) // 2
        batch_shaking_seq_tag = torch.zeros(len(batch_spots),
                                            shaking_seq_len).long()
        for batch_id, spots in enumerate(batch_spots):
            for sp in spots:
                shaking_ind = self.matrix_ind2shaking_ind[sp[0]][sp[1]]
                tag_id = sp[2]
                batch_shaking_seq_tag[batch_id][shaking_ind] = tag_id
        return batch_shaking_seq_tag

    def spots2shaking_tag4batch(self, batch_spots):
        '''
        convert spots to batch shaking seq tag
        spots: [(rel_id, start_ind, end_ind, tag_id), ], for head relation and tail_relation
        return:
            batch_shake_seq_tag: (batch_size, rel_size, shaking_seq_len)
        '''
        shaking_seq_len = self.matrix_size * (self.matrix_size + 1) // 2
        batch_shaking_seq_tag = torch.zeros(len(batch_spots), len(self.rel2id),
                                            shaking_seq_len).long()
        for batch_id, spots in enumerate(batch_spots):
            for sp in spots:
                shaking_ind = self.matrix_ind2shaking_ind[sp[1]][sp[2]]
                tag_id = sp[3]
                rel_id = sp[0]
                batch_shaking_seq_tag[batch_id][rel_id][shaking_ind] = tag_id
        return batch_shaking_seq_tag

    def get_spots_fr_shaking_tag(self, shaking_tag):
        '''
        shaking_tag -> spots
        shaking_tag: (rel_size, shaking_seq_len)
        spots: [(rel_id, start_ind, end_ind, tag_id), ]
        '''
        spots = []
        shaking_tag_nonzero = shaking_tag.nonzero()
        if len(shaking_tag_nonzero) > 10000:
            logger.warning(
                f'Too much relations head pairs or tail pairs ({len(shaking_tag_nonzero)}) are predicted! Probably because of bad data initialization!'
            )
            shaking_tag_nonzero = shaking_tag_nonzero[:10000, ...]
        for shaking_inds in shaking_tag_nonzero:
            rel_id = shaking_inds[0].item()
            tag_id = shaking_tag[rel_id][shaking_inds[1]].item()
            matrix_inds = self.shaking_ind2matrix_ind[shaking_inds[1]]
            spot = (rel_id, matrix_inds[0], matrix_inds[1], tag_id)
            spots.append(spot)
        return spots

    def get_sharing_spots_fr_shaking_tag(self, shaking_tag):
        '''
        shaking_tag -> spots
        shaking_tag: (shaking_seq_len, )
        spots: [(start_ind, end_ind, tag_id), ]
        '''
        spots = []
        shaking_tag_nonzero = shaking_tag.nonzero()
        if len(shaking_tag_nonzero) > 100:
            logger.warning(
                f'Too much entities ({len(shaking_tag_nonzero)}) are predicted! Probably because of bad data initialization!'
            )
            shaking_tag_nonzero = shaking_tag_nonzero[:100, ...]
        for shaking_ind in shaking_tag_nonzero:
            shaking_ind_ = shaking_ind[0].item()
            tag_id = shaking_tag[shaking_ind_]
            matrix_inds = self.shaking_ind2matrix_ind[shaking_ind_]
            spot = (matrix_inds[0], matrix_inds[1], tag_id)
            spots.append(spot)
        return spots

    def decode_fr_shaking_tag(self, ent_shaking_tag, head_rel_shaking_tag,
                              tail_rel_shaking_tag):
        '''
        ent shaking tag: (shaking_seq_len, )
        head rel and tail rel shaking_tag: size = (rel_size, shaking_seq_len, )
        '''
        ent_list = []
        rel_list = []

        ent_matrix_spots = self.get_sharing_spots_fr_shaking_tag(
            ent_shaking_tag)
        head_rel_matrix_spots = self.get_spots_fr_shaking_tag(
            head_rel_shaking_tag)
        tail_rel_matrix_spots = self.get_spots_fr_shaking_tag(
            tail_rel_shaking_tag)

        # entity
        head_ind2entities = {}
        for sp in ent_matrix_spots:
            tag_id = sp[2].item()
            # Modified
            # if tag_id != self.tag2id_ent["ENT-H2T"]:
            #     continue
            if tag_id == self.tag2id_ent['None']:
                continue

            head_key = sp[
                0]  # take head as the key to entity list start with the head token
            if head_key not in head_ind2entities:
                head_ind2entities[head_key] = []
            # Modified
            ent = ((sp[0], sp[1]), self.id2tag_ent[tag_id])
            head_ind2entities[head_key].append(ent)
            ent_list.append(ent)

        # tail relation
        tail_rel_memory_set = set()
        for sp in tail_rel_matrix_spots:
            rel_id = sp[0]
            tag_id = sp[3]
            if tag_id == self.tag2id_tail_rel['REL-ST2OT']:
                tail_rel_memory = '{}-{}-{}'.format(rel_id, sp[1], sp[2])
                tail_rel_memory_set.add(tail_rel_memory)
            elif tag_id == self.tag2id_tail_rel['REL-OT2ST']:
                tail_rel_memory = '{}-{}-{}'.format(rel_id, sp[2], sp[1])
                tail_rel_memory_set.add(tail_rel_memory)

        # head relation
        for sp in head_rel_matrix_spots:
            rel_id = sp[0]
            tag_id = sp[3]

            if tag_id == self.tag2id_head_rel['REL-SH2OH']:
                subj_head_key, obj_head_key = sp[1], sp[2]
            elif tag_id == self.tag2id_head_rel['REL-OH2SH']:
                subj_head_key, obj_head_key = sp[2], sp[1]
            else:
                continue

            if subj_head_key not in head_ind2entities or obj_head_key not in head_ind2entities:
                # no entity start with subj_head_key and obj_head_key
                continue
            subj_list = head_ind2entities[
                subj_head_key]  # all entities start with this subject head
            obj_list = head_ind2entities[
                obj_head_key]  # all entities start with this object head

            # go over all subj-obj pair to check whether the relation exists
            for subj in subj_list:
                for obj in obj_list:
                    tail_rel_memory = '{}-{}-{}'.format(
                        rel_id, subj[0][1], obj[0][1])
                    if tail_rel_memory not in tail_rel_memory_set:
                        # no such relation
                        continue

                    rel_list.append((subj[0], obj[0], self.id2rel[rel_id]))
        return ent_list, rel_list
