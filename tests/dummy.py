from copy import deepcopy

from fastNLP.io import DataBundle

from fastie.dataset import build_dataset


def dummy_ner_dataset() -> DataBundle:
    data_bundle = build_dataset([{
        'tokens': [
            'EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British',
            'lamb', '.'
        ],
        'entity_mentions': [[[0], 'ORG'], [[2], 'MISC'], [[6], 'MISC']]
    }, {
        'tokens': ['Peter', 'Blackburn'],
        'entity_mentions': [[[0, 1], 'PER']]
    }, {
        'tokens': ['BRUSSELS', '1996-08-22'],
        'entity_mentions': [[[0], 'LOC']]
    }, {
        'tokens': [
            'The', 'European', 'Commission', 'said', 'on', 'Thursday', 'it',
            'disagreed', 'with', 'German', 'advice', 'to', 'consumers', 'to',
            'shun', 'British', 'lamb', 'until', 'scientists', 'determine',
            'whether', 'mad', 'cow', 'disease', 'can', 'be', 'transmitted',
            'to', 'sheep', '.'
        ],
        'entity_mentions': [[[1, 2], 'ORG'], [[9], 'MISC'], [[15], 'MISC']]
    }, {
        'tokens': [
            'Germany', "\'s", 'representative', 'to', 'the', 'European',
            'Union', "'s", 'veterinary', 'committee', 'Werner', 'Zwingmann',
            'said', 'on', 'Wednesday', 'consumers', 'should', 'buy',
            'sheepmeat', 'from', 'countries', 'other', 'than', 'Britain',
            'until', 'the', 'scientific', 'advice', 'was', 'clearer', '.'
        ],
        'entity_mentions': [[[0], 'LOC'], [[5, 6], 'ORG'], [[10, 11], 'PER'],
                            [[23], 'LOC']]
    }, {
        'tokens': [
            'Rabinovich', 'is', 'winding', 'up', 'his', 'term', 'as',
            'ambassador', '.'
        ],
        'entity_mentions': [[[0], 'PER']]
    }, {
        'tokens': [
            'He', 'will', 'be', 'replaced', 'by', 'Eliahu', 'Ben-Elissar', ',',
            'a', 'former', 'Israeli', 'envoy', 'to', 'Egypt', 'and',
            'right-wing', 'Likud', 'party', 'politician', '.'
        ],
        'entity_mentions': [[[5, 6], 'PER'], [[10], 'MISC'], [[13], 'LOC'],
                            [[16], 'ORG']]
    }, {
        'tokens': [
            'Israel', 'on', 'Wednesday', 'sent', 'Syria', 'a', 'message', ',',
            'via', 'Washington', ',', 'saying', 'it', 'was', 'committed', 'to',
            'peace', 'and', 'wanted', 'to', 'open', 'negotiations', 'without',
            'preconditions', '.'
        ],
        'entity_mentions': [[[0], 'LOC'], [[4], 'LOC'], [[9], 'LOC']]
    }])
    data_bundle.set_dataset(deepcopy(data_bundle.get_dataset('train')), 'dev')
    data_bundle.set_dataset(deepcopy(data_bundle.get_dataset('train')), 'test')
    data_bundle.set_dataset(deepcopy(data_bundle.get_dataset('train')),
                            'infer')
    data_bundle.get_dataset('infer').delete_field('entity_mentions')
    return data_bundle


def dummy_re_dataset() -> DataBundle:
    data_bundle = build_dataset([{
        'sent_id':
        5,
        'tokens': [
            'Retired', 'General', 'Electric', 'Co', '.', 'Chairman', 'Jack',
            'Welch', 'is', 'seeking', 'work-related', 'documents', 'of', 'his',
            'estranged', 'wife', 'in', 'his', 'high-stakes', 'divorce', 'case',
            '.'
        ],
        'entity_mentions': [[[6, 8], 'PER'], [[5, 6], 'PER'], [[13, 14],
                                                               'PER'],
                            [[17, 18], 'PER'], [[1, 4], 'ORG'],
                            [[15, 16], 'PER']],
        'relation_mentions': [[[5, 6], [1, 4], 'ORG-AFF'],
                              [[13, 14], [15, 16], 'PER-SOC'],
                              [[15, 16], [13, 14], 'PER-SOC']],
        'doc_key':
        'APW_ENG_20030325.0786'
    }, {
        'sent_id':
        6,
        'tokens': [
            'The', 'Welches', 'disclosed', 'their', 'plans', 'to', 'divorce',
            'a', 'year', 'ago', ',', 'shortly', 'after', 'Suzy', 'Wetlaufer',
            '_', 'then', 'editor', 'of', 'the', 'Harvard', 'Business',
            'Review', '_', 'revealed', 'she', 'had', 'become', 'romantically',
            'involved', 'with', 'Welch', 'while', 'working', 'on', 'a',
            'story', 'about', 'him', '.'
        ],
        'entity_mentions': [[[31, 32], 'PER'], [[38, 39], 'PER'],
                            [[1, 2], 'PER'], [[3, 4], 'PER'], [[13, 15],
                                                               'PER'],
                            [[17, 18], 'PER'], [[25, 26], 'PER'],
                            [[20, 23], 'ORG']],
        'relation_mentions': [[[17, 18], [20, 23], 'ORG-AFF'],
                              [[13, 15], [31, 32], 'PER-SOC'],
                              [[31, 32], [13, 15], 'PER-SOC']],
        'doc_key':
        'APW_ENG_20030325.0786'
    }, {
        'sent_id':
        7,
        'tokens': [
            'In', 'court', 'papers', 'filed', 'this', 'week', 'in', 'state',
            'Supreme', 'Court', 'in', 'New', 'York', ',', 'Welch', 'requested',
            'a', 'deposition', 'next', 'month', 'of', 'David', 'Heleniak', ',',
            'a', 'senior', 'partner', 'in', 'the', 'law', 'firm', 'of',
            'Shearman', '&amp;', 'Sterling', 'in', 'New', 'York', '.'
        ],
        'entity_mentions': [[[9, 10], 'ORG'], [[11, 13], 'GPE'], [[7, 8],
                                                                  'GPE'],
                            [[36, 38], 'GPE'], [[21, 23], 'PER'],
                            [[26, 27], 'PER'], [[30, 31], 'ORG'],
                            [[32, 35], 'ORG'], [[14, 15], 'PER']],
        'relation_mentions': [[[9, 10], [7, 8], 'PART-WHOLE'],
                              [[9, 10], [11, 13], 'GEN-AFF'],
                              [[26, 27], [30, 31], 'ORG-AFF'],
                              [[30, 31], [36, 38], 'GEN-AFF']],
        'doc_key':
        'APW_ENG_20030325.0786'
    }])
    data_bundle.set_dataset(deepcopy(data_bundle.get_dataset('train')), 'dev')
    data_bundle.set_dataset(deepcopy(data_bundle.get_dataset('train')), 'test')
    data_bundle.set_dataset(deepcopy(data_bundle.get_dataset('train')),
                            'infer')
    data_bundle.get_dataset('infer').delete_field('entity_mentions')
    data_bundle.get_dataset('infer').delete_field('relation_mentions')
    return data_bundle
