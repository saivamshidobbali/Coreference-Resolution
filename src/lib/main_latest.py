import sys
import nltk
import re
import pprint
#import Tkinter
import xml.etree.ElementTree as ET
from lib.stat_parser import Parser, display_tree
from nltk.corpus import wordnet as wn

from difflib import SequenceMatcher as editDifference
from nltk.corpus import conll2000
from nltk.chunk import conlltags2tree, tree2conlltags
from nltk.chunk import ne_chunk
from nltk import pos_tag

#anaphor - mams noun phrases
#ner_sentences - all np NER (label*)

def find_coref(anaphor_idx):
    ans = []
    anaphor = ner_sentences[anaphor_idx]
    ner_flags = []
    anaphor_indices = range(0,len(ner_sentences)-1)

    for anaphor_idx  in anaphor_indices:
        coref = ner_sentences[anaphor_idx ]

        if anaphor['label'] is not None and anaphor['label'] == coref['label']:
            if editDifference(None, anaphor['text'], coref['text']).ratio() > 0.65:
                ans.append("ner match "+anaphor['text']+" ===== "+ coref['text'])
                ner_flags.append(anaphor_idx)




    for anaphor_idx  in anaphor_indices:
        coref = ner_sentences[anaphor_idx ]
        if anaphor_idx in ner_flags:
            continue
        # exact string match or just the head nouns match
        if anaphor['text'] == coref['text'] or anaphor['text'].split()[-1] == coref['text'].split()[-1]:
                ans.append("exact string match "+ anaphor['text']+" ===== "+ coref['text'])
                ner_flags.append(anaphor_idx)



        # strings are very similar
        if editDifference(None, anaphor['text'], coref['text']).ratio() > 0.65:
                ans.append("similar string match "+ anaphor['text'] + " ===== "+ coref['text'])
                ner_flags.append(anaphor_idx)




    for anaphor_idx  in anaphor_indices:
        coref = ner_sentences[anaphor_idx ]
        if anaphor_idx in ner_flags:
            continue
        # look for synonyms
        w1 = wn.synsets(anaphor['text'].split()[-1], pos=wn.NOUN)
        w2 = wn.synsets(coref['text'].split()[-1], pos=wn.NOUN)
        if w1 and w2:
            if(w1[0].wup_similarity(w2[0])) > 0.65:
                ans.append("semantic match "+ anaphor['text']+ " ===== "+ coref['text'])
                ner_flags.append(anaphor_idx)


    return ans

def extract_entity_names(t):
    entity_names = []

    if hasattr(t, 'label') and t.label:
        if t.label() != 'S':
            entity_names.append(' '.join([child[0] for child in t]))
        else:
            for child in t:
                entity_names.extend(extract_entity_names(child))

    return entity_names


def extract_entity_labels(t):
    entity_names = []

    if hasattr(t, 'label') and t.label:
        if t.label() != 'S':
            entity_names.append(' '.join(t.label()))
        else:
            for child in t:
                entity_names.extend(extract_entity_labels(child))

    return entity_names


def add_path_to_sys_path():
    sys.path.append('./pyStatParser/')

ner_sentences = []
np_coref_list = []
np_list = []
sentence_parse = []

def main():
    add_path_to_sys_path()
    grammar = "NP: {<DT>?<JJ.*>*<NN.*>+}"
    parser = Parser()

    filename = sys.argv[1]
    with open(filename) as f:
         content = f.readlines()

    for string in content:
        xml_data = ET.fromstring(string)

        for child in xml_data:
            np_coref_list.append(child.text)

    count = 0
    sent_num = []
    for sentence in content:
        count+=1
        xml_data = ET.fromstring(sentence)
        pure_text = ET.tostring(xml_data, encoding='utf8', method='text')
        parse_tree = parser.parse(pure_text)
        sentence_parse.append(parse_tree)

        for node in parse_tree.subtrees(filter=lambda t: t.height() < 4 and (t.label() == 'NP' or t.label() == 'NNP')):
            constituents = [constituent for constituent in node.flatten()]
            np_list.append(" ".join(constituents))
            sent_num.append(count)


        for child in xml_data:
            if child not in np_list:
                 np_list.append(child.text)

    for np in np_list:
        name =  extract_entity_names(nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(np))))
        label  = extract_entity_labels(nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(np))))

        if name:
            namedEntity = label[-1]
        else:
            namedEntity = None

        ner_sentences.append({"text": np,
                               "NE" : name,
                              "label": namedEntity})

    print(np_list)
    print("lengths ", len(np_list), len(ner_sentences))
    for anaphor_idx, anaphor in enumerate(np_list):
        if anaphor in np_coref_list:
            print("===================Anaphor==================", anaphor)
            ans = find_coref(anaphor_idx)
            if ans != None:
                for elem in ans:
                   print(elem)
main()
