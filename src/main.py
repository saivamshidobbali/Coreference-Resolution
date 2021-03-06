import sys
import nltk
import re
import pprint
import xml.etree.ElementTree as ET
from lib.stat_parser import Parser, display_tree
from nltk.corpus import wordnet as wn
import copy
from nltk import Tree
import os

from difflib import SequenceMatcher as editDifference
from nltk.corpus import conll2000
from nltk.chunk import conlltags2tree, tree2conlltags
from nltk.chunk import ne_chunk
from nltk import pos_tag

RESPONSE_DIR = "./responses"

def find_coref(anaphor, named_entity_resolution_list, sent_num, np_list_copy):
    ans = []
    anaphor_indices = range(0,len(named_entity_resolution_list))
    
    for idx  in anaphor_indices:
        coref = copy.deepcopy(named_entity_resolution_list[idx])    
                
        if anaphor['label'] is not None and anaphor['label'] == coref['label']:
            if editDifference(None, anaphor['text'], coref['text']).ratio() > 1.25:
                ans.append(str(sent_num[idx])+" "+np_list_copy[idx])

                named_entity_resolution_list[idx]['label'] = "bullshit"
                named_entity_resolution_list[idx]['text'] = "bullshit"
                continue
                
                
        if anaphor['text'] == coref['text'] or anaphor['text'].split()[-1] == coref['text'].split()[-1]:

                ans.append(str(sent_num[idx])+" "+np_list_copy[idx])
                named_entity_resolution_list[idx]['label'] = "bullshit"
                named_entity_resolution_list[idx]['text'] = "bullshit"
                continue


        elif editDifference(None, anaphor['text'], coref['text']).ratio() > 1.25:

                ans.append(str(sent_num[idx])+" "+np_list_copy[idx])

                named_entity_resolution_list[idx]['label'] = "bullshit"
                named_entity_resolution_list[idx]['text'] = "bullshit"
                continue


        w1 = wn.synsets(anaphor['text'].split()[-1], pos=wn.NOUN)
        w2 = wn.synsets(coref['text'].split()[-1], pos=wn.NOUN)
        if w1 and w2:
                 if(w1[0].wup_similarity(w2[0])) > 0.95:
                    ans.append(str(sent_num[idx])+" "+np_list_copy[idx])

                    named_entity_resolution_list[idx]['label'] = "bullshit"
                    named_entity_resolution_list[idx]['text'] = "bullshit"
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



def main():
     global RESPONSE_DIR

     if len(sys.argv) == 3:
       RESPONSE_DIR = sys.argv[2]
       filelist = [ f for f in os.listdir(sys.argv[2])]
     else:
       filelist = [ f for f in os.listdir(RESPONSE_DIR)]
       
     for f in filelist:
         os.remove(os.path.join(RESPONSE_DIR, f))
   

     print(RESPONSE_DIR) 
     f = open(sys.argv[1],"r")
     filenames = f.readlines()

     for file in filenames:
      try:
        process(file.rstrip('\n'))
      except:
        continue
    
    
def process(filename):    
    grammar = "NP: {<DT>?<JJ.*>*<NN.*>+}"
    np_list = []
   
    np_coref_list = []
    ner_sentences = []
    parser = Parser()

    with open(filename) as f:
         content = f.readlines()

    coref_id_list = []
    for string in content:
     try:
        xml_data = ET.fromstring(string)

        for child in xml_data:
            coref_id_list.append(child.attrib['ID'])
            np_coref_list.append(child.text)
     except:
      continue

    sent_num = []
    parse_tree=""
    
    for sentence in content:
     try:
        xml_data = ET.fromstring(sentence)
        pure_text = ET.tostring(xml_data, encoding='utf8', method='text')
        wordlist = nltk.word_tokenize(pure_text)
        pos_tagged  =  nltk.pos_tag(wordlist)


        #grammar = r"""
        #  NP: {<DT|PP\$>?<JJ>*<NN>}   # chunk determiner/possessive, adjectives and noun
        #      {<NNP>+}                # chunk sequences of proper nouns
        #"""
        
        cp = nltk.RegexpParser(grammar)
        parse_tree = cp.parse(pos_tagged)
       

        before_length = len(np_list)
        for node in parse_tree.subtrees(filter=lambda t: t.height() < 4 and (t.label() == 'NP')):
         try:
            x = ""
            for i,elem in enumerate(node):
                 if i == 0:
                    x = elem[0]
                 else:
                    x = x +" "+elem[0]
            np_list.append(x)
            sent_num.append(xml_data.attrib['ID'])
         except:
           continue
       
        for child in xml_data:
         try:
            if child.text in np_list:
                 index = np_list.index(child.text)
                 del np_list[index]
                 del sent_num[-1]
         except:
           continue
     except:
           continue
    
    for np in np_list:
      try:
        name =  extract_entity_names(nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(np))))
        label  = extract_entity_labels(nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(np))))

        if name:
            namedEntity = label[-1]
        else:
            namedEntity = None

        ner_sentences.append({"text": np.lower(),
                               "NE" : [x.lower() for x in name],
                              "label": namedEntity})
      except:
        continue
        
    ner_coref_sentences = []
    for np in np_coref_list:
     try:
        name =  extract_entity_names(nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(np))))
        label  = extract_entity_labels(nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(np))))

        if name:
            namedEntity = label[-1]
        else:
            namedEntity = None

        ner_coref_sentences.append({"text": np.lower(),
                               "NE" : [x.lower() for x in name],
                              "label": namedEntity})
     except:
      continue
    
    np_list_copy = list(np_list)
    np_coref_list_copy = list(np_coref_list)
    
    np_list = [x.lower() for x in np_list]
    np_coref_list = [x.lower() for x in np_coref_list]
   
    filename = os.path.basename(filename)
    print(filename)
    filename = RESPONSE_DIR+"/"+ filename.split('.')[0]+".response"
    print(filename)
    file = open(filename,"a+")
    
    for anaphor_idx, anaphor in enumerate(np_coref_list):
     try:
            file.writelines("<COREF ID=\""+ coref_id_list[anaphor_idx]+"\">"+ np_coref_list_copy[anaphor_idx]+ "</COREF>")
            file.writelines("\n")
                       
            temp = []
            for elem in ner_sentences:
                temp.append(copy.deepcopy(elem))
            
            
            for elem in ner_coref_sentences:
                 if elem['text'] == anaphor:
                       anaphor_ner = elem
                       break             
                
            ans = find_coref(anaphor_ner, temp, sent_num, np_list_copy)
            if ans != None:
                for elem in ans:
                    elem = elem.split()
                    print_data ="{"+elem[0]+"} "+"{"+elem[-1]+"}"
                    file.writelines(print_data)
                    file.writelines("\n")
              
            file.writelines("\n")
     except:
         continue
main()
