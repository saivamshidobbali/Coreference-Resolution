import sys
import nltk
import re
import xml.etree.ElementTree as ET
from nltk.corpus import wordnet
import copy
from nltk import Tree
import os
from difflib import SequenceMatcher
from nltk.chunk import ne_chunk
from nltk import pos_tag
from hobbs import *

from nltk.tree import Tree

from lib.stat_parser.parser import Parser as CkyStatParser                                                                                                                                                  


nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('names')

RESPONSE_DIR = "./responses"


def find_coref(anaphor, named_entity_resolution_list, sent_num, np_list_copy):
    ans = []
    anaphor_indices = range(0,len(named_entity_resolution_list))

    for idx  in anaphor_indices:
        coref = copy.deepcopy(named_entity_resolution_list[idx])    
                
        if anaphor['label'] is not None and anaphor['label'] == coref['label']:
            if SequenceMatcher(None, anaphor['text'], coref['text']).ratio() > 0.99:
                ans.append(str(sent_num[idx])+" "+np_list_copy[idx])

                named_entity_resolution_list[idx]['label'] = "None"
                named_entity_resolution_list[idx]['text'] = "None"
                continue
                
                
        if anaphor['text'] == coref['text'] or anaphor['text'].split()[-1] == coref['text'].split()[-1]:

                ans.append(str(sent_num[idx])+" "+np_list_copy[idx])
                named_entity_resolution_list[idx]['label'] = "None"
                named_entity_resolution_list[idx]['text'] = "None"
                continue


        elif SequenceMatcher(None, anaphor['text'], coref['text']).ratio() > 0.99:

                ans.append(str(sent_num[idx])+" "+np_list_copy[idx])

                named_entity_resolution_list[idx]['label'] = "None"
                named_entity_resolution_list[idx]['text'] = "None"
                continue


        w1 = wordnet.synsets(anaphor['text'].split()[-1], pos=wordnet.NOUN)
        w2 = wordnet.synsets(coref['text'].split()[-1], pos=wordnet.NOUN)
        if w1 and w2:
                 if(w1[0].wup_similarity(w2[0])) > 0.95:
                    ans.append(str(sent_num[idx])+" "+np_list_copy[idx])

                    named_entity_resolution_list[idx]['label'] = "None"
                    named_entity_resolution_list[idx]['text'] = "None"
    return ans

def get_ner_names(t):
    ner_names = []
    if hasattr(t,'label') and t.label:
       if t.label() == 'S':
            for st in t:
                new_list=[]
                new_list = get_ner_names(st)	
                len_st = len(new_list)
                for index in range(0,len_st):
                    ner_names.append(new_list[index])	
       else:
            ner_names.append(' '.join([st[0] for st in t]))

    return ner_names


def get_ner_labels(t):
    ner_labels = []

    if hasattr(t,'label')  and  t.label :
        if t.label() == 'S':
            for st in t:
                new_list=[]
                new_list = get_ner_labels(st)	
                len_st = len(new_list)
                for index in range(0,len_st):
                    ner_labels.append(new_list[index])	
        else:
           ner_labels.append(' '.join(t.label()))


    return ner_labels



def main():
     global RESPONSE_DIR

     if len(sys.argv) == 3:
       RESPONSE_DIR = sys.argv[2]
       filelist = [ f for f in os.listdir(sys.argv[2])]
     else:
       filelist = [ f for f in os.listdir(RESPONSE_DIR)]
       
     for f in filelist:
         os.remove(os.path.join(RESPONSE_DIR, f))
   


     f = open(sys.argv[1],"r")
     filenames = f.readlines()

     for file in filenames:
        process(file.rstrip('\n'))

def hobbs_resolution(pronouns_list, sentence_parse_dict):
    for index,pronoun in enumerate(pronouns_list):
           hobbs_main(pronoun, sentence_parse_dict[index])
        
   
def pronouns(content):

    cky_stat_parser = CkyStatParser() 
    pronouns_list = []
    sent_num_for_pronouns = []

    temp_sentence_parse_list = []
    sentence_parse_dict = {}
    pronoun_count = -1

    for sentence in content:
        
        xml_data = ET.fromstring(sentence)
        pure_text = ET.tostring(xml_data, encoding='utf8', method='text')

        parse_tree = cky_stat_parser.nltk_parse(pure_text) 
        wordlist = nltk.word_tokenize(pure_text)
        pos_tagged  =  nltk.pos_tag(wordlist)

        #cp = nltk.RegexpParser(grammar)
        #parse_tree = cp.parse(pos_tagged)

        temp_sentence_parse_list.append(parse_tree)

        for (a,b) in pos_tagged:
              if b == 'PRP':

                  pronoun_count += 1
                  pronouns_list.append(a)
                  
                  sentence_parse_dict[pronoun_count] = temp_sentence_parse_list
                  sent_num_for_pronouns.append(xml_data.attrib['ID'])

    return (sentence_parse_dict, pronouns_list, sent_num_for_pronouns)
    
def process(filename):    
    grammar = "NP: {<DT>?<JJ.*>*<NN.*>+}"
    np_list = []
   
    np_coref_list = []
    ner_sentences = []


    with open(filename) as f:
         content = f.readlines()

    coref_id_list = []
    for string in content:
        xml_data = ET.fromstring(string)

        for child in xml_data:
            coref_id_list.append(child.attrib['ID'])
            np_coref_list.append(child.text)

    sent_num = []
    parse_tree=""
    sent_num_for_pronouns = []
    sentence_parse_dict = []

    sentence_parse_dict, pronouns_list, sent_num_for_pronouns = pronouns(content)

    hobbs_resolution(pronouns_list, sentence_parse_dict)
    #print(sentence_parse_dict)
  
    for sentence in content:
        xml_data = ET.fromstring(sentence)
        pure_text = ET.tostring(xml_data, encoding='utf8', method='text')
        wordlist = nltk.word_tokenize(pure_text)
        pos_tagged  =  nltk.pos_tag(wordlist)
        cp = nltk.RegexpParser(grammar)
        parse_tree = cp.parse(pos_tagged)
        print("#############################################=================") 
        for node in parse_tree.subtrees(filter=lambda t: t.height() < 4 and (t.label() == 'NP')):
            x = ""
            for i,elem in enumerate(node):
                 if i == 0:
                    x = elem[0]
                 else:
                    x = x +" "+elem[0]
            np_list.append(x)
            sent_num.append(xml_data.attrib['ID'])
       
        for child in xml_data:
            if child.text in np_list:
                 index = np_list.index(child.text)
                 del np_list[index]
                 del sent_num[-1]
    
    for np in np_list:
        name =  get_ner_names(nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(np))))
        label  = get_ner_labels(nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(np))))

        if name:
            namedEntity = label[-1]
        else:
            namedEntity = None

        ner_sentences.append({"text": np.lower(),
                               "NE" : [x.lower() for x in name],
                              "label": namedEntity})
        
    ner_coref_sentences = []
    for np in np_coref_list:
        name =  get_ner_names(nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(np))))
        label  = get_ner_labels(nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(np))))

        if name:
            namedEntity = label[-1]
        else:
            namedEntity = None

        ner_coref_sentences.append({"text": np.lower(),
                               "NE" : [x.lower() for x in name],
                              "label": namedEntity})
    
    np_list_copy = list(np_list)
    np_coref_list_copy = list(np_coref_list)
    np_coref_list = [x.lower() for x in np_coref_list]
    np_list = [x.lower() for x in np_list]
    
    filename = os.path.basename(filename)
    filename = RESPONSE_DIR+"/"+ filename.split('.')[0]+".response"
    file = open(filename,"a+")

    #hobbs_resolution(pronouns_list, parse_tree_list)
    
    for anaphor_idx, anaphor in enumerate(np_coref_list):
            file.writelines("<COREF ID=\""+ coref_id_list[anaphor_idx]+"\">"+ np_coref_list_copy[anaphor_idx]+ "</COREF>")
            file.writelines("\n")
                       
            for elem in ner_coref_sentences:
                 if elem['text'] == anaphor:
                       anaphor_ner = elem
                       break             
                
            ans = find_coref(anaphor_ner, ner_sentences, sent_num, np_list_copy)
            if ans != None:
                for elem in ans:
                    elem = elem.split()
                    print_data ="{"+elem[0]+"} "+"{"+elem[-1]+"}"
                    file.writelines(print_data)
                    file.writelines("\n")
              
            file.writelines("\n")
main()
