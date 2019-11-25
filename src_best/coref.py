import sys
import nltk
import re
import xml.etree.ElementTree as ET
from nltk.corpus import wordnet
from nltk.corpus import names
import copy
from nltk import Tree
import os
from difflib import SequenceMatcher
from nltk.chunk import ne_chunk
from nltk import pos_tag
import random
from nltk.stem import WordNetLemmatizer
from nltk.tree import Tree

#from lib.stat_parser.parser import Parser as CkyStatParser                                                                                                                                                  
import spacy

nlp = spacy.load('en_core_web_sm')

nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('names')

RESPONSE_DIR = "./responses"


stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
                    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
                    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
dates = ["today","tomorrow", "sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "yesterday", "day", "night"]
male_pronouns = ['he','his','him','himself']                                                                                                                                                            
neutral_pronouns = ['it','its', 'itself']
female_pronouns = ['she','her','hers','herself']
ambiguous_pronouns_singular = ['you','me','i','yours', 'my', 'mine']
plural_pronouns = ['they','their','those','them','these', 'all', 'theirs', 'we', 'ours', 'our', 'everybody', 'us']
appositive_dict = {}


pronouns_list = []
sent_num_for_pronouns = []
Lemmatizer = WordNetLemmatizer()
flag_full_np = 0

def get_gender(string):
    labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
                     [(name, 'female') for name in names.words('female.txt')])
    random.shuffle(labeled_names)

    def gender_features(word):
        return {'last_letter': word[-1]}                                                                                                                                                                    

    featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
    train_set, test_set = featuresets[500:], featuresets[:500]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    return classifier.classify(gender_features(string))

# returns singular, plural, or 'None' for a given pos.
def get_plurality(pos):
    numberDict = {
        "NN":         "singular",
        "NNP":        "singular",
        "he":         "singular",
        "she":        "singular",
        "him":        "singular",
        "her":        "singular",
        "it":         "singular",
        "himself":    "singular",
        "herself":    "singular",
        "itself":     "singular",
        "NNS":        "plural",
        "NNPS":       "plural",
        "they":       "plural",
        "them":       "plural",
        "themselves": "plural",
        "PRP":        None,
    }

    if pos in numberDict:
        return numberDict[pos]
    else:
        return None



def Most_Common(lst):                                                                                                                                                                                       
    if(lst == []):
        return
    else:
        return max(((item, lst.count(item)) for item in set(lst)), key=lambda a: a[1])[0]


articles = ['a', 'an' ,'the']
def in_btw_word_matcher(anaphor, coref):
    global articles
    global stopwords
    anaphor  = anaphor.split()
    
    coref = coref.split('-')
    
    if len(coref) == 1:
          return "None"

    pronouns = ['he','his','him','himself', 'it','its', 'itself','she','her','hers','herself','you','me','i','yours', 'my', 'mine' 'they','their','those','them','these', 'all', 'theirs', 'we', 'ours', 'our', 'everybody', 'us']

    for word in anaphor:
        if word not in articles and word not in pronouns and word not in stopwords: 
            for word_coref in coref:
                   if word == word_coref:
                          return word_coref
    return "None"            


def in_btw_word_matcher_space(anaphor, coref):
    global articles
    global stopwords
    anaphor  = anaphor.split()
    
    coref = coref.split()
    
    if len(coref) == 1:
          return "None"

    pronouns = ['he','his','him','himself', 'it','its', 'itself','she','her','hers','herself','you','me','i','yours', 'my', 'mine' 'they','their','those','them','these', 'all', 'theirs', 'we', 'ours', 'our', 'everybody', 'us']

    for word in anaphor:
        if word not in articles and word not in pronouns and word not in stopwords: 
            for word_coref in coref:
                   if word == word_coref:
                          return word_coref
    return "None"            

  
    

def find_coref(anaphor, named_entity_resolution_list, sent_num, np_coref_list_copy, np_list_copy, sent_num_of_coref_list, anaphor_idx):
    ans = []
    anaphor_indices = range(0,len(named_entity_resolution_list))
    global flag_full_np
    global pronouns_list
    global sent_num_for_pronouns

    #PRONOUN MATCH
    np = anaphor['text']

    for i,pronoun in enumerate(pronouns_list):
       
       if (sent_num_for_pronouns[i] > sent_num_of_coref_list[anaphor_idx]) and (pronoun == np):
                  ans.append(str(sent_num_for_pronouns[i])+" "+pronoun)
                  pronouns_list[i] = "-1"
                  break
       elif (sent_num_for_pronouns[i] > sent_num_of_coref_list[anaphor_idx]):
                  if(np in male_pronouns and pronoun in male_pronouns):
                          ans.append(str(sent_num_for_pronouns[i])+" "+pronoun)
                          pronouns_list[i] = "-1"
                          break

                  elif(np in female_pronouns and pronoun in female_pronouns):
                          ans.append(str(sent_num_for_pronouns[i])+" "+pronoun)
                          pronouns_list[i] = "-1"
                          break
      
                  elif(np in neutral_pronouns and pronoun in neutral_pronouns):
                          ans.append(str(sent_num_for_pronouns[i])+" "+pronoun)
                          pronouns_list[i] = "-1"
                          break
  
                  elif(np in ambiguous_pronouns_singular and pronoun in ambiguous_pronouns_singular):
                          ans.append(str(sent_num_for_pronouns[i])+" "+pronoun)
                          pronouns_list[i] = "-1"
                          break

                  elif(np in plural_pronouns and pronoun in plural_pronouns):
                          ans.append(str(sent_num_for_pronouns[i])+" "+pronoun)
                          pronouns_list[i] = "-1"
                          break
    """
    if ((index != -1) and (sent_num_of_coref_list[anaphor_idx] < sent_num_for_pronouns[index])):
     if(np in male_pronouns and np in pronouns_list):
       if len(noun_phrases[i]) == 2 and noun_phrases[i][0] != noun_phrases[j][0]:
               noun_phrases[i].append(noun_phrases[j][0])

       elif(np1[1] in female_pronouns and np2[1] in female_pronouns):
           if len(noun_phrases[i]) == 2 and noun_phrases[i][0] != noun_phrases[j][0]:
               noun_phrases[i].append(noun_phrases[j][0])

       elif(np1[1] in neutral_pronouns and np2[1] in neutral_pronouns):
           if len(noun_phrases[i]) == 2 and noun_phrases[i][0] != noun_phrases[j][0]:
               noun_phrases[i].append(noun_phrases[j][0])

       elif(np1[1] in ambiguous_pronouns_singular and np2[1] in ambiguous_pronouns_singular):
           if len(noun_phrases[i]) == 2 and noun_phrases[i][0] != noun_phrases[j][0]:
               noun_phrases[i].append(noun_phrases[j][0])

       elif(np1[1] in plural_pronouns and np2[1] in plural_pronouns):
           if len(noun_phrases[i]) == 2 and noun_phrases[i][0] != noun_phrases[j][0]:
               noun_phrases[i].append(noun_phrases[j][0])
    """  
    """
    #PROPER NOUNS
    semantic_class = []
    uppercase = re.compile(".*[A-Z].*")
    np1_list = anaphor['text'].split()
   
    np2_list = np2[1].split()
    #for pronoun in pronouns_list:
      if uppercase.match(np1[1]) and (not np1[1][-1].find("The ") > -1) or (not np1[1].find("A ") > -1):
          for synset in wn.synsets(np1_list[-1].lower(), wn.NOUN):
              semantic_class = semantic_class + [synset.lexname()]
      
          classs = Most_Common(semantic_class)
          if (classs == 'noun.person') and (np2_list[-1].lower() in male_pronouns or np2_list[-1].lower() in female_pronouns or np2_list[-1].lower() in ambiguous_pronouns_singular):
              if(len(np1)==2 and noun_phrases[i][0] != noun_phrases[j][0]):
                  np1.append(noun_phrases[j][0])
          if (classs == 'noun.person' or classs == 'noun.plant' or classs == 'noun.state' or classs == 'noun.process' or classs == 'noun.object' 
              or classs == 'noun.atrifact' or classs == 'noun.phenomenon') and (np2_list[-1].lower() in neutral_pronouns or np2_list[-1].lower() in plural_pronouns):
              if(len(np1)==2 and noun_phrases[i][0] != noun_phrases[j][0]):
                  np1.append(noun_phrases[j][0]) 
    """


    #print(anaphor['text'])
    
    for idx  in anaphor_indices:

        coref = named_entity_resolution_list[idx]   
       
        if(len(coref['text'].strip()) <= 2):
               continue

        if (int(sent_num_of_coref_list[anaphor_idx]) >= int( sent_num[idx])):
                   continue
       
        if (coref['text'] == "None") or (coref['text'].lower() in stopwords) or (coref['text'].split()[-1].lower() in stopwords):
             continue


  

        #print(sent_num[idx],"===",anaphor['text'], "===================", coref['text'])
        # exact string match
        if anaphor['text'] == coref['text']:
                  print(sent_num[idx],"===",anaphor['text'], "===================", coref['text'])
                  flag_full_np = 1
                  ans.append(str(sent_num[idx])+" "+np_list_copy[idx]+" FULL")
                  named_entity_resolution_list[idx]['label'] = "None"
                  named_entity_resolution_list[idx]['text'] = "None"
                  continue

        """
        # substring match
        INDEX = coref['text'].find(anaphor['text'])
        LENGTH = len(anaphor['text'])
        if INDEX != -1:          
                  temp = str(np_list_copy[idx])
                  ans.append(str(sent_num[idx])+" "+temp[INDEX:INDEX+LENGTH]+ " FULL")
                  named_entity_resolution_list[idx]['label'] = "None"
                  named_entity_resolution_list[idx]['text'] = "None"
                  continue
        """
        """ 
        INDEX = anaphor['text'].find(coref['text'])
        if INDEX != -1:          
                  ans.append(str(sent_num[idx])+" "+str(np_list_copy[idx])+ " FULL")
                  named_entity_resolution_list[idx]['label'] = "None"
                  named_entity_resolution_list[idx]['text'] = "None"
                  continue
        """
        # head noun match
        if anaphor['text'].split()[-1] == coref['text'].split()[-1]:

                  print("HN",sent_num[idx],"===",anaphor['text'], "===================", coref['text'])
                  ans.append(str(sent_num[idx])+" "+np_list_copy[idx])
                  named_entity_resolution_list[idx]['label'] = "None"
                  named_entity_resolution_list[idx]['text'] = "None"
                  continue


        # hiphen match
        ret_word = in_btw_word_matcher(anaphor['text'], coref['text'])
        if ret_word != "None":
                  print("HY",sent_num[idx],"===",anaphor['text'], "===================", coref['text'])
                 
                  for word in np_list_copy[idx].split('-'):
                       print(word)
                       if ret_word == word.lower():
                           ret_word = word

                  ans.append(str(sent_num[idx])+" "+ret_word)
                  named_entity_resolution_list[idx]['label'] = "None"
                  named_entity_resolution_list[idx]['text'] = "None"
                  continue

        """
        ret_word = in_btw_word_matcher_space(anaphor['text'], coref['text'])
        if ret_word != "None":
                  
                  for word in np_list_copy[idx].split():
                       print(word)
                       if ret_word == word.lower():
                           ret_word = word

                  ans.append(str(sent_num[idx])+" "+ret_word)
                  named_entity_resolution_list[idx]['label'] = "None"
                  named_entity_resolution_list[idx]['text'] = "None"
                  continue
                 
        """           
        # partial string match 
   
        if SequenceMatcher(None, anaphor['text'], coref['text']).ratio() > 0.75:

                    print("PM",sent_num[idx],"===",anaphor['text'], "===================", coref['text'])
                    """
                    matches  =   SequenceMatcher(None, anaphor['text'], coref['text']).get_matching_blocks()

                    size = 0
                    for match in matches:
                            if size < match.size:
                                size = match.size
                                max_match_block = match
 
                    ans.append(str(sent_num[idx])+" "+ np_coref_list_copy[anaphor_idx][max_match_block.a:max_match_block.a + max_match_block.size]+" FULL")
                    """
                    ans.append(str(sent_num[idx])+" "+np_list_copy[idx])
                    named_entity_resolution_list[idx]['label'] = "None"
                    named_entity_resolution_list[idx]['text'] = "None"
                    continue

        # ner match
        if anaphor['label'] is not None and anaphor['label'] == coref['label']:
             if SequenceMatcher(None, anaphor['text'], coref['text']).ratio() > 0.65:

                print("ner",sent_num[idx],"===",anaphor['text'], "===================", coref['text'])
                ans.append(str(sent_num[idx])+" "+np_list_copy[idx])

                named_entity_resolution_list[idx]['label'] = "None"
                named_entity_resolution_list[idx]['text'] = "None"
                continue
                
                
        
        flag_date = -1
        #DATE MATCH
        np1_list = anaphor['text'].split()
        np2_list = coref['text'].split()
        for npp1 in np1_list:
                    for npp2 in np2_list:
                        h1 = re.findall(r"(..?)[-/](..?)[-/](..?)|(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",npp1.lower())
                        h2 = re.findall(r"(..?)[-/](..?)[-/](..?)|(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",npp2.lower())
                        if(h1 != [] and npp2.lower() in dates or h2 != [] and npp1.lower() in dates):
                              ans.append(str(sent_num[idx])+" "+np_list_copy[idx])
                              named_entity_resolution_list[idx]['label'] = "None"
                              named_entity_resolution_list[idx]['text'] = "None"
                              flag_date = 0
                              break

    
                        elif(anaphor['text'].lower() in dates and coref['text'].lower() in dates):
                              ans.append(str(sent_num[idx])+" "+np_list_copy[idx])
                              named_entity_resolution_list[idx]['label'] = "None"
                              named_entity_resolution_list[idx]['text'] = "None"
                              flag_date = 0
                              break
                    if flag_date == 0:
                         break
        if flag_date == 0:
             continue
        
        #ABBREVATION MATCH
        flag = 0
        flag1 = 0
        np1_list = anaphor['text'].split()
        np2_list = coref['text'].split()
        if(len(np2_list) == 1 and len(np1_list) == len(np2_list[0])):
                    for i,npp1 in enumerate(np1_list):
                        if np2_list[0][i] == '.':
                              continue

                        if(npp1[0].lower()==np2_list[0][i].lower()):
                            continue
                        else:
                            flag = 1
                            break
       
                    if(flag == 0):
                       ans.append(str(sent_num[idx])+" "+np_list_copy[idx])
                       named_entity_resolution_list[idx]['label'] = "None"
                       named_entity_resolution_list[idx]['text'] = "None"
                       continue
      
        """ 
        #SEMANTIC MATCH
        Semantic_Class1 = []
        Semantic_Class2 = []

        for synset in wordnet.synsets(anaphor['text'].split()[-1], pos=wordnet.NOUN):
              Semantic_Class1 = Semantic_Class1 + [synset.lexname()]
              
        for syn_set in  wordnet.synsets(coref['text'].split()[-1], pos=wordnet.NOUN):
                  Semantic_Class2 = Semantic_Class2 + [syn_set.lexname()]
   
        if(Most_Common(Semantic_Class1) == Most_Common(Semantic_Class2)):
                    print("WORDNET")
                    ans.append(str(sent_num[idx])+" "+np_list_copy[idx])
                    named_entity_resolution_list[idx]['label'] = "None"
                    named_entity_resolution_list[idx]['text'] = "None"
  
        """
        """
        anaphor_lemma = Lemmatizer.lemmatize(anaphor['text'])
        coref_lemma = Lemmatizer.lemmatize(coref['text'])
        #print(anaphor['text'], "*******",coref['text'], "************", anaphor_lemma, "********", coref_lemma)
        if anaphor_lemma == coref_lemma:
                     ans.append(str(sent_num[idx])+" "+np_list_copy[idx])
 
                     named_entity_resolution_list[idx]['label'] = "None"
                     named_entity_resolution_list[idx]['text'] = "None"
                     continue
 
        elif SequenceMatcher(None, anaphor_lemma, coref_lemma).ratio() > 0.85:
                     ans.append(str(sent_num[idx])+" "+np_list_copy[idx])
 
                     named_entity_resolution_list[idx]['label'] = "None"
                     named_entity_resolution_list[idx]['text'] = "None"
                     continue
                 
        """
        w1 = wordnet.synsets(anaphor['text'].split()[-1], pos=wordnet.NOUN)
        w2 = wordnet.synsets(coref['text'].split()[-1], pos=wordnet.NOUN)
        if w1 and w2:
                 print(w1)
                 print(w2)
                 if(w1[0].wup_similarity(w2[0])) > 0.90:
                    print("wordnet")
                    ans.append(str(sent_num[idx])+" "+np_list_copy[idx])
 
                    named_entity_resolution_list[idx]['label'] = "None"
                    named_entity_resolution_list[idx]['text'] = "None"
                    continue
        """
        # check for gender and number, or string agreement
        if anaphor['gender'] == coref['gender'] and  anaphor['plurality'] == coref['plurality']:
                    ans.append(str(sent_num[idx])+" "+np_list_copy[idx])
                    named_entity_resolution_list[idx]['label'] = "None"
                    named_entity_resolution_list[idx]['text'] = "None"
        """
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
    global pronouns_list
    global sent_num_for_pronouns

    #cky_stat_parser = CkyStatParser() 
    pronouns_list = []
    sent_num_for_pronouns = []
    temp_sentence_parse_list = []
    sentence_parse_dict = {}
    pronoun_count = -1

    for sentence in content:
        
        xml_data = ET.fromstring(sentence)
        pure_text = ET.tostring(xml_data, encoding='utf8', method='text')

        #parse_tree = cky_stat_parser.nltk_parse(pure_text) 
        wordlist = nltk.word_tokenize(str(pure_text))
        pos_tagged  =  nltk.pos_tag(wordlist)

        #cp = nltk.RegexpParser(grammar)
        #parse_tree = cp.parse(pos_tagged)
        #temp_sentence_parse_list.append(parse_tree)

        for (a,b) in pos_tagged:
              if b == 'PRP':
                  pronoun_count += 1
                  pronouns_list.append(a)
                  #sentence_parse_dict[pronoun_count] = temp_sentence_parse_list
                  sent_num_for_pronouns.append(xml_data.attrib['ID'])

    return (pronouns_list, sent_num_for_pronouns)

def process(filename): 
   
    global pronouns_list
    global sent_num_for_pronouns

    grammar = "NP: {<DT>?<JJ.*>*<NN.*>+}"
    np_list = []
   
    np_coref_list = []
    ner_sentences = []
    sent_num_of_coref_list = []
    coref_id_list = []

    with open(filename) as f:
         content = f.readlines()
    
    for string in content:
        xml_data = ET.fromstring(string)

        for child in xml_data:
            sent_num_of_coref_list.append(xml_data.attrib['ID'])
            coref_id_list.append(child.attrib['ID'])
            np_coref_list.append(child.text)


    sent_num = []
    parse_tree=""
    sent_num_for_pronouns = []
    sentence_parse_dict = []

    pronouns_list, sent_num_for_pronouns = pronouns(content)
    #hobbs_resolution(pronouns_list, sentence_parse_dict)
    #print(sentence_parse_dict)
    global appositive_dict

    for sentence in content:
        xml_data = ET.fromstring(sentence)
        pure_text = ET.tostring(xml_data, encoding='utf8', method='text')
        wordlist = nltk.word_tokenize(str(pure_text))
        pos_tagged  =  nltk.pos_tag(wordlist)
        cp = nltk.RegexpParser(grammar)
        parse_tree = cp.parse(pos_tagged)
        for node in parse_tree.subtrees(filter=lambda t: t.height() < 4 and (t.label() == 'NP')):
            x = ""
            for i,elem in enumerate(node):
                 if i == 0:
                    x = elem[0]
                 else:
                    x = x +" "+elem[0]
            np_list.append(x)
            sent_num.append(xml_data.attrib['ID'])

        """
        doc = nlp(str(pure_text))

        for chunk in doc.noun_chunks:
            np_list.append(chunk.text)
            sent_num.append(xml_data.attrib['ID'])
        """
        """
        for child in xml_data:
            if child.text in np_list:
                 index = np_list.index(child.text)
                 del np_list[index]
                 del sent_num[-1]
        """

        #appositive = ""
        for j, np in enumerate(np_list):
             for child in xml_data:
                   if sent_num[j] == xml_data.attrib['ID'] and  child.text.find(np) != -1:
                     """
                     if j != len(np_list)-1:
                         appositive = np_list[j+1]
                     """
                     print(np_list[j])
                     np_list[j] = "-1"
                     sent_num[j] = "-1"


        np_list = filter(lambda x: x != "-1",np_list) 
        sent_num = filter(lambda x: x != "-1", sent_num)

        """
        for child in xml_data:
               index = pure_text.find(child.text) + len(child.text)
               if index > len(str(pure_text))-2:
                    break
               if pure_text[index] == ",":
                      if(pure_text[index+1:].startswith(appositive)):
                                 appositive_dict[child.text] = appositive            


        print(appositive_dict)
        """
    #cky_stat_parser = CkyStatParser()    
    for np in np_list:
        print(np)
        #tagged = cky_stat_parser.nltk_parse(np.lower())
        name =  get_ner_names(nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(np))))
        label  = get_ner_labels(nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(np))))

        if name:
            namedEntity = label[-1]
        else:
            namedEntity = None

        ner_sentences.append({"text": np.lower(),
                               "NE" : [x.lower() for x in name],
                              "label": namedEntity,
                              "gender": get_gender(np.lower())})

        #"plurality": get_plurality(tagged.pos()[-1][1]),

                
    ner_coref_sentences = []
    for np in np_coref_list:

        #tagged = cky_stat_parser.nltk_parse(np.lower())
        name =  get_ner_names(nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(np))))
        label  = get_ner_labels(nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(np))))

        if name:
            namedEntity = label[-1]
        else:
            namedEntity = None

        ner_coref_sentences.append({"text": np.lower(),
                               "NE" : [x.lower() for x in name],
                              "label": namedEntity,
                              "gender": get_gender(np.lower())})
    
    np_list_copy = list(np_list)
    np_coref_list_copy = list(np_coref_list)
    
    np_coref_list = [x.lower() for x in np_coref_list]
    np_list = [x.lower() for x in np_list]
    #print("np_coref_list", np_coref_list)
    #print("np_list", np_list)
    
    filename = os.path.basename(filename)
    filename = RESPONSE_DIR+"/"+ filename.split('.')[0]+".response"
    file = open(filename,"a+")

    #hobbs_resolution(pronouns_list, parse_tree_list)
    global flag_full_np 
    for anaphor_idx, anaphor in enumerate(np_coref_list):
            file.writelines("<COREF ID=\""+ coref_id_list[anaphor_idx]+"\">"+ np_coref_list_copy[anaphor_idx]+ "</COREF>")
            file.writelines("\n")
                       
            for elem in ner_coref_sentences:
                 if elem['text'] == anaphor:
                       anaphor_ner = elem
                       break             
                
            ans = find_coref(anaphor_ner, ner_sentences, sent_num, np_coref_list_copy, np_list_copy, sent_num_of_coref_list, anaphor_idx)
            if ans != None:
                for elem in ans:
                    elem = elem.split()
                    
                    if elem[-1] == "FULL":
                          print_data ="{"+elem[0]+"} "+"{"+' '.join(elem[1:-1])+"}"
                    else:
                          print_data ="{"+elem[0]+"} "+"{"+elem[-1]+"}"

                    file.writelines(print_data)
                    file.writelines("\n")

            flag_full_np = 0 
            file.writelines("\n")
main()
