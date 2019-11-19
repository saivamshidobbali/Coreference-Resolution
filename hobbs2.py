"""
Implementation of Hobbs' algorithm for pronoun resolution.
Chris Ward, 2014
"""

import sys
import nltk
from nltk.corpus import names
from nltk import Tree
try:
    import queue
except ImportError:
    import Queue as queue

# Labels for nominal heads
nominal_labels = ["NN", "NNS", "NNP", "NNPS", "PRP"]

def get_pos(tree, node):
    """ Given a tree and a node, return the tree position
    of the node.
    """
    print(tree.treepositions())

    for pos in tree.treepositions():
        if tree[pos] == node:
            print(pos)
            return pos

def get_dom_np(sents, pos):
    """ Finds the position of the NP that immediately dominates
    the pronoun.

    Args:
        sents: list of trees (or tree) to search
        pos: the tree position of the pronoun to be resolved
    Returns:
        tree: the tree containing the pronoun
        dom_pos: the position of the NP immediately dominating
            the pronoun
    """
    # start with the last tree in sents
    tree = sents[-1]

    # get the NP's position by removing the last element from
    # the pronoun's
    dom_pos = pos[:-1]
    return tree, dom_pos

def walk_to_np_or_s(tree, pos):
    """ Takes the tree being searched and the position from which
    the walk up is started. Returns the position of the first NP
    or S encountered and the path taken to get there from the
    dominating NP. The path consists of a list of tree positions.

    Args:
        tree: the tree being searched
        pos: the position from which the walk is started
    Returns:
        path: the path taken to get the an NP or S node
        pos: the position of the first NP or S node encountered
    """
    print("Now in walk pos",pos)
    print("tree",tree)
    path = [pos]
    still_looking = True
    while still_looking:
        # climb one level up the tree by removing the last element
        # from the current tree position
        pos = pos[:-1]
        print("walk pos  ",pos)
        path.append(pos)
        # if an NP or S node is encountered, return the path and pos
        if "NP" in tree[pos].label() or tree[pos].label() == "S" or tree[pos].label() == "SBAR" or tree[pos].label() =="SBARQ" or tree[pos].label() =="SINV" or tree[pos].label() =="SQ":
            still_looking = False
        if pos==():
            still_looking = False

    return path, pos

def bft(tree):
    """ Perform a breadth-first traversal of a tree.
    Return the nodes in a list in level-order.

    Args:
        tree: a tree node
    Returns:
        lst: a list of tree nodes in left-to-right level-order
    """
    lst = []
    queue1 = queue.Queue()
    queue1.put(tree)
    while not queue1.empty():
        node = queue1.get()
        lst.append(node)
        for child in node:
            if isinstance(child, nltk.Tree):
                queue1.put(child)
    return lst

def count_np_nodes(tree):
    """ Function from class to count NP nodes.
    """
    np_count = 0
    if not isinstance(tree, nltk.Tree):
        return 0
    elif "NP" in tree.label() and tree.label() not in nominal_labels:
        return 1 + sum(count_np_nodes(c) for c in tree)
    else:
        return sum(count_np_nodes(c) for c in tree)

def check_for_intervening_np(tree, pos, proposal, pro):
    """ Check if subtree rooted at pos contains at least
    three NPs, one of which is:
        (i)   not the proposal,
        (ii)  not the pronoun, and
        (iii) greater than the proposal

    Args:
        tree: the tree being searched
        pos: the position of the root subtree being searched
        proposal: the position of the proposed NP antecedent
        pro: the pronoun being resolved (string)
    Returns:
        True if there is an NP between the proposal and the  pronoun
        False otherwise
    """
    bf = bft(tree[pos])
    bf_pos = [get_pos(tree, node) for node in bf]

    if count_np_nodes(tree[pos]) >= 3:
        for node_pos in bf_pos:
            if "NP" in tree[node_pos].label() \
            and tree[node_pos].label() not in nominal_labels:
                if node_pos != proposal and node_pos != get_pos(tree, pro):
                    if node_pos < proposal:
                        return True
    return False

def traverse_left(tree, pos, path, pro, check=1):
    """ Traverse all branches below pos to the left of path in a
    left-to-right, breadth-first fashion. Returns the first potential
    antecedent found.

    If check is set to 1, propose as an antecedent any NP node
    that is encountered which has an NP or S node between it and pos.

    If check is set to 0, propose any NP node encountered as the antecedent.

    Args:
        tree: the tree being searched
        pos: the position of the root of the subtree being searched
        path: the path taked to get to pos
        pro: the pronoun being resolved (string)
        check: whether or not there must be an intervening NP
    Returns:
        tree: the tree containing the antecedent
        p: the position of the proposed antecedent
    """
    # get the results of breadth first search of the subtree
    # iterate over them
    print("now in traverse left")
    breadth_first = bft(tree[pos])

    # convert the treepositions of the subtree rooted at pos
    # to their equivalents in the whole tree
    bf_pos = [get_pos(tree, node) for node in breadth_first]

    if check == 1:
        for p in bf_pos:
            if p<path[0] and p not in path:
                if "NP" in tree[p].label() and match(tree, p, pro):
                    if check_for_intervening_np(tree, pos, p, pro) == True:
                        return tree, p

    elif check == 0:
        for p in bf_pos:
            if p<path[0] and p not in path:
                if "NP" in tree[p].label() and match(tree, p, pro):
                    return tree, p

    return None, None

def traverse_right(tree, pos, path, pro):
    """ Traverse all the branches of pos to the right of path p in a
    left-to-right, breadth-first manner, but do not go below any NP
    or S node encountered. Propose any NP node encountered as the
    antecedent. Returns the first potential antecedent.

    Args:
        tree: the tree being searched
        pos: the position of the root of the subtree being searched
        path: the path taken to get to pos
        pro: the pronoun being resolved (string)
    Returns:
        tree: the tree containing the antecedent
        p: the position of the antecedent
    """
    breadth_first = bft(tree[pos])
    bf_pos = [get_pos(tree, node) for node in breadth_first]

    for p in bf_pos:
        if p>path[0] and p not in path:
            if "NP" in tree[p].label() or tree[p].label() == "S":
                if "NP" in tree[p].label() and tree[p].label() not in nominal_labels:
                    if match(tree, p, pro):
                        return tree, p
                return None, None

def traverse_tree(tree, pro):
    """ Traverse a tree in a left-to-right, breadth-first manner,
    proposing any NP encountered as an antecedent. Returns the
    tree and the position of the first possible antecedent.

    Args:
        tree: the tree being searched
        pro: the pronoun being resolved (string)
    """
    # Initialize a queue and enqueue the root of the tree
    print("now in traverse tree")
    queue1 = queue.Queue()
    queue1.put(tree)
    while not queue1.empty():
        node = queue1.get()
        # if the node is an NP, return it as a potential antecedent
        if "NP" in node.label() and match(tree, get_pos(tree,node), pro):
            return tree, get_pos(tree, node)
        for child in node:
            if isinstance(child, nltk.Tree):
                queue1.put(child)
    # if no antecedent is found, return None
    return None, None

def match(tree, pos, pro):
    """ Takes a proposed antecedent and checks whether it matches
    the pronoun in number and gender

    Args:
        tree: the tree in which a potential antecedent has been found
        pos: the position of the potential antecedent
        pro: the pronoun being resolved (string)
    Returns:
        True if the antecedent and pronoun match
        False otherwise
    """
    if number_match(tree, pos, pro) and gender_match(tree, pos, pro):
        return True
    return False

def number_match(tree, pos, pro):
    """ Takes a proposed antecedent and pronoun and checks whether
    they match in number.
    """
    m = {"NN":          "singular",
         "NNP":         "singular",
         "he":          "singular",
         "she":         "singular",
         "him":         "singular",
         "her":         "singular",
         "it":          "singular",
         "himself":     "singular",
         "herself":     "singular",
         "itself":      "singular",
         "NNS":         "plural",
         "NNPS":        "plural",
         "they":        "plural",
         "them":        "plural",
         "themselves":  "plural",
         "its" : "plural",
         "PRP":         None}

    # if the label of the nominal dominated by the proposed NP and
    # the pronoun both map to the same number feature, they match
    for c in tree[pos]:
        if isinstance(c, nltk.Tree) and c.label() in nominal_labels:
            if m[c.label()] == m[pro]:
                return True
    return False

def gender_match(tree, pos, pro):
    """ Takes a proposed antecedent and pronoun and checks whether
    they match in gender. Only checks for mismatches between singular
    proper name antecedents and singular pronouns.
    """
    male_names = (name.lower() for name in names.words('male.txt'))
    female_names = (name.lower() for name in names.words('female.txt'))
    male_pronouns = ["he", "him", "himself"]
    female_pronouns = ["she", "her", "herself"]
    neuter_pronouns = ["it", "itself","its"]

    for c in tree[pos]:
        if isinstance(c, nltk.Tree) and c.label() in nominal_labels:
            # If the proposed antecedent is a recognized male name,
            # but the pronoun being resolved is either female or
            # neuter, they don't match
            if c.leaves()[0].lower() in male_names:
                if pro in female_pronouns:
                    return False
                elif pro in neuter_pronouns:
                    return False
            # If the proposed antecedent is a recognized female name,
            # but the pronoun being resolved is either male or
            # neuter, they don't match
            elif c.leaves()[0].lower() in female_names:
                if pro in male_pronouns:
                    return False
                elif pro in neuter_pronouns:
                    return False
            # If the proposed antecedent is a numeral, but the
            # pronoun being resolved is not neuter, they don't match
            elif c.leaves()[0].isdigit():
                if pro in male_pronouns:
                    return False
                elif pro in female_pronouns:
                    return False

    return True

def hobbs(sents, pos):
    """ The implementation of Hobbs' algorithm.

    Args:
        sents: list of sentences to be searched
        pos: the position of the pronoun to be resolved
    Returns:
        proposal: a tuple containing the tree and position of the
            proposed antecedent
    """
    #pos means
    # The index of the most recent sentence in sents
    sentence_id = len(sents)-1
    print("sentence_id",sentence_id)
    # The number of sentences to be searched
    num_sents = len(sents)
    print(num_sents)
    # Step 1: begin at the NP node immediately dominating the pronoun
    print("sents",sents)
    tree, pos = get_dom_np(sents, pos)
    print("hobbs tree",tree)
    print(" NP pos",pos)
    #print(tree[pos])
    # String representation of the pronoun to be resolved
    pro = tree[pos].leaves()[0].lower()
    #pro="him"
    print(pro)
    # Step 2: Go up the tree to the first NP or S node encountered
    path, pos = walk_to_np_or_s(tree, pos)
    print("walk path==",path)
    print("walk pos==",pos)
    # Step 3: Traverse all branches below pos to the left of path
    # left-to-right, breadth-first. Propose as an antecedent any NP
    # node that is encountered which has an NP or S node between it and pos
    proposal = traverse_left(tree, pos, path, pro)
    print("proposal",proposal)

    while proposal == (None, None):

        # Step 4: If pos is the highest S node in the sentence,
        # traverse the surface parses of previous sentences in order
        # of recency, the most recent first; each tree is traversed in
        # a left-to-right, breadth-first manner, and when an NP node is
        # encountered, it is proposed as an antecedent
        if pos == ():
            # go to the previous sentence
            sentence_id -= 1
            # if there are no more sentences, no antecedent found
            if sentence_id < 0:
                return None
            # search new sentence
            proposal = traverse_tree(sents[sentence_id], pro)
            print(proposal)
            if proposal != (None, None):
                return proposal

        # Step 5: If pos is not the highest S in the sentence, from pos,
        # go up the tree to the first NP or S node encountered.
        path, pos = walk_to_np_or_s(tree, pos)

        # Step 6: If pos is an NP node and if the path to pos did not pass
        # through the nominal node that pos immediately dominates, propose pos
        # as the antecedent.
        if "NP" in tree[pos].label() and tree[pos].label() not in nominal_labels:
            for c in tree[pos]:
                if isinstance(c, nltk.Tree) and c.label() in nominal_labels:
                    if get_pos(tree, c) not in path and match(tree, pos, pro):
                        proposal = (tree, pos)
                        if proposal != (None, None):
                            return proposal

        # Step 7: Traverse all branches below pos to the left of path,
        # in a left-to-right, breadth-first manner. Propose any NP node
        # encountered as the antecedent.
        proposal = traverse_left(tree, pos, path, pro, check=0)
        if proposal != (None, None):
            return proposal

        # Step 8: If pos is an S node, traverse all the branches of pos
        # to the right of path in a left-to-right, breadth-forst manner, but
        # do not go below any NP or S node encountered. Propose any NP node
        # encountered as the antecedent.
        if tree[pos].label() == "S":
            proposal = traverse_right(tree, pos, path, pro)
            if proposal != (None, None):
                return proposal

    return proposal


def resolve_reflexive(sents, pos):
    """ Resolves reflexive pronouns by going to the first S
    node above the NP dominating the pronoun and searching for
    a matching antecedent. If none is found in the lowest S
    containing the anaphor, then the sentence probably isn't
    grammatical or the reflexive is being used as an intensifier.
    """
    tree, pos = get_dom_np(sents, pos)

    pro = tree[pos].leaves()[0].lower()

    # local binding domain of a reflexive is the lowest clause
    # containing the reflexive and a binding NP
    path, pos = walk_to_s(tree, pos)

    proposal = traverse_tree(tree, pro)

    return proposal

def walk_to_s(tree, pos):
    """ Takes the tree being searched and the position from which
    the walk up is started. Returns the position of the first S
    encountered and the path taken to get there from the
    dominating NP. The path consists of a list of tree positions.

    Args:
        tree: the tree being searched
        pos: the position from which the walk is started
    Returns:
        path: the path taken to get the an S node
        pos: the position of the first S node encountered
    """
    path = [pos]
    still_looking = True
    while still_looking:
        # climb one level up the tree by removing the last element
        # from the current tree position
        pos = pos[:-1]
        path.append(pos)
        # if an S node is encountered, return the path and pos
        if tree[pos].label() == "S":
            still_looking = False
    return path, pos




def demo():
    tree1 = Tree.fromstring('(S (NP (NNP John) ) (VP (VBD said) (SBAR (-NONE- 0) \
        (S (NP (PRP he) ) (VP (VBD likes) (NP (NNS dogs) ) ) ) ) ) )')
    tree2 = Tree.fromstring('(S (NP (NNP John) ) (VP (VBD said) (SBAR (-NONE- 0) \
        (S (NP (NNP Mary) ) (VP (VBD likes) (NP (PRP him) ) ) ) ) ) )')
    tree3 = Tree.fromstring('(S (NP (NNP John)) (VP (VBD saw) (NP (DT a) \
        (JJ flashy) (NN hat)) (PP (IN at) (NP (DT the) (NN store)))))')
    tree4 = Tree.fromstring('(S (NP (PRP He)) (VP (VBD showed) (NP (PRP it)) \
        (PP (IN to) (NP (NNP Terrence)))))')
    tree5 = Tree.fromstring("(S(NP-SBJ (NNP Judge) (NNP Curry))\
        (VP(VP(VBD ordered)(NP-1 (DT the) (NNS refunds))\
        (S(NP-SBJ (-NONE- *-1))(VP (TO to) (VP (VB begin)\
        (NP-TMP (NNP Feb.) (CD 1))))))(CC and)\
        (VP(VBD said)(SBAR(IN that)(S(NP-SBJ (PRP he))(VP(MD would)\
        (RB n't)(VP(VB entertain)(NP(NP (DT any) (NNS appeals))(CC or)\
        (NP(NP(JJ other)(NNS attempts)(S(NP-SBJ (-NONE- *))(VP(TO to)\
        (VP (VB block) (NP (PRP$ his) (NN order))))))(PP (IN by)\
        (NP (NNP Commonwealth) (NNP Edison)))))))))))(. .))")
    tree6 = Tree.fromstring('(S (NP (NNP John) ) (VP (VBD said) (SBAR (-NONE- 0) \
        (S (NP (NNP Mary) ) (VP (VBD likes) (NP (PRP herself) ) ) ) ) ) )')

    print( "Sentence 1:")
    print( tree1)
    tree, pos = hobbs([tree1], (1,1,1,0,0))
    print( "Proposed antecedent for 'he':", tree[pos], '\n')

    print( "Sentence 2:")
    print( tree2)
    tree, pos = hobbs([tree2], (1,1,1,1,1,0))
    print( "Proposed antecedent for 'him':", tree[pos], '\n')

    print( "Sentence 3:")
    print( tree3)
    print( "Sentence 4:")
    print( tree4)
    tree, pos = hobbs([tree3,tree4], (1,1,0))
    print( "Proposed antecedent for 'it':", tree[pos])
    tree, pos = hobbs([tree3,tree4], (0,0))
    print( "Proposed antecedent for 'he':", tree[pos], '\n')

    print( "Sentence 5:")
    print( tree5)
    tree, pos = hobbs([tree5], (1,2,1,1,0,0))
    print( "Proposed antecedent for 'he':", tree[pos], '\n')

    print( "Sentence 6:")
    print( tree6)
    tree, pos = resolve_reflexive([tree6], (1,1,1,1,1,0))
    print( "Proposed antecedent for 'herself':", tree[pos], '\n')


def main(argv):
    if len(sys.argv) == 2 and argv[1] == "demo":
        demo()
    else:
        if len(sys.argv) > 3 or len(sys.argv) < 2:
            print( "Enter the file and the pronoun to resolve.")
        elif len(sys.argv) == 3:
            p = ["He", "he", "Him", "him", "She", "she", "Her",
                "her", "It", "it", "They", "they","its"]
            r = ["Himself", "himself", "Herself", "herself",
                "Itself", "itself", "Themselves", "themselves"]
            fname = sys.argv[1]
            pro = sys.argv[2]
            print("helloo*********")
            '''
            with open(fname) as f:
                trees = f.read()

            with open(fname) as f:
                sents = f.readlines()
            for s in sents:
                print("*****s****",s)
            '''
            trees= [Tree('SQ', [Tree('VP+VBZ', ['nyt960222.0558'])]), Tree('SQ', [Tree('VP+VBZ', ['A3160'])]), Tree('SQ', [Tree('VP+VBZ', ['BC-GROUNDED-JETS-BOS'])]), Tree('FRAG', [Tree('NP', [Tree('JJ', ['@']), Tree('NN', ['amp'])]), Tree(':', [';']), Tree('NP', [Tree('NN', ['LR'])]), Tree(':', [';'])]), Tree('NP', [Tree('NP+CD', ['02-22'])]), Tree('SQ', [Tree('VP+VBZ', ['BC-GROUNDED-JETS-BOS'])]), Tree('NX+NX', [Tree('JJ', ['NAVY']), Tree('JJ', ['GROUNDS']), Tree('JJ', ['F-14']), Tree('JJ', ["'S"]), Tree('JJ', ['AFTER']), Tree('JJ', ['THIRD']), Tree('JJ', ['RECENT']), Tree('NN', ['CRASH'])]), Tree('NP+VP', [Tree('NP', [Tree('NN', ['('])]), Tree('NP', [Tree('NN', ['For'])]), Tree('VB', ['use']), Tree('PP', [Tree('IN', ['by']), Tree('NP', [Tree('NP', [Tree('NP', [Tree('NNP', ['New']), Tree('NNP', ['York'])]), Tree('NNP', ['Times']), Tree('NNP', ['News'])]), Tree('NNP', ['Service']), Tree('NNS', ['clients']), Tree('NN', [')'])])])]), Tree('PP+PP', [Tree('IN', ['by']), Tree('NP', [Tree('JJ', ['CHRIS']), Tree('NN', ['BLACK'])])]), Tree('NX+NP', [Tree('NNP', ['c.1996']), Tree('DT', ['The']), Tree('NNP', ['Boston']), Tree('NNP', ['Globe'])]), Tree('FRAG', [Tree('NP', [Tree('JJ', ['WASHINGTON']), Tree('JJ', ['@']), Tree('NN', ['amp'])]), Tree(':', [';']), Tree('NP', [Tree('NNP', ['MD'])]), Tree(':', [';']), Tree('NP', [Tree('DT', ['The']), Tree('NN', ['Navy'])]), Tree('NP', [Tree('NP', [Tree('NP', [Tree('VBN', ['suspended']), Tree('NNS', ['operations'])]), Tree('PP', [Tree('IN', ['for']), Tree('NP', [Tree('PRP$', ['its']), Tree('JJ', ['entire']), Tree('NN', ['fleet'])])])]), Tree('PP', [Tree('IN', ['of']), Tree('NP', [Tree('JJ', ['F-14']), Tree('NN', ['airplanes'])])])]), Tree('NP', [Tree('NNP', ['Thursday'])]), Tree('PP', [Tree('IN', ['after']), Tree('NP', [Tree('DT', ['the']), Tree('JJ', ['third']), Tree('``', ['``']), Tree('NNP', ['Tomcat']), Tree("''", ["''"]), Tree('NN', ['crash']), Tree('PP', [Tree('IN', ['in']), Tree('NP', [Tree('DT', ['a']), Tree('NN', ['month'])])])])]), Tree('NP', [Tree('JJ', ['@']), Tree('NN', ['amp'])]), Tree(':', [';']), Tree('NP', [Tree('NNP', ['MD'])]), Tree(':', [';']), Tree('NP', [Tree('NP', [Tree('DT', ['the']), Tree('NN', ['second'])]), Tree('PP', [Tree('IN', ['in']), Tree('NP', [Tree('DT', ['a']), Tree('NN', ['week'])])])]), Tree('.', ['.'])]), Tree('SINV', [Tree('NP', [Tree('DT', ['the']), Tree('NN', ['72-hour'])]), Tree('``', ['``']), Tree('NP', [Tree('NN', ['safety']), Tree('NN', ['stand-down'])]), Tree("''", ["''"]), Tree('VP', [Tree('VBD', ['came'])]), Tree('S', [Tree('ADVP', [Tree('RB', ['shortly'])]), Tree('VP', [Tree('PP', [Tree('IN', ['after']), Tree('NP', [Tree('NP', [Tree('DT', ['an']), Tree('JJ', ['F-14']), Tree('NN', ['fighter'])]), Tree('PP', [Tree('IN', ['from']), Tree('NP', [Tree('DT', ['the']), Tree('NNP', ['USS']), Tree('NNP', ['Nimitz'])])])])]), Tree('VBN', ['crashed']), Tree('PP', [Tree('IN', ['during']), Tree('NP', [Tree('NP', [Tree('NN', ['flight']), Tree('NNS', ['operations'])]), Tree('PP', [Tree('IN', ['in']), Tree('NP', [Tree('DT', ['the']), Tree('NN', ['northern'])])])])])])]), Tree('NP', [Tree('NP', [Tree('NNP', ['Persian']), Tree('NNP', ['Gulf'])]), Tree('PP', [Tree('IN', ['at']), Tree('IN', ['about']), Tree('NP', [Tree('JJ', ['6:30']), Tree('JJ', ['a.m.']), Tree('NN', ['EST'])])])]), Tree('.', ['.'])]), Tree('NP+S', [Tree('NP', [Tree('DT', ['the']), Tree('NN', ['Navy'])]), Tree('VP', [Tree('VBD', ['announced']), Tree('PP', [Tree('IN', ['that']), Tree('NP', [Tree('NP', [Tree('DT', ['the']), Tree('CD', ['two']), Tree('NNP', ['Naval']), Tree('NNP', ['aviators'])]), Tree('PP', [Tree('IN', ['on']), Tree('NP', [Tree('NN', ['board']), Tree('S+VP', [Tree('VBD', ['were']), Tree('VP', [Tree('VBN', ['rescued']), Tree('PP', [Tree('IN', ['by']), Tree('NP', [Tree('DT', ['a']), Tree('JJ', ['SH-60']), Tree('NN', ['helicopter'])])])])])])])])]), Tree('SBAR', [Tree('IN', ['after']), Tree('S', [Tree('NP', [Tree('PRP', ['they'])]), Tree('VP', [Tree('VBN', ['ejected']), Tree('PP', [Tree('IN', ['from']), Tree('NP', [Tree('DT', ['the']), Tree('NN', ['aircraft'])])])]), Tree('.', ['.'])])])])]), Tree('NP+S', [Tree('NP', [Tree('DT', ['the']), Tree('NN', ['Navy'])]), Tree('VP', [Tree('VBD', ['said']), Tree('SBAR', [Tree('IN', ['that']), Tree('S', [Tree('NP', [Tree('DT', ['both']), Tree('NNS', ['men'])]), Tree('VP', [Tree('VBD', ['were']), Tree('PP', [Tree('IN', ['in']), Tree('NP', [Tree('JJ', ['good']), Tree('NN', ['condition'])])])]), Tree('.', ['.'])])])])]), Tree('S', [Tree('NP', [Tree('CD', ['one']), Tree('NNS', ['aviator'])]), Tree('VP', [Tree('VBN', ['suffered']), Tree('NP', [Tree('JJ', ['minor']), Tree('NN', ['abrasions'])])]), Tree('.', ['.'])]), Tree('S', [Tree('NP', [Tree('DT', ['the']), Tree('NN', ['Navy'])]), Tree('VP', [Tree('VBZ', ['is']), Tree('VBG', ['investigating']), Tree('NP', [Tree('DT', ['the']), Tree('NN', ['cause'])])]), Tree('.', ['.'])]), Tree('SINV', [Tree('S', [Tree('NP', [Tree('DT', ['another']), Tree('NNP', ['Tomcat'])]), Tree('VP', [Tree('VBN', ['crashed']), Tree('PP', [Tree('IN', ['off']), Tree('NP', [Tree('NP', [Tree('NP', [Tree('DT', ['the']), Tree('NN', ['coast'])]), Tree('PP', [Tree('IN', ['of']), Tree('NP', [Tree('NNP', ['California'])])])]), Tree('PP', [Tree('IN', ['during']), Tree('NP', [Tree('NP', [Tree('NNS', ['exercises'])]), Tree('PP', [Tree('IN', ['on']), Tree('NP', [Tree('DT', ['the']), Tree('JJ', ['aircraft']), Tree('NN', ['carrier'])])])])])])])])]), Tree('NP', [Tree('NP', [Tree('NNP', ['Carl']), Tree('NNP', ['Vinson'])]), Tree('PP', [Tree('IN', ['on']), Tree('NP', [Tree('NNP', ['Sunday'])])])]), Tree('.', ['.'])]), Tree('S', [Tree('NP', [Tree('DT', ['both']), Tree('JJ', ['crew']), Tree('NNS', ['members'])]), Tree('VP', [Tree('VBD', ['were']), Tree('VP', [Tree('VBN', ['killed'])])]), Tree('.', ['.'])]), Tree('NP+SBAR+S', [Tree('ADVP', [Tree('RB', ['late'])]), Tree('S', [Tree('NP', [Tree('JJ', ['last']), Tree('NN', ['month'])]), Tree(',', [',']), Tree('NP', [Tree('DT', ['an']), Tree('NN', ['F-14'])]), Tree('VP', [Tree('VBN', ['crashed']), Tree('PP', [Tree('IN', ['near']), Tree('NP', [Tree('NNP', ['Nashville'])])])])]), Tree('S', [Tree(',', [',']), Tree('VP', [Tree('VBG', ['killing']), Tree('NP', [Tree('NP', [Tree('NP', [Tree('DT', ['both']), Tree('JJ', ['crew']), Tree('NNS', ['members'])]), Tree('CC', ['and']), Tree('CD', ['three']), Tree('NNS', ['civilians'])]), Tree('PP', [Tree('IN', ['on']), Tree('NP', [Tree('DT', ['the']), Tree('NN', ['ground'])])])])]), Tree('.', ['.'])])]), Tree('S+SBAR', [Tree('IN', ['although']), Tree('S', [Tree('NP', [Tree('DT', ['the']), Tree('NN', ['crash'])]), Tree('NP', [Tree('NNP', ['Thursday'])]), Tree('VP', [Tree('VBD', ['was']), Tree('NP', [Tree('DT', ['the']), Tree('JJ', ['third']), Tree('NNP', ['Tomcat']), Tree('NN', ['crash']), Tree('DT', ['this']), Tree('NN', ['year'])])])]), Tree(',', [',']), Tree('S', [Tree('NP', [Tree('NNP', ['Defense']), Tree('NNP', ['Department'])]), Tree('NP', [Tree('NN', ['spokesman']), Tree('NNP', ['Kenneth']), Tree('NNP', ['Bacon'])]), Tree('VP', [Tree('VBD', ['said']), Tree('SBAR+S', [Tree('NP', [Tree('DT', ['the']), Tree('NN', ['Navy'])]), Tree('VP', [Tree('VBD', ['had']), Tree('VBN', ['uncovered']), Tree('NP', [Tree('NP', [Tree('DT', ['no']), Tree('NN', ['evidence'])]), Tree('PP', [Tree('IN', ['of']), Tree('NP', [Tree('DT', ['a']), Tree('NN', ['flaw'])])])]), Tree('PP', [Tree('IN', ['in']), Tree('NP', [Tree('DT', ['the']), Tree('NN', ['aircraft'])])])])])]), Tree('.', ['.'])])]), Tree('SINV', [Tree('S', [Tree('NP', [Tree('DT', ['A']), Tree('NN', ['safety'])]), Tree('VP', [Tree('VB', ['stand-down']), Tree('VBZ', ['is']), Tree('NP', [Tree('DT', ['a']), Tree('NN', ['procedure'])])])]), Tree('NP', [Tree('NP', [Tree('DT', ['the']), Tree('JJ', ['military']), Tree('NNS', ['uses'])]), Tree('PP', [Tree('TO', ['to']), Tree('NP', [Tree('JJ', ['conduct']), Tree('DT', ['an']), Tree('JJ', ['intensive']), Tree('NN', ['review'])])])]), Tree('.', ['.'])]), Tree('S+SBAR', [Tree('IN', ['during']), Tree('S', [Tree('NP', [Tree('DT', ['this']), Tree('JJ', ['three-day']), Tree('NN', ['period'])]), Tree(',', [',']), Tree('NP', [Tree('NP', [Tree('DT', ['the']), Tree('CD', ['three']), Tree('NNS', ['classes'])]), Tree('PP', [Tree('IN', ['of']), Tree('NP', [Tree('NN', ['F-14s'])])])]), Tree('VP', [Tree('MD', ['will']), Tree('VB', ['stay']), Tree('PRT', [Tree('RP', ['out'])]), Tree('PP', [Tree('IN', ['of']), Tree('NP', [Tree('DT', ['the']), Tree('NN', ['air'])])])])]), Tree('IN', ['so']), Tree('S', [Tree('NP', [Tree('NNP', ['Navy']), Tree('NNP', ['personnel'])]), Tree('VP', [Tree('MD', ['can']), Tree('VB', ['review']), Tree('NP', [Tree('NP', [Tree('JJ', ['maintenance']), Tree('NN', ['procedures'])]), Tree('CC', ['and']), Tree('NN', ['equipment'])])])]), Tree(',', [',']), Tree('S', [Tree('CC', ['plus']), Tree('VP', [Tree('VB', ['check']), Tree('NP', [Tree('NP', [Tree('DT', ['each']), Tree('NN', ['aircraft'])]), Tree('PP', [Tree('IN', ['for']), Tree('NP', [Tree('JJ', ['weaknesses']), Tree('CC', ['or']), Tree('NNS', ['deficiencies'])])])])]), Tree('.', ['.'])])]), Tree('NP+SBAR+S', [Tree('S', [Tree('NP', [Tree('NP', [Tree('NP', [Tree('DT', ['the']), Tree('NN', ['F-14s'])]), Tree('PP', [Tree('IN', ['on']), Tree('NP', [Tree('DT', ['the']), Tree('NN', ['deck'])])])]), Tree('PP', [Tree('IN', ['of']), Tree('NP', [Tree('DT', ['the']), Tree('NN', ['USS'])])])]), Tree('NP', [Tree('NNP', ['America'])]), Tree('VP', [Tree('MD', ['will']), Tree('VB', ['be']), Tree('VBN', ['allowed']), Tree('S+VP', [Tree('TO', ['to']), Tree('VP', [Tree('VB', ['fly']), Tree('PRT', [Tree('RP', ['off'])]), Tree('PP', [Tree('IN', ['because']), Tree('NP', [Tree('DT', ['the']), Tree('JJ', ['aircraft']), Tree('NN', ['carrier'])])])])])])]), Tree('S', [Tree('VP', [Tree('VBZ', ['is']), Tree('VBG', ['returning']), Tree('PP', [Tree('IN', ['from']), Tree('NP', [Tree('DT', ['a']), Tree('NN', ['deployment'])])])]), Tree('.', ['.'])])]), Tree('S', [Tree('NP', [Tree('NN', ['carrier']), Tree('NN', ['decks'])]), Tree('VP', [Tree('VBP', ['are']), Tree('ADVP', [Tree('RB', ['routinely'])]), Tree('VBN', ['cleared']), Tree('PP', [Tree('IN', ['of']), Tree('NP', [Tree('NP', [Tree('JJ', ['aircraft']), Tree('DT', ['the']), Tree('NN', ['day'])]), Tree('PP', [Tree('IN', ['before']), Tree('NP', [Tree('DT', ['the']), Tree('NN', ['carrier']), Tree('NN', ['docks'])])])])])]), Tree('.', ['.'])]), Tree('S+SBAR', [Tree('``', ['``']), Tree('S', [Tree('NP', [Tree('PRP', ['We'])]), Tree('VP', [Tree('VBP', ['do']), Tree('RB', ['not']), Tree('VB', ['know']), Tree('SBAR', [Tree('WHNP', [Tree('WP', ['what'])]), Tree('S+VP', [Tree('VBZ', ['has']), Tree('VP', [Tree('VBD', ['caused']), Tree('NP', [Tree('NP', [Tree('DT', ['these']), Tree('CD', ['three']), Tree('NNS', ['crashes'])]), Tree('DT', ['this']), Tree('NN', ['year'])])])])])])]), Tree(',', [',']), Tree('S', [Tree("''", ["''"]), Tree('NP', [Tree('NNP', ['Bacon'])]), Tree('VP', [Tree('VBD', ['said'])]), Tree('.', ['.'])])]), Tree('NP+S', [Tree('NP', [Tree('``', ['``']), Tree('DT', ['The']), Tree('NN', ['goal']), Tree('PP', [Tree('IN', ['of']), Tree('NP', [Tree('DT', ['the']), Tree('NN', ['stand-down'])])])]), Tree('VP', [Tree('VBZ', ['is']), Tree('S+VP', [Tree('TO', ['to']), Tree('VP', [Tree('VB', ['take']), Tree('NP', [Tree('NP', [Tree('DT', ['a']), Tree('NN', ['time'])]), Tree('PP', [Tree('IN', ['out']), Tree('S+VP', [Tree('TO', ['to']), Tree('VP', [Tree('VB', ['look']), Tree('SBAR', [Tree('IN', ['at']), Tree('S', [Tree('VP', [Tree('VBG', ['everything']), Tree('VBN', ['involved']), Tree('PP', [Tree('IN', ['with']), Tree('NP', [Tree('DT', ['the']), Tree('NN', ['F-14s'])])])]), Tree('.', ['.'])])])])])])])])])])]), Tree('S+SBAR', [Tree('S', [Tree('NP', [Tree('JJ', ['Adm.']), Tree('NNP', ['Jeremy']), Tree('NNP', ['Boorda'])]), Tree(',', [',']), Tree('NP', [Tree('NP', [Tree('DT', ['the']), Tree('NN', ['chief'])]), Tree('PP', [Tree('IN', ['of']), Tree('NP', [Tree('NAC', [Tree('NNP', ['Naval']), Tree('NNP', ['Operations']), Tree(',', [',']), Tree('NNP', ['informed']), Tree('NNP', ['Deputy']), Tree('NNP', ['Defense']), Tree('NNP', ['Secretary']), Tree('NNP', ['John']), Tree('NNP', ['White']), Tree('PP', [Tree('IN', ['of']), Tree('NP', [Tree('DT', ['the']), Tree('JJS', ['latest'])])])]), Tree('NN', ['crash'])])])]), Tree('NP', [Tree('NNP', ['Thursday'])]), Tree('VP', [Tree('VBG', ['morning']), Tree('CC', ['and']), Tree('VBD', ['told']), Tree('SBAR+S', [Tree('NP', [Tree('PRP', ['him'])]), Tree('NP', [Tree('PRP', ['he'])]), Tree('VP', [Tree('VBD', ['was']), Tree('VBG', ['ordering']), Tree('NP', [Tree('NP', [Tree('DT', ['the']), Tree('NN', ['standdown'])]), Tree('PP', [Tree('IN', ['for']), Tree('NP', [Tree('DT', ['the']), Tree('JJ', ['entire']), Tree('NN', ['fleet']), Tree('PP', [Tree('IN', ['of']), Tree('NP', [Tree('QP', [Tree('JJR', ['more']), Tree('IN', ['than']), Tree('CD', ['300'])]), Tree('NNS', ['aircraft'])])])])])])])])])]), Tree(',', [',']), Tree('S', [Tree('NP', [Tree('NNP', ['Bacon'])]), Tree('VP', [Tree('VBD', ['said'])]), Tree('.', ['.'])])]), Tree('S', [Tree('``', ['``']), Tree('SBAR', [Tree('WHNP', [Tree('WP', ['What'])]), Tree('S+VP', [Tree('NP', [Tree('DT', ['the']), Tree('NN', ['Navy'])]), Tree('VBZ', ['is']), Tree('VBG', ['trying']), Tree('TO', ['to']), Tree('VP', [Tree('VBP', ['do']), Tree('VBZ', ['is']), Tree('VB', ['find']), Tree('PRT', [Tree('RP', ['out'])]), Tree('SBAR', [Tree('WHNP', [Tree('WP', ['what'])]), Tree('S+VP', [Tree('VBD', ['caused']), Tree('NP', [Tree('DT', ['these']), Tree('NN', ['accidents'])])])])])])]), Tree(',', [',']), Tree("''", ["''"]), Tree('NP', [Tree('NNP', ['Bacon'])]), Tree('VP', [Tree('VBD', ['said'])]), Tree('.', ['.'])]), Tree('PRN+SBAR', [Tree('IN', ['in']), Tree('S', [Tree('NP', [Tree('DT', ['the']), Tree('JJ', ['past']), Tree('CD', ['five']), Tree('NNS', ['years'])]), Tree(',', [',']), Tree('NP', [Tree('CD', ['32']), Tree('NNS', ['F-14s'])]), Tree('VP', [Tree('VBP', ['have']), Tree('VP', [Tree('VBN', ['crashed'])])]), Tree('.', ['.'])])]), Tree('S', [Tree('NP', [Tree('NP', [Tree('CD', ['one'])]), Tree('PP', [Tree('IN', ['of']), Tree('NP', [Tree('DT', ['the']), Tree('NNP', ['Navy']), Tree('POS', ["'s"])])])]), Tree('NP', [Tree('JJ', ['first']), Tree('JJ', ['female']), Tree('NN', ['pilots'])]), Tree(',', [',']), Tree('NP', [Tree('JJ', ['Lt.']), Tree('NNP', ['Kara']), Tree('NNP', ['Hultgren'])]), Tree(',', [',']), Tree('VP', [Tree('VBD', ['died']), Tree('PP', [Tree('IN', ['while']), Tree('S+VP', [Tree('VBG', ['trying']), Tree('TO', ['to']), Tree('VP', [Tree('VB', ['land']), Tree('NP', [Tree('NP', [Tree('DT', ['an']), Tree('NN', ['F-14'])]), Tree('PP', [Tree('IN', ['on']), Tree('NP', [Tree('DT', ['an']), Tree('JJ', ['aircraft']), Tree('NN', ['carrier'])])])])])])]), Tree('PP', [Tree('IN', ['in']), Tree('NP', [Tree('CD', ['1994'])])])]), Tree('.', ['.'])]), Tree('NP+S', [Tree('NP', [Tree('DT', ['the']), Tree('NN', ['Navy'])]), Tree('VP', [Tree('VBD', ['concluded']), Tree('SBAR', [Tree('IN', ['that']), Tree('S', [Tree('NP', [Tree('NN', ['engine']), Tree('NN', ['failure'])]), Tree('VP', [Tree('VBD', ['caused']), Tree('NP', [Tree('DT', ['that']), Tree('NN', ['crash'])])]), Tree('.', ['.'])])])])]), Tree('S+SBAR', [Tree('S', [Tree('NP', [Tree('DT', ['the']), Tree('NNP', ['Tomcat']), Tree(',', [',']), Tree('DT', ['a']), Tree('NP+QP', [Tree('$', ['$']), Tree('CD', ['38']), Tree('CD', ['million'])]), Tree(',', [',']), Tree('JJ', ['twin-engine']), Tree('NN', ['aircraft'])]), Tree('VP', [Tree('VBN', ['built']), Tree('PP', [Tree('IN', ['by']), Tree('NP', [Tree('DT', ['the']), Tree('NNP', ['Grumman']), Tree('NNP', ['Aerospace']), Tree('NNP', ['Corp.'])])])])]), Tree(',', [',']), Tree('S', [Tree('VP', [Tree('VBZ', ['is']), Tree('SBAR+S', [Tree('NP', [Tree('DT', ['the']), Tree('JJ', ['standard']), Tree('NN', ['fighter'])]), Tree('VP', [Tree('VB', ['plane']), Tree('VBN', ['used']), Tree('PP', [Tree('IN', ['by']), Tree('NP', [Tree('NP', [Tree('DT', ['the']), Tree('NN', ['Navy'])]), Tree('PP', [Tree('IN', ['for']), Tree('NP', [Tree('JJ', ['close-in']), Tree('NN', ['air']), Tree('NN', ['combat'])])])])])])])]), Tree('.', ['.'])])]), Tree('S', [Tree('NP', [Tree('DT', ['the']), Tree('NN', ['aircraft'])]), Tree('VP', [Tree('VBZ', ['has']), Tree('VBN', ['been']), Tree('PP', [Tree('IN', ['in']), Tree('NP', [Tree('NP', [Tree('NNS', ['operation'])]), Tree('PP', [Tree('IN', ['since']), Tree('NP', [Tree('CD', ['1973'])])])])])]), Tree('.', ['.'])]), Tree('S', [Tree('NP', [Tree('NP', [Tree('DT', ['the']), Tree('JJS', ['oldest']), Tree('NN', ['version'])]), Tree('PP', [Tree('IN', ['of']), Tree('NP', [Tree('DT', ['the']), Tree('NN', ['aircraft'])])])]), Tree(',', [',']), Tree('NP', [Tree('DT', ['the']), Tree('NN', ['F-14A'])]), Tree(',', [',']), Tree('VP', [Tree('VBZ', ['is']), Tree('VBN', ['scheduled']), Tree('S+VP', [Tree('TO', ['to']), Tree('VP', [Tree('VB', ['be']), Tree('VBN', ['phased']), Tree('PRT', [Tree('RP', ['out'])]), Tree('PP', [Tree('IN', ['by']), Tree('NP', [Tree('DT', ['the']), Tree('NN', ['year']), Tree('CD', ['2004'])])])])])]), Tree('.', ['.'])]), Tree('NP+S', [Tree('NP', [Tree('NP', [Tree('DT', ['the']), Tree('NN', ['crash']), Tree('NN', ['rate'])]), Tree('PP', [Tree('IN', ['for']), Tree('NP', [Tree('DT', ['the']), Tree('NN', ['F-14'])])])]), Tree('VP', [Tree('VBD', ['dropped']), Tree('ADVP', [Tree('RB', ['dramatically'])]), Tree('PP', [Tree('IN', ['after']), Tree('NP', [Tree('NP', [Tree('DT', ['the']), Tree('JJ', ['initial']), Tree('NNS', ['years'])]), Tree('PP', [Tree('IN', ['of']), Tree('NP', [Tree('NN', ['deployment'])])])])]), Tree('CC', ['and']), Tree('VBZ', ['has']), Tree('ADVP', [Tree('RB', ['generally'])]), Tree('VB', ['gone']), Tree('PRT', [Tree('RP', ['down'])]), Tree('SBAR', [Tree('IN', ['since']), Tree('S', [Tree('ADVP', [Tree('RB', ['then'])]), Tree('PP', [Tree('IN', ['except']), Tree('NP', [Tree('PP', [Tree('IN', ['for']), Tree('NP', [Tree('JJ', ['occasional']), Tree('NN', ['spurts'])])]), Tree('PP', [Tree('IN', ['of']), Tree('NP', [Tree('NN', ['accidents'])])])])]), Tree(',', [',']), Tree('VP', [Tree('VBG', ['according']), Tree('PP', [Tree('TO', ['to']), Tree('NP', [Tree('NNP', ['Naval']), Tree('NNS', ['records'])])])]), Tree('.', ['.'])])])])]), Tree('SBARQ', [Tree('SQ', [Tree('VBZ', ['bacon']), Tree('VBD', ['said']), Tree('NP', [Tree('DT', ['the']), Tree('NN', ['Navy'])]), Tree('VP', [Tree('VBD', ['had']), Tree('NP', [Tree('NP', [Tree('DT', ['no']), Tree('JJ', ['single']), Tree('NN', ['explanation'])]), Tree('PP', [Tree('IN', ['for']), Tree('NP', [Tree('DT', ['the']), Tree('NNS', ['increases'])])])]), Tree('PP', [Tree('IN', ['in']), Tree('NP', [Tree('JJ', ['F-14']), Tree('NN', ['accidents'])])])])]), Tree('.', ['.'])]), Tree('S+SBAR', [Tree('``', ['``']), Tree('S', [Tree('NP', [Tree('DT', ['The']), Tree('NN', ['Navy'])]), Tree('VP', [Tree('VB', ['informs']), Tree('NP', [Tree('PRP', ['me'])])])]), Tree('IN', ['that']), Tree('S', [Tree('NP', [Tree('PRP', ['they'])]), Tree('VP', [Tree('VBP', ['have']), Tree('VBN', ['been']), Tree('VB', ['unable']), Tree('S+VP', [Tree('TO', ['to']), Tree('VP', [Tree('VB', ['find']), Tree('NP', [Tree('NP', [Tree('DT', ['a']), Tree('JJ', ['common']), Tree('NN', ['thread'])]), Tree('PP', [Tree('TO', ['to']), Tree('NP', [Tree('DT', ['these']), Tree('NN', ['accidents'])])])])])])])]), Tree(',', [',']), Tree('S', [Tree("''", ["''"]), Tree('NP', [Tree('NNP', ['Bacon'])]), Tree('VP', [Tree('VBD', ['said'])]), Tree('.', ['.'])])]), Tree('SINV', [Tree('``', ['``']), Tree('S', [Tree('NP', [Tree('DT', ['This'])]), Tree('VP', [Tree('VBZ', ['is']), Tree('NP', [Tree('DT', ['a']), Tree('NN', ['mystery'])])])]), Tree('.', ['.']), Tree("''", ["''"])]), Tree('NX+NX', [Tree('JJ', ['NYT-02-22-96']), Tree('NN', ['1924EST'])])]
            #trees = [Tree.fromstring(s) for s in sents]
            #print("tree",trees)
            ##trees = Tree.fromstring(sents)
            print("******###############",trees[26])
            print("#####")
            #find him in the list
            index=[]

            for list_ele in trees:
                #str_list_ele = str(list_ele)
                leaves=list_ele.leaves()
                print(leaves)
                if pro in leaves:
                    print("***##**",list_ele)
                    index.append(trees.index(list_ele))
                else:
                    print("not")
            for pis in index:
                print("p-->",pis)
                print("part trees",trees[pis])
                pos = get_pos(trees[pis], pro) # ******include all the pronouns in the sentence*******
                print("positiom",pos)
                #pos = pos[:-1]
                print("positiom",pos)

                if pro in p:
                    tree, pos = hobbs(trees[:pis+1], pos)  #trees = all sentences
                    for t in trees:
                        print( t, '\n')
                    print( "Proposed antecedent for '"+pro+"':", tree[pos])
                elif pro in r:
                    tree, pos = resolve_reflexive(trees, pos)
                    for t in trees:
                        print( t, '\n')
                    print( "Proposed antecedent for '"+pro+"':", tree[pos])
                else:
                    print("pronoun not in list")

if __name__ == "__main__":
    main(sys.argv)
