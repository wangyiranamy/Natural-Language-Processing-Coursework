"""
COMS W4705 - Natural Language Processing - Spring 2019
Homework 2 - Parsing with Context Free Grammars 
Yassine Benajiba
"""
import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg

### Use the following two functions to check the format of your data structures in part 3 ###
def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and \
          isinstance(split[0], int)  and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str): # Leaf nodes may be strings
                continue 
            if not isinstance(bps, tuple):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(bps))
                return False
            if len(bps) != 2:
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(bps))
                return False
            for bp in bps: 
                if not isinstance(bp, tuple) or len(bp)!=3:
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(bp))
                    return False
    return True

def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True



class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar): 
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar

    def is_in_language(self, tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        # TODO, part 2
        parsetable = []  # n*(n+1)
        for i in range(len(tokens)):
            parsetable.append([])
            for j in range(len(tokens)+1):
                parsetable[i].append(set())
        # Initialize the bottom words
        for i in range(len(tokens)):
            parsetable[i][i+1] = set([head[0] for head in self.grammar.rhs_to_rules[(tokens[i],)]])
        for length in range(2, len(tokens)+1):
            for i in range(0, len(tokens)-length+1):
                j = i+length
                for k in range(i+1, j):
                    for b in parsetable[i][k]:
                        for c in parsetable[k][j]:
                            setbc = set([head[0] for head in self.grammar.rhs_to_rules[(b, c)]])
                            parsetable[i][j] = parsetable[i][j].union(setbc)
        if self.grammar.startsymbol in parsetable[0][-1]:
            return True
        else:
            return False
       
    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """

        # TODO, part 3
        table = {}
        probs = {}
        for i in range(len(tokens)):
            dicw = dict((head[0], math.log(head[2]))for head in self.grammar.rhs_to_rules[(tokens[i],)])
            table[i, i + 1] = {}
            probs[i, i + 1] = {}
            for (h, p) in dicw.items():
                table[i, i+1][h] = tokens[i]
                probs[i, i+1][h] = p
        for length in range(2, len(tokens) + 1):
            for i in range(0, len(tokens) - length + 1):
                j = i + length
                table[i, j] = {}
                probs[i, j] = {}
                best = {}
                for k in range(i + 1, j):
                    for b in table[i, k]:
                        for c in table[k, j]:
                            dicbc = dict((head[0], math.log(head[2])) for head in self.grammar.rhs_to_rules[(b, c)])
                            for h, p in dicbc.items():
                                if h not in best.keys():
                                    prob = probs[i, k][b] + probs[k, j][c] + p
                                    best[h] ={}
                                    best[h][(b, c, i, j, k)] = prob
                                else:

                                    prob = probs[i, k][b] + probs[k, j][c] + p
                                    best[h][(b, c, i, j, k)] = prob
                for nt in best:
                    bb, cc, bi, bj, bk = max(best[nt], key=lambda k: best[nt][k])
                    table[i, j][nt] = ((bb, bi, bk), (cc, bk, bj))
                    probs[i, j][nt] = best[nt][(bb, cc, bi, bj, bk)]
        return table, probs


def get_tree(chart, i, j, nt):
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    # TODO: Part 4
    def get_tree_help(chart, i, j, nt):
        tree = (nt,)
        if type(chart[i, j][nt]) == tuple:
            left, right = chart[i, j][nt]
            tree += get_tree_help(chart, left[1], left[2], left[0])
            tree += get_tree_help(chart, right[1], right[2], right[0])
            tree = (tree,)
        else:
            tree += (chart[i, j][nt],)
            tree = (tree,)
        return tree
    return get_tree_help(chart, i, j, nt)[0]
 
       
if __name__ == "__main__":
    
    with open('atis3.pcfg','r') as grammar_file: 
        grammar = Pcfg(grammar_file)
        parser = CkyParser(grammar)
        # toks ='i would like to travel to westchester .'.split()
        toks =['flights', 'from','miami', 'to', 'cleveland','.']
        # toks = ['miami', 'flights', 'cleveland', 'from', 'to', '.']
        print(parser.is_in_language(toks))
        print('=============================')
        table, probs = parser.parse_with_backpointers(toks)
        print(table)
        print(probs)
        print(get_tree(table, 0, len(toks), grammar.startsymbol))
        assert check_table_format(table)
        assert check_probs_format(probs)

