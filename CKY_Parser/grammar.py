"""
COMS W4705 - Natural Language Processing - Spring 2019
Homework 2 - Parsing with Context Free Grammars 
Yassine Benajiba
"""

import sys
from collections import defaultdict
from math import fsum
import math

class Pcfg(object): 
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file): 
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None 
        self.read_rules(grammar_file)      
 
    def read_rules(self,grammar_file):
        
        for line in grammar_file: 
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line: 
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else: 
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()
                    
     
    def parse_rule(self,rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";",1) 
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False. 
        """
        # TODO, Part 1
        for root in self.rhs_to_rules.keys():
            if len(root)==2:
                if root[0].isupper()==False or root[1].isupper()==False:
                    return False
                for head in self.rhs_to_rules[root]:
                    if head[0].isupper()==False:
                        return False
            else:
                if root[0].islower()==False and root[0].isupper()==True:
                    return False
                for head in self.rhs_to_rules[root]:
                    if head[0].isupper()==False:
                        return False
        for root in self.lhs_to_rules.keys():
            prob = math.fsum(num[2] for num in self.lhs_to_rules[root])
            if (prob-1) > 1e-11:
                return False

        return True


if __name__ == "__main__":
    with open(sys.argv[1],'r') as grammar_file:
    # with open('atis3.pcfg', 'r') as grammar_file:
        grammar = Pcfg(grammar_file)
    # print(grammar.lhs_to_rules[('NPBAR')])
    # print(grammar.lhs_to_rules.keys())
    if grammar.verify_grammar():
        print('The grammar is valid.')
    else:
        print('The grammar is not valid in CNF.')
        
