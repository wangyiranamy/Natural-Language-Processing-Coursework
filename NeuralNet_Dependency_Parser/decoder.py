from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import keras

from extract_training_data import FeatureExtractor, State

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        self.uniquepairs =np.load('data/uniquepairs.npy')
        
        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1, len(words)))
        state.stack.append(0)

        while state.buffer:
            features = self.extractor.get_input_representation(words, pos, state)
            features = features.reshape(1,6)
            possible_transitions = {}
            predict_probs = self.model.predict(features)[0]
            for i, prob in enumerate(predict_probs):
                if prob > 0.000000000e+00:
                    possible_transitions[i]=prob
            possible_transitions = sorted(possible_transitions.items(), key=lambda kv: -kv[1])
            # print(possible_transitions)
            for transitionid, prob in possible_transitions:
                action = self.uniquepairs[transitionid][0]
                deprel = self.uniquepairs[transitionid][1]
                if action == 'shift':
                    if len(state.buffer)==1 and not state.stack:
                        state.shift()
                        break
                    elif len(state.buffer)>1:
                        state.shift()
                        break
                    elif len(possible_transitions)==1:
                        state.shift()
                        break
                elif action == 'left_arc':
                    if state.stack and state.stack[-1]!=0:
                        state.left_arc(deprel)
                        break
                else:
                    if state.stack:
                        state.right_arc(deprel)
                        break
            # print(state.stack, state.buffer)
            # TODO: Write the body of this loop for part 4 

        result = DependencyStructure()
        # print(state.deps)
        for p, c, r in state.deps:
            result.add_deprel(DependencyEdge(c, words[c], pos[c], p, r))
        return result 
        

if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)

    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file:
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
        
