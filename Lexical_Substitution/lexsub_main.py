#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# from nltk.tokenize.stanford import StanfordTokenizer
import gensim
import string
import numpy as np

# Participate in the 4705 lexical substitution competition (optional): NO
# Alias: [please invent some name]

def tokenize(s):
    s = "".join(" " if x in string.punctuation else x for x in s.lower())
    return s.split() 

def get_candidates(lemma, pos):
    # Part 1
    possible_synonyms = set()
    for lexeme in wn.lemmas(lemma, pos=pos):
        synset = lexeme.synset()
        for w in synset.lemmas():
            synonym = w.name().replace('_', ' ')
            possible_synonyms.add(synonym)
    possible_synonyms.remove(lemma)
    return possible_synonyms

def smurf_predictor(context):
    """
    Just suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context):
    target_lemma, target_pos = context.lemma, context.pos
    possible_synonyms = {}
    for lexeme in wn.lemmas(target_lemma, pos=target_pos):
        synset = lexeme.synset()
        for w in synset.lemmas():
            synonym = w.name().replace('_', ' ')
            if synonym in possible_synonyms:
                possible_synonyms[synonym]+=w.count()
            else:
                possible_synonyms[synonym]=w.count()

    possible_synonyms.pop(target_lemma, None)
    predictor = [k for k, v in possible_synonyms.items() if v == max(possible_synonyms.values())][0]
    return predictor # replace for part 2


def wn_simple_lesk_predictor(context):
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    target_lemma, target_pos, context_words = context.lemma, context.pos, context.left_context+context.right_context
    context_occur = {} #the dictionary stores the co-occurence between context words and examples+definition
    most_freq_synset = {} #the dictionary stores the co-occurence for one synset
    most_freq_word = {} #get the most frequent synonyms from the highest score synsets
    for word in context_words:
        word = word.lower()
        if word not in string.punctuation:
            if word not in stop_words:
                word = lemmatizer.lemmatize(word)
                context_occur[word]=0

    def update_context_occur(synset):
        '''
        :param synset: input a synset
        :return: sum of the time the context words occurs in the example and definition of that synset
        '''
        for key in context_occur:
            context_occur[key]=0
        (synset.examples()).append(synset.definition())
        examdefi = synset.examples()
        for sentence in examdefi:
            sentence = tokenize(sentence)
            for word in sentence:
                word = lemmatizer.lemmatize(word)
                if word in context_occur:
                    context_occur[word] += 1
        return sum(context_occur.values())

    for lexeme in wn.lemmas(target_lemma, pos=target_pos):
        synset = lexeme.synset()
        score = update_context_occur(synset)
        if synset not in most_freq_synset:
            most_freq_synset[synset] = score
        else:
            most_freq_synset[synset]+=score
        for hyper_syn in synset.hypernyms():
            score = update_context_occur(hyper_syn)
            if hyper_syn not in most_freq_synset:
                most_freq_synset[hyper_syn] = score
            else:
                most_freq_synset[hyper_syn] += score

    max_syn_score = max(most_freq_synset.values())
    if max_syn_score == 0:
        # if there is no co-occurence synsets, then use part2 predictor
        return wn_frequency_predictor(context)
    for key, val in most_freq_synset.items():
        if val == max_syn_score:
            for w in key.lemmas():
                synonym = w.name().replace('_', ' ')
                if synonym in most_freq_word:
                    most_freq_word[synonym] += w.count()
                else:
                    most_freq_word[synonym] = w.count()
    most_freq_word.pop(target_lemma, None)
    if most_freq_word.keys():
        predic = [k for k, v in most_freq_word.items() if v == max(most_freq_word.values())][0]
        return predic
    else:
        return wn_frequency_predictor(context)

     #replace for part 3
   
class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)

    def cos(self, v1, v2):
        return np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))


    def predict_nearest(self,context):
        stop_words = stopwords.words('english')
        target_lemma, target_pos = context.lemma, context.pos
        v1 = self.model.wv[target_lemma]
        possible_synonyms = {}
        for lexeme in wn.lemmas(target_lemma, pos=target_pos):
            synset = lexeme.synset()
            for w in synset.lemmas():
                synonym = w.name().replace('_', ' ')
                synonym_words = synonym.split()
                # if synonym consists of only one word and it is embedded in wv
                # calculate the cosine distance between the target and synonym
                # if synonym not in wv use 'UNK' embedding
                if len(synonym_words)==1 and synonym_words[0] in self.model.wv:
                        v2 = self.model.wv[synonym_words[0]]
                        possible_synonyms[synonym] = self.cos(v1, v2)
                else:
                    # if the synonym compose of multiple words, e.g take up, in order to
                    # first get rid of the stop words, then find the first word in wv if available
                    # calculate the cosine distance between the first word and target
                    # use the first word as the prediction
                    synonym_words = [w for w in synonym_words if w not in stop_words]
                    if synonym_words[0] in self.model.wv:
                        v2 = self.model.wv[synonym_words[0]]
                        possible_synonyms[synonym_words[0]] = self.cos(v1, v2)

        possible_synonyms.pop(target_lemma, None)
        predictor = [k for k, v in possible_synonyms.items() if v == max(possible_synonyms.values())][0]
        return predictor # replace for part 4

    def predict_nearest_with_context(self, context):
        stop_words = stopwords.words('english')
        target_lemma, target_pos, context_left, context_right = context.lemma, context.pos, context.left_context,context.right_context
        sentence_vector = self.model.wv[target_lemma]

        def get_five_context(context, right=True):
            '''
            :param context: left/right context of the target word
            :return: five context words that removes stop words, punctuation
            '''
            context_norm = []
            for word in context:
                word = word.lower()
                if word not in string.punctuation:
                    if word not in stop_words:
                        if word.isnumeric() == False:
                            context_norm.append(word)
                        # context_norm.append(word)
                        # word = word.split('-')
                        # for w in word:
                        #     if w.isnumeric()==False:
                        #         context_norm.append(w)
            if len(context_norm)<=5:
                return context_norm
            elif right:
                return context_norm[:5]
            else:
                return context_norm[-5:]
        context_word = get_five_context(context_left,False)+get_five_context(context_right,True)

        for word in context_word:
            if word in self.model.wv:
                sentence_vector = sentence_vector+self.model.wv[word]
            else:
                sentence_vector = sentence_vector+self.model.wv['UNK']

        possible_synonyms = {}
        for lexeme in wn.lemmas(target_lemma, pos=target_pos):
            synset = lexeme.synset()
            for w in synset.lemmas():
                synonym = w.name().replace('_', ' ')
                synonym_words = synonym.split()
                # if synonym consists of only one word and it is embedded in wv
                # calculate the cosine distance between the target and the
                if len(synonym_words) == 1 and synonym_words[0] in self.model.wv:
                        v2 = self.model.wv[synonym_words[0]]
                        possible_synonyms[synonym] = self.cos(sentence_vector, v2)
                else:
                    # if the synonym compose of multiple words, e.g take up, in order to
                    # first get rid of the stop words, then find the first word in wv if available
                    # calculate the cosine distance between the first word and target
                    # use the first word as the prediction
                    synonym_words = [w for w in synonym_words if w not in stop_words]
                    if synonym_words[0] in self.model.wv:
                        v2 = self.model.wv[synonym_words[0]]
                        possible_synonyms[synonym_words[0]] = self.cos(sentence_vector, v2)

        possible_synonyms.pop(target_lemma, None)
        predictor = [k for k, v in possible_synonyms.items() if v == max(possible_synonyms.values())][0]

        return predictor # replace for part 5

    def predict_competition(self,context):
        stop_words = stopwords.words('english')
        target_lemma, target_pos, context_left, context_right = context.lemma, context.pos, context.left_context, context.right_context
        sentence_vector = self.model.wv[target_lemma]

        def get_five_context(context, right=True):
            '''
            :param context: left/right context of the target word
            :return: five context words that removes stop words, punctuation
            '''
            context_norm = []
            for word in context:
                word = word.lower()
                if word not in string.punctuation:
                    if word not in stop_words:
                        if word.isnumeric() == False:
                            context_norm.append(word)
                        else:
                            context_norm.append('NUMBER')
            if len(context_norm) <= 5:
                return context_norm
            elif right:
                return context_norm[:5]
            else:
                return context_norm[-5:]

        context_word = get_five_context(context_left, False) + get_five_context(context_right, True)

        for word in context_word:
            if word in self.model.wv:
                sentence_vector = sentence_vector + self.model.wv[word]
            else:
                sentence_vector = sentence_vector + self.model.wv['UNK']

        possible_synonyms = {}
        possible_synonyms_count = {}
        for lexeme in wn.lemmas(target_lemma, pos=target_pos):
            synset = lexeme.synset()
            for w in synset.lemmas():
                synonym = w.name().replace('_', ' ')
                synonym_words = [synonym]
                # synonym_words = synonym.split()
                count_w = w.count()
                # if synonym consists of only one word and it is embedded in wv
                # calculate the cosine distance between the target and the
                if len(synonym_words) == 1 and synonym_words[0] in self.model.wv:
                    v2 = self.model.wv[synonym_words[0]]
                    possible_synonyms[synonym] = self.cos(sentence_vector, v2)
                    if synonym in possible_synonyms_count:
                        possible_synonyms_count[synonym] += count_w
                    elif synonym not in possible_synonyms_count:
                        possible_synonyms_count[synonym] = count_w
                else:
                    # if the synonym compose of multiple words, e.g take up, in order to
                    # first get rid of the stop words, then find the first word in wv if available
                    # calculate the cosine distance between the first word and target
                    # use the first word as the prediction
                    synonym_words = [w for w in synonym_words if w not in stop_words]
                    if synonym_words[0] in self.model.wv:
                        v2 = self.model.wv[synonym_words[0]]
                        possible_synonyms[synonym_words[0]] = self.cos(sentence_vector, v2)
                        if synonym_words[0] in possible_synonyms_count:
                            possible_synonyms_count[synonym_words[0]] += count_w
                        elif synonym_words[0] not in possible_synonyms_count:
                            possible_synonyms_count[synonym_words[0]] = count_w

        possible_synonyms.pop(target_lemma, None)
        possible_synonyms_count.pop(target_lemma, None)
        sum_count = [sum(possible_synonyms_count.values()) if sum(possible_synonyms_count.values())!=0 else 1][0]
        # sum_distance = sum(possible_synonyms.values())
        for key in possible_synonyms:
            possible_synonyms[key] = possible_synonyms[key]+0*possible_synonyms_count[key]/sum_count

        predictor = [k for k, v in possible_synonyms.items() if v == max(possible_synonyms.values())][0]

        return predictor

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    predictor = Word2VecSubst(W2VMODEL_FILENAME)
    # print(get_candidates('slow', 'a'))
    for context in read_lexsub_xml(sys.argv[1] ):
        #print(context)  # useful for debugging
        # prediction = smurf_predictor(context)
        # prediction = wn_frequency_predictor(context)
        # prediction = wn_simple_lesk_predictor(context)
        # prediction = predictor.predict_nearest_with_context(context)
        prediction = predictor.predict_competition(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
