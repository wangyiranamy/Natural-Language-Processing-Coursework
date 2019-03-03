import sys
from collections import defaultdict, Counter
import numpy as np
import math
import random
import os
import os.path

"""
COMS W4705 - Natural Language Processing - Fall 2018
Homework 1 - Programming Component: Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile, 'r') as corpus:
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence


def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  


def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """
    input = sequence
    output = []
    if(n==1):
        input.insert(0, 'START')
    else:
        # 'n-1' number of START
        for i in range(0, n-1):
            input.insert(0, 'START')
    input.append('STOP')
    for i in range(0,len(input)-n+1):
            output.append(input[i:i + n])
    return [tuple(l) for l in output]


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count n-grams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)

        #Store the count of all words in the corpus
        #self.totalnumberofwords = sum(Counter(generator).values())
        #self.totalnumberofwords = sum(1 for _ in generator)

    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """

        self.unigramcounts = Counter()
        self.bigramcounts = Counter()
        self.trigramcounts = Counter()

        for each in corpus:
            unigrams = get_ngrams(each, 1)
            bigrams = get_ngrams(each, 2)
            trigrams = get_ngrams(each, 3)

            # self.unigramcounts = sum(Counter(self.unigrams).values())
            # self.bigramcounts = sum(Counter(self.bigrams).values())
            # self.trigramcounts = sum(Counter(self.trigrams).values())

            for unigram in unigrams:
                self.unigramcounts[unigram] += 1

            for bigram in bigrams:
                self.bigramcounts[bigram] += 1

            for trigram in trigrams:
                self.trigramcounts[trigram] += 1

                if trigram[:2] == ('START', 'START'):
                    # For computing the probability of trigrams having ('START', 'START', 'xxx')
                    self.bigramcounts[('START', 'START')] += 1

        return


    def raw_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        if self.bigramcounts[trigram[:2]] != 0:
            rptri = self.trigramcounts[trigram]/self.bigramcounts[trigram[:2]]
            return rptri
        else:
            # Incase the first two words of the trigram are not seen in the training set as a bigram
           return 0.0


    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        if self.unigramcounts[bigram[:1]] != 0:
            rpbi = self.bigramcounts[bigram]/self.unigramcounts[bigram[:1]]
            return rpbi
        else:
            #Incase the first word of the bigram is not seen in the training as a unigram
            return 0.0


    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """
        # P(wi) = count(wi) / count(totalnumberofwords)

        if not hasattr(self, 'totalnumberofwords'):
            # Calculate this attribute if the total number of words of the model hasnt been computed yet
            self.totalnumberofwords = sum(self.unigramcounts.values())
            self.totalnumberofwords -= - self.unigramcounts[('START',)] + self.unigramcounts[('STOP',)]

        rpuni = self.unigramcounts[unigram]/self.totalnumberofwords

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  
        return rpuni

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        # Optional
        i = 0
        newsentence = list()
        thistrigram = (None, 'START', 'START')

        while i < t and thistrigram[2] != 'STOP':
            firstword = thistrigram[1]
            secondword = thistrigram[2]

            # First find all the trigrams starting with these two words
            alltrigrams = [eachtri for eachtri in self.trigramcounts.keys() if eachtri[:2] == (firstword, secondword) ]
            probabilities = [self.raw_trigram_probability(trigram) for trigram in alltrigrams]

            thirdword = np.random.choice([eachtri[2] for eachtri in alltrigrams], 1, p=probabilities)[0]
            thistrigram = (firstword, secondword, thirdword)
            newsentence.append(thirdword)
            i += 1

        return newsentence


    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0

        # Initially, I tried to change the lambda values to see if the accuracy changes, but did not get any big change. So I kept them as we learned in class itself.

        smprob = 0.0
        # smoothedprob = (lambda1*rpuni)+(lambda2*rpbi)+(lambda3*rptri)
        smprob = (lambda1 * self.raw_trigram_probability(trigram)) + (lambda2 * self.raw_bigram_probability(trigram[1:])) + (lambda3 * self.raw_unigram_probability(trigram[2:]))

        return smprob
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        alltrigrams = get_ngrams(sentence, 3)

        trigram_prob = [self.smoothed_trigram_probability(each) for each in alltrigrams]

        return sum(math.log2(probability) for probability in trigram_prob)

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """

        l = 0
        M = 0

        for each in corpus:
            l += self.sentence_logprob(each)
            M += len(each)

        l/=M

        return 2 ** (-l)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp_1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp_2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            total += 1
            correct += (pp_1 < pp_2)
    
        for f in os.listdir(testdir2):
            pp_2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            pp_1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            total += 1
            correct += (pp_2 < pp_1)

        return correct/total


if __name__ == "__main__":

    model = TrigramModel(sys.argv[1])

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
    # Testing perplexity: 
    dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    pp = model.perplexity(dev_corpus)
    print(pp)

    # Essay scoring experiment: 
    acc = essay_scoring_experiment("hw1_data/ets_toefl_data/train_high.txt", "hw1_data/ets_toefl_data/train_low.txt", "hw1_data/ets_toefl_data/test_high", "hw1_data/ets_toefl_data/test_low")
    acc2 = essay_scoring_experiment("hw1_data/ets_toefl_data/test_high/88.txt", "hw1_data/ets_toefl_data/train_low/5301.txt", "hw1_data/ets_toefl_data/test_high", "hw1_data/ets_toefl_data/test_low")
    print(acc)

    print(model.generate_sentence())
