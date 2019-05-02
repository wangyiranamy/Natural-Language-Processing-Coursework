# Natural-Language-Processing-Coursework
Lexical Substitution

Implementation on a lexical substitution task, using both WordNet and pre-trained Word2Vec word embeddings.

Packages Requirement:
nltk wordnet
gensim

Description:
• lexsub_trial.xml - input trial data containing 300 sentences with a single target word each.
• gold.trial - gold annotations for the trial data (substitues for each word suggested by 5 judges).
• lexsub_xml.py - an XML parser that reads lexsub_trial.xml into Python objects.
• lexsub_main.py - this is the main scaffolding code I completed.lexsub_main.py. loads the XML file, calls a predictor method on each context, and then print output suitable for the SemEval scoring script. The purpose of the predictor methods is to select an appropriate lexical substitute for the word in context. 
• score.pl - the scoring script provided for the SemEval 2007 lexical substitution task.

The instance variables of Context are as follows:
• cid - running ID of this instance in the input file (needed to produce the correct output
for the scoring script).
• word_form - the form of the target word in the sentence (for example 'tighter').
• lemma - the lemma of the target word (for example 'tight').
• pos - this can be either 'n' for noun, 'v' for verb, 'a', for adjective, or 'r' for adverb.
• left_context - a list of tokens that appear to the left of the target word. For example
['Anyway', ',', 'my', 'pants', 'are', 'getting']
• right_context - a list of tokens that appear to the right of the target word. For example
['every','day','.']


