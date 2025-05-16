import sys
from collections import defaultdict, Counter
import math
import random
import os
import os.path
import numpy as np

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
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



def get_ngrams(sequence:list, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1 
    """
    sequence.append("STOP")
    p=['START']*(n-1)
    p.extend(sequence)
    k=[]
    for i in range(n-1,len(p)):
      k.append(tuple(p[i-n+1:i+1]))
    return k


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = {} # might want to use defaultdict or Counter instead
        self.bigramcounts = {} 
        self.trigramcounts = {} 

        ##Your code here
        unigrams=[]
        bigrams=[]
        trigrams=[]
        for x in corpus:
            unigrams.extend(get_ngrams(x,1))
            bigrams.extend(get_ngrams(x,2))
            trigrams.extend(get_ngrams(x,3))
        
        self.tot_unigrams=len(unigrams)
        self.unigramcounts=Counter(unigrams)
        self.bigramcounts=Counter(bigrams)
        self.trigramcounts=Counter(trigrams)

        return 

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        if trigram[0:2] not in self.bigramcounts:
            return self.unigramcounts[trigram[2:]]/self.tot_unigrams
        return self.trigramcounts[trigram]/self.bigramcounts[trigram[0:2]]

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        if bigram[0:1] not in self.unigramcounts:
            return self.unigramcounts[bigram[1:]]/self.tot_unigrams
        return self.bigramcounts[bigram]/self.unigramcounts[bigram[0:1]]
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  
        return self.unigramcounts[unigram]/self.tot_unigrams

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        sentence=[]
        start=('START','START')

        for _ in range(t):
            
            filtered_items = {k[2]: self.raw_trigram_probability(k) for k, _ in self.trigramcounts.items() if k[0:2]==start and k[2]!='UNK'}
            
            if not filtered_items:
                break
            else:
                filtered_items={k: v for k, v in sorted(filtered_items.items(), key=lambda item: item[1], reverse=True)}
                if len(filtered_items)>2:
                    k=sum(list(filtered_items.values())[:len(filtered_items)//2])
                    b=np.random.choice(list(filtered_items.keys())[:len(filtered_items)//2], p=[x/k for x in list(filtered_items.values())[:len(filtered_items)//2]])
                else:
                    k=sum(list(filtered_items.values()))
                    b=np.random.choice(list(filtered_items.keys()), p=[x/k for x in list(filtered_items.values())])
                sentence.append(str(b))
                if str(b)=='STOP':
                    break
                start=(start[1], str(b))

        return sentence    

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0

        return lambda1*self.raw_trigram_probability(trigram)+lambda2*self.raw_bigram_probability(trigram[1:])+lambda3*self.raw_unigram_probability(trigram[2:])
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        p=get_ngrams(sentence, 3)
        prob=0
        for x in p:
            prob+=math.log2(self.smoothed_trigram_probability(x))

        return prob

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        M=0
        tot=0
        for x in corpus:
            tot+=self.sentence_logprob(x)
            M+=len(x)+1 #include STOP
        l=tot/M
        
        return 2**(-l)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0

        for f in os.listdir(testdir1):
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            total += 1
            if pp1 < pp2:
                correct += 1

        for f in os.listdir(testdir2):
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            total += 1
            if pp1 > pp2:
                correct += 1

        return correct / total

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
    acc = essay_scoring_experiment('ets_toefl_data/train_high.txt', "ets_toefl_data/train_low.txt", "ets_toefl_data/test_high", "ets_toefl_data/test_low")
    print(acc)

