"""
 * @Author: radsn23
 * @Date: 2019-03-07 12:48:25
 * @Last Modified by: radsn23
 * @Last Modified time: 2019-03-07 12:48:53
"""
import numpy as np
import pandas as pd
import argparse, os
import string
from collections import defaultdict
from viterbi import Viterbi
from constants import VOCAB,TRAIN_POS,STATS
#import all the suffixes
from constants import NOUN_SUFFIX,VERB_SUFFIX,ADJ_SUFFIX, ADV_SUFFIX
from constants import RARE_TAGS,ALPHA,HOME
alpha = ALPHA

train_df =pd.read_csv(HOME+'data/train.csv')

def make_vocab(train_df, min_num=2):
    """
    Generate the observation space O from the tokens in
    the training set, with their counts
    """

    vocab = defaultdict(int)
    for i,row in train_df.iterrows():
        if row['pos']:
            vocab[row['word']]+=1

    vocab = [key for key,count in vocab.items() if count>=min_num]
    vocab = sorted(vocab)
    #Add for unknown and newlines
    vocab.extend(RARE_TAGS)
    with open(VOCAB, 'w') as out:
        for v in vocab:
            out.write('{0}\n'.format(v))
    out.close()

    print('Vocabulary generated of size '+ str(len(vocab)))

    return vocab

def load_file(inputfile):
    with open(inputfile,"r") as f:
        loaded_file = [x.rstrip('\n') for x in f]
    return loaded_file

def generate_stats(train_df, vocab=VOCAB):
    """
    Generate the counts for emission and transition
    from the training data
    """
    #START state
    prev = "START"
    vocab = set(load_file(VOCAB))
    #print(len(vocab), type(vocab))
    emissCounts = defaultdict(int)
    transCounts = defaultdict(int)
    tagCounts = defaultdict(int)
    for i, row in train_df.iterrows():
        if  not row['pos'].split():
            word='NO_WORD'
            tag='START'
        else:
            word,tag = row['word'],row['pos']
            if word not in vocab:
                word = assign_rare_tags(word)

        emissCounts[" ".join([tag, word])]+=1
        transCounts[" ".join([prev, tag])]+=1
        tagCounts[tag]+=1
        prev=tag

    stats = []

    with open(STATS, "w") as out:

        for k, v in emissCounts.items():
            line = "E {0} {1}\n".format(k,v)
            stats.append(line)
            out.write(line)

        for k,v in transCounts.items():
            line = "T {0} {1}\n".format(k,v)
            stats.append(line)
            out.write(line)

        for tag in tagCounts:
            line = "C {0} {1}\n".format(tag, tagCounts[tag])
            stats.append(line)
            out.write(line)

    out.close()
    print('Statistics file for training data generated with size '\
            +str(len(stats)))

    return stats

def assign_rare_tags(word):
    """
    To deal with open classes and random characters
    """

    punctuation = set(string.punctuation)
    if any(w in punctuation for w in word):
        return "PUNCT"

    elif any(w.isdigit() for w in word):
        return "DIGIT"

    elif any(w.isupper() for w in word):
        return "CAPS"

    elif any(word.endswith(s) for s in NOUN_SUFFIX):
        return "RARE_NOUN"

    elif any(word.endswith(s) for s in VERB_SUFFIX):
        return "RARE_VERB"

    elif any(word.endswith(s) for s in ADJ_SUFFIX):
        return "RARE_ADJ"

    elif any(word.endswith(s) for s in ADV_SUFFIX):
        return "RARE_ADV"

    return "RARE_WORD"


def load_stats(stats):
    emissCounts = defaultdict(dict)
    transCounts = defaultdict(dict)
    tagCounts = defaultdict(dict)

    for line in stats:
        if line.startswith('C'):
            _,tag,count = line.split()
            tagCounts[tag] = int(count)
            continue

        cat,tag,x,count = line.split()
        if cat=='T':
            transCounts[tag][x] = int(count)
        else:

            emissCounts[tag][x] = int(count)
    print('Statistics have been loaded...')
    return emissCounts, transCounts, tagCounts


def generate_matrices(transCounts, emissCounts, \
        tagCounts, tags, vocab):

    """
    Generate

    A:  a KxK matrix of transition probs from
    state i to state j

    B: a KxN matrix of emission probs of obser-
    ving a word o after state i
    """

    K = len(tags)
    N = len(vocab)
    A = [[0]*K for i in range(K)]
    B = [[0]*N for i in range(K)]

    #For A
    for i in range(K):
        for j in range(K):
            prev,tag = tags[i],tags[j]

            if (prev in transCounts) and (tag in \
                    transCounts[prev]):
                count = transCounts[prev][tag]
            else:
                count = 0
            A[i][j] = (count + alpha)/(tagCounts[prev]+ \
                    alpha*K)
    # For B
    for i in range(K):
        for j in range(N):
            tag = tags[i]
            word = vocab[j]
            if word in emissCounts[tag]:
                count = emissCounts[tag][word]
            else:
                count = 0

            B[i][j] = (count + alpha)/(tagCounts[tag] + \
                    alpha*N)
    print('A and B matrices have been generated...')
    return A,B

if __name__=="__main__":
    #for unit testing, probably
    generate_stats(train_df)

