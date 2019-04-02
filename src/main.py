"""
 * @Author: radsn23
 * @Date: 2019-03-07 12:49:27
 * @Last Modified by:   radsn23
 * @Last Modified time: 2019-03-07 12:49:27
"""
import os,time
import numpy as np
import argparse
import re, math
import pandas as pd
from collections import defaultdict
import hmm
from constants import TRAIN_POS, VALIDATE_POS, \
        TEST_WORDS,VALIDATE_WORDS, TEST_POS,VOCAB,STATS,HOME
from viterbi import Viterbi
from constants import PRED_V_POS,PRED_T_POS


def dataloader( data_file, mode):
    with open(data_file,'r') as inputfile:
        data = inputfile.read().splitlines()

    if mode=='train':
        data = [re.split(r'\t+',d) for d in data]
        train_df = pd.DataFrame(data, columns=['word', 'pos'])
        train_df.word = train_df.word.replace('','NO_WORD')
        train_df.pos = train_df.pos.fillna('START')
        #print(train_df.head(60))
        return train_df

    elif mode=='test' or mode=='validate':
        vocab = hmm.load_file(VOCAB)
        orig_df, prep_df= [],[]
        df = pd.DataFrame(data, columns=['word'])
        for i,word in df.iterrows():
            word = word.iloc[0]
            if not word:
                orig_df.append(word.strip())
                prep_df.append('NO_WORD')

            elif word.strip() not in vocab:
                orig_df.append(word.strip())
                prep_df.append(hmm.assign_rare_tags(word))
            else:
                orig_df.append(word)
                prep_df.append(word)
        assert(len(orig_df)==len(open(data_file,'r').readlines()))
        assert(len(orig_df)==len(prep_df))

        return orig_df,prep_df

    else:
        print('Invalid mode selected, please choose among train,validate,test')
        return pd.DataFrame()


class POSTagger(object):
    def __init__(self, corpus):
        self.corpus = corpus
        self.A = []
        self.B = []

    def train(self):
        print('Training started...')
        start_train = time.time()
        train_df = dataloader(self.corpus + TRAIN_POS, 'train')
        train_df.to_csv(HOME+'data/train.csv')
        #load vocab
        if not os.path.isfile(VOCAB):
            print('Generating vocabulary')
            self.vocab = hmm.make_vocab(train_df)
        else:
            self.vocab = hmm.load_file(VOCAB)
        #load stats
        if not os.path.isfile(STATS):
            print('Generating tagger statistics')
            print(train_df.head())
            self.stats = hmm.generate_stats(train_df)
        else:
            self.stats = [line.strip() for line in open(STATS,'r')]
            emissCounts,transCounts, tagCounts = \
                    hmm.load_stats(self.stats)

        #make A,B
        self.tags = sorted(tagCounts.keys())
        self.A,self.B = hmm.generate_matrices(transCounts,emissCounts,\
                tagCounts,self.tags, self.vocab)

        print('Training finished in '+ str(time.time() - start_train))

    def validate(self):
        print('Validation started...')
        start_val = time.time()
        self.pred_tags = []
        valid_orig, valid_prep = dataloader(self.corpus + \
                VALIDATE_WORDS, 'validate')
        tagger = Viterbi(self.vocab, self.tags, valid_prep, self.A, self.B)
        preds = tagger.decode()
        for word,tag in zip(valid_orig, preds):
            self.pred_tags.append((word,tag))

        with open(PRED_V_POS,'w') as out:
            for word,tag in self.pred_tags:
                if not word:
                    out.write("\n")
                else:
                    out.write("{0}\t{1}\n".format(word,tag))
        out.close()
        print('Validation ended, file has been written in '+ str(time.time()-\
                start_val))


    def test(self):
        print('Test started...')
        start_test = time.time()
        self.pred_tags = []
        test_orig, test_prep = dataloader(self.corpus + TEST_WORDS, 'test')
        tagger = Viterbi(self.vocab, self.tags, test_prep, self.A, self.B)
        preds = tagger.decode()
        for word,tag in zip(test_orig, preds):
            self.pred_tags.append((word,tag))

        with open(PRED_T_POS,'w') as out:
            for word,tag in self.pred_tags:
                if not word:
                    out.write("\n")
                else:
                    out.write("{0}\t{1}\n".format(word,tag))
        out.close()
        print('Test finished, file has been written in '+ str(time.time()-\
                start_test))


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', type=str, required=True, \
            help = 'Add the WSJ folder')
    args = parser.parse_args()
    corpus_dict = {'train':'WSJ_02-21.pos','validate':'WSJ_24.pos',\
            'test':'WSJ_23.words'}
    corpus = args.dir
    print(corpus)

    trainer = POSTagger(corpus)
    trainer.train()
    trainer.validate()
    trainer.test()
