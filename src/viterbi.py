"""
 * @Author: radsn23
 * @Date: 2019-03-07 12:50:41
 * @Last Modified by:   radsn23
 * @Last Modified time: 2019-03-07 12:50:41
"""
import numpy as np
import os,math
from collections import defaultdict
from tqdm import tqdm

class Viterbi(object):
    def __init__(self,O, S, Y, A, B):
        """
        For reference:

        https://www.wikiwand.com/en/Viterbi_algorithm
        """
        self.O = O # Observation Space
        self.S = S # State Space
        self.Y = Y # Sequence of observations
        self.A = A # Transition matrix
        self.B = B # Emission matrix
        self.K = len(self.S)
        self.N = len(self.O)
        self.T = len(self.Y)
        #T1 stores the probabilities of the most likely paths
        self.T1 = [[0]*self.T for i in range(self.K)]
        #T2 stores the tags for the most likely path
        self.T2 = [[None]*self.T for i in range(self.K)]
        self.X = [None]*self.T #Predicted tags

        self.wordIndex = defaultdict(int)
        for i, word in enumerate(self.O):
            self.wordIndex[word] = i


    def decode(self):
        print('Viterbi decoding started..')
        starts = self.S.index("START")
        for i in range(self.K):
            if self.A[starts][i] == 0:
                self.T1[i][0] = np.NINF
            else:
                self.T1[i][0] = math.log(self.A[starts][i])\
                        + math.log(self.B[i][self.wordIndex[\
                        self.Y[0]]])
            self.T2[i][0] = 0

        for i in tqdm(range(1,self.T)):
            for j in range(self.K):
                bestProb = np.NINF
                bestPath = None

                for k in range(self.K):
                    prob = self.T1[k][i-1]+\
                            math.log(self.A[k][j])+ \
                            math.log(self.B[j][self.wordIndex[\
                            self.Y[i]]])
                    if prob>bestProb:
                        bestProb=prob
                        bestPath=k
                self.T1[j][i] = bestProb
                self.T2[j][i] = bestPath

        z = [None]*self.T
        argmax = self.T1[0][self.T-1]
        for i in range(1, self.K):
            if self.T1[i][self.T-1]>argmax:
                argmax = self.T1[i][self.T-1]
                z[self.T-1] = i
        self.X[self.T-1] = self.S[z[self.T-1]]
        for i in range(self.T-1,0,-1):
            z[i-1]=self.T2[z[i]][i]
            self.X[i-1] = self.S[z[i-1]]

        return self.X

