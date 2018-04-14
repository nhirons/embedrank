import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import cosine_similarity

class EmbedRank:
    
    def load_word_embeddings(self, words, word_embeddings):
        self.words = words
        self.X = word_embeddings # rows as words, columns as embedded features
        
    def load_json(self, path):
        with open(path) as json_data:
            d = json.load(json_data)
        print('Loading words'.ljust(30,'.'), end = '')
        words = np.array(list(d.keys()))
        print('Done.')
        print('Loading embeddings'.ljust(30,'.'), end = '')
        word_embeddings = np.array(list(d.values()))
        self.load_word_embeddings(words, word_embeddings)
        print('Done.')
    
    def generate_2D_example(self, examples):
        self.words = list(map(chr, range(65, 65+examples)))
        self.X = np.random.uniform(low = 0, high = 10, size = (examples,2))
    
    def plot2D(self):
        if self.X.shape[1] == 2:
            fig, ax = plt.subplots()
            ax.scatter(self.X[:,0], self.X[:,1])
            for i, txt in enumerate(self.words):
                ax.annotate(txt, (self.X[i,0],self.X[i,1]), xytext = (self.X[i,0]+0.15,self.X[i,1]-0.15))
            plt.show()

    def load_similarity_matrix(self, type = 'cosine', gamma_scale = None):
        print('Creating S'.ljust(30,'.'), end = '')
        if type == 'cosine':
            self.S = cosine_similarity(self.X)
        elif type == 'rbf':
            self.S = rbf_kernel(self.X, gamma = gamma_scale*(1/len(self.X)))
        self.S_below = np.zeros_like(self.S) # initialize threshold mask
        print('Done.')

    def threshold(self, thres = 0.2):
        print('Thresholding'.ljust(30,'.'), end = '')
        self.S_below = self.S < thres
        self.S[self.S_below] = 0
        print('Done.')
    
    def load_transition_matrix(self, scale = 'softmax', tau = 0.5):
        print('Loading T'.ljust(30,'.'), end = '')
        below = self.S_below.sum(axis = 1)[:, np.newaxis]
        if scale == 'softmax':
            row_sums = np.exp(self.S / tau).sum(axis = 1)[:, np.newaxis]
            row_sums_adj = row_sums - (below + np.exp(1/ tau)) # Adjust for thresholds and diagonals
            self.T = np.exp(self.S / tau) / row_sums_adj
        elif scale == 'linear':
            row_sums_adj = self.S.sum(axis = 1)[:, np.newaxis] - 1
            self.T = self.S / row_sums_adj
        to_keep = 1 - (np.eye(len(self.S)) + self.S_below)
        self.T *= to_keep
        print('Done.')

    def pagerank(self, alpha, max_iter = 50, eps = 10e-9):
        '''Where T is row stochastic (i.e. (i,j) 
        represents edge probability from i to j)'''
        n = self.T.shape[0]
        G = alpha * self.T + (1 - alpha) * (1 / n) * np.ones_like(self.T)
        v_prev = np.zeros((n,))
        v = np.ones((n,)) / n
        i = 0
        while (np.linalg.norm(v-v_prev, 1) > eps) and (i < max_iter):
            print('Power iteration {}'.format(i).ljust(30,'.'), end = '')
            v_prev = v
            v = v @ G
            i += 1
            print('Done.')
        return v
    
    def load_word_probs(self):
        print('Running PageRank'.ljust(30,'.'), end = '')
        G = nx.from_numpy_matrix(np.asmatrix(self.T))
        self.probs = nx.pagerank_scipy(G)
        print('Done.')
    
    def load_ranks(self):
        print('Gettings ranks'.ljust(30,'.'), end = '')
        sort_inds = np.array(sorted(self.probs, key=self.probs.get, reverse = True))
        self.ranked = np.array(self.words)[sort_inds]
        self.ranked_labels = sort_inds.argsort()
        print('Done.')
    
    def plot_2D_with_ranks(self):
        if self.X.shape[1] == 2:
            rank_labels = [word+': '+ str(rank+1) for word,rank in zip(self.words, self.ranked_labels)]
            fig, ax = plt.subplots()
            ax.scatter(self.X[:,0], self.X[:,1])
            for i, txt in enumerate(rank_labels):
                ax.annotate(txt, (self.X[i,0],self.X[i,1]), xytext = (self.X[i,0]+0.15,self.X[i,1]-0.15))
            plt.show()


'''
er = EmbedRank()
er.load_json('embeddings-freq.json')
#er.generate_2D_example(15)
#er.plot2D()
er.load_similarity_matrix('cosine')
er.threshold(0.2)
er.load_transition_matrix(scale = 'linear')
#er.load_word_probs()
#er.load_ranks()
#er.plot_2D_with_ranks()
'''
