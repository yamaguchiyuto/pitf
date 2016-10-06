import itertools
import copy
import numpy as np

class PITF:
    def __init__(self,alpha=0.0001,lamb=0.1,k=30,max_iter=100,data_shape=None,verbose=0):
        self.alpha = alpha
        self.lamb = lamb
        self.k = k
        self.max_iter = max_iter
        self.data_shape = data_shape
        self.verbose = verbose

    def _init_latent_vectors(self,data_shape):
        latent_vector = {}
        latent_vector['u'] = np.random.normal(loc=0, scale=0.1, size=(data_shape[0],self.k))
        latent_vector['i'] = np.random.normal(loc=0, scale=0.1, size=(data_shape[1],self.k))
        latent_vector['tu'] = np.random.normal(loc=0, scale=0.1, size=(data_shape[2],self.k))
        latent_vector['ti'] = np.random.normal(loc=0, scale=0.1, size=(data_shape[2],self.k))
        return latent_vector

    def _calc_number_of_dimensions(self,data,validation):
        u_max = -1
        i_max = -1
        t_max = -1
        for u,i,t in data:
            if u > u_max: u_max = u
            if i > i_max: i_max = i
            if t > t_max: t_max = t
        if not validation is None:
            for u,i,t in validation:
                if u > u_max: u_max = u
                if i > i_max: i_max = i
                if t > t_max: t_max = t
        return (u_max+1,i_max+1,t_max+1)

    def _draw_negative_sample(self,t):
        r = np.random.randint(self.data_shape[2]) # sample random index
        while r==t:
            r = np.random.randint(self.data_shape[2]) # repeat while the same index is sampled
        return r

    def _sigmoid(self,x):
        return 1.0 / (1.0 + np.exp(-x))

    def _score(self,data):
        if data is None: return "No validation data"
        correct = 0.
        for u,i,answer_t in data:
            predicted = self.predict(u,i)
            if predicted == answer_t: correct += 1
        return correct / data.shape[0]

    def fit(self,data,validation=None):
        if self.data_shape is None: self.data_shape = self._calc_number_of_dimensions(data,validation)
        self.latent_vector_ = self._init_latent_vectors(self.data_shape)
        remained_iter = self.max_iter
        while True:
            remained_iter -= 1
            np.random.shuffle(data)
            for u,i,t in data:
                nt = self._draw_negative_sample(t)
                y_diff = self.y(u,i,t) - self.y(u,i,nt)
                delta = 1-self._sigmoid(y_diff)
                self.latent_vector_['u'][u] += self.alpha * (delta * (self.latent_vector_['tu'][t] - self.latent_vector_['tu'][nt]) - self.lamb * self.latent_vector_['u'][u])
                self.latent_vector_['i'][i] += self.alpha * (delta * (self.latent_vector_['ti'][t] - self.latent_vector_['ti'][nt]) - self.lamb * self.latent_vector_['i'][i])
                self.latent_vector_['tu'][t] += self.alpha * (delta * self.latent_vector_['u'][u] - self.lamb * self.latent_vector_['tu'][t])
                self.latent_vector_['tu'][nt] += self.alpha * (delta * -self.latent_vector_['u'][u] - self.lamb * self.latent_vector_['tu'][nt])
                self.latent_vector_['ti'][t] += self.alpha * (delta * self.latent_vector_['i'][i] - self.lamb * self.latent_vector_['ti'][t])
                self.latent_vector_['ti'][nt] += self.alpha * (delta * -self.latent_vector_['i'][i] - self.lamb * self.latent_vector_['ti'][nt])
            if self.verbose==1: print "%s\t%s" % (self.max_iter-remained_iter, self._score(validation))
            if remained_iter <= 0:
                break
        return self

    def y(self,i,j,k):
        return self.latent_vector_['tu'][k].dot(self.latent_vector_['u'][i]) + self.latent_vector_['ti'][k].dot(self.latent_vector_['i'][j])

    def predict(self,i,j):
        y = self.latent_vector_['tu'].dot(self.latent_vector_['u'][i]) + self.latent_vector_['ti'].dot(self.latent_vector_['i'][j])
        return y.argmax()

    def predict2(self,x):
        y = self.latent_vector_['u'][x[:,0]].dot(self.latent_vector_['tu'].T) + self.latent_vector_['i'][x[:,1]].dot(self.latent_vector_['ti'].T)
        return y.argmax(axis=1)
