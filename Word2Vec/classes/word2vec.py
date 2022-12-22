import numpy as np


class Word2Vec:
    
    
    def __init__(self, vocabulary, window_size, K=2, lr=0.01, 
                 dim=300, random_state=40, distribute=False):
        self.window_size = window_size
        self.random_state = random_state
        self.dim = dim
        self.vocabulary = vocabulary
        self.K = K
        self.lr = lr
        
        self.vocabulary.load_file()
        self.vocabulary.create_vocabulary()
        
        if distribute:
            self.distribution = self.vocabulary.sorted_distribution
            self.index_probability = []
            for index in self.distribution.keys():
                self.index_probability.append(self.distribution[index])
                
            self.index_probability = np.array(self.index_probability) ** 3 / 4
            self.index_probability = self.index_probability / np.sum(self.index_probability)
        else:
            self.index_probability = None
        
        self.random_init()
        
    def single_forward_pass(self, output_word, center_word):
        output_word_index = self.vocabulary[output_word]
        u_o = self.U[:, output_word_index]
        
        center_word_index = self.vocabulary[center_word]
        v_c = self.V[:, center_word_index]
        
        similarity = np.dot(u_o, v_c)
        
        negative_sampling_indexes = np.random.choice(len(self.vocabulary) - 1, 
                                                     size=(self.K,), 
                                                     replace=False, 
                                                     p=self.index_probability)
        
        negative_similarity = np.dot(self.U[:, negative_sampling_indexes].T, v_c)
        
        total_loss = - np.log(self.sigmoid(similarity)) - np.sum(np.log(1 - self.sigmoid(negative_similarity)))
        return total_loss, negative_sampling_indexes
            
    def forward(self, window_words, center_word):
        
        loss = 0
        sampling = []
        for word in window_words:
            l, n_s_i = self.single_forward_pass(word, center_word)
            loss += l
            sampling.append(n_s_i)
            
        return loss, sampling
    
    def backward(self, window_words, center_word, sampling_indexes):
        
        grad_v_c = 0
        grad_window_words = []
        grad_sample_words = []
            
        for n, word in enumerate(window_words):
            new_g_u_s = []
            g_v_c, g_u_w, g_u_s = self.single_backward_pass(word, center_word, sampling_indexes[n])
            grad_v_c += g_v_c[0]
            for sample in g_u_s:
                grad_u_s, index = sample
                grad_u_s *= 2 * self.window_size
                new_g_u_s.append([grad_u_s, index])
                
            grad_window_words.append(g_u_w)
            grad_sample_words.append(new_g_u_s)
            
        return [grad_v_c, g_v_c[1]], grad_window_words, grad_sample_words
            
    def single_backward_pass(self, output_word, center_word, negative_sampling_indexes):
        
        output_word_index = self.vocabulary[output_word]
        u_o = self.U[:, output_word_index]
        
        center_word_index = self.vocabulary[center_word]
        v_c = self.V[:, center_word_index]
        
        similarity = np.dot(u_o, v_c)
        grad_v_c = - (1.0 - self.sigmoid(similarity)) * u_o
        
        negative_similarity = self.sigmoid(np.dot(self.U[:, negative_sampling_indexes].T, v_c)) * self.U[:, negative_sampling_indexes]
        negative_similarity = np.sum(negative_similarity, axis=1)
                
        grad_v_c += negative_similarity
        grad_u_o = - (1.0 - self.sigmoid(similarity)) * v_c
        grad_u_s = []
        
        for index in negative_sampling_indexes:
            u_w = self.U[:, index]
            g_w_s = self.sigmoid(np.dot(u_w, v_c)) * v_c
            grad_u_s.append([g_w_s, index])
            
        return [grad_v_c, center_word_index], [grad_u_o, output_word_index], grad_u_s

    def update(self, grads):
        
        grad_v_c, grad_window_words, grad_sampling_words = grads
        self.V[:, grad_v_c[1]] -= self.lr * grad_v_c[0]
        
        for grad_s in grad_sampling_words:
            for g_s in grad_s:
                grad, index = g_s
                self.U[:, index] -= self.lr * grad
    
        for grad_u_o in grad_window_words:
            self.U[:, grad_u_o[1]] -= self.lr * grad_u_o[0]
                                
    def sigmoid(self, x, derivative=False):
        x += 1e-12
        if not derivative:
            return 1 / (1 + np.exp(-x))    
        return self.sigmoid(x) * (1.0 - self.sigmoid(x))
    
    def random_init(self):
        np.random.seed(self.random_state)
        self.U = np.random.randn(self.dim, len(self.vocabulary)) * 0.1
        self.V = np.random.randn(self.dim, len(self.vocabulary)) * 0.1
            
    def predict(self, words):
        
        assert isinstance(words, list), 'words argument must be a list'
        similarity = np.zeros(self.U.shape[1])
        for word in words:
            index = self.vocabulary[word]
            context_vector = self.U[:, index]
            similarity += self.V.T.dot(context_vector)
        exp = np.exp(similarity - np.max(similarity))
        return exp / np.sum(exp)