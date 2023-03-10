import re
import numpy as np

class Vocabulary:
    
    def __init__(self, path):
        
        self.path = path
        self.word_to_index = {}
        self.index_to_word = {}
        self.created = False

    def load_file(self):
        
        with open(self.path, 'r', encoding='utf-8') as f:
            self.file = f.read().lower().strip().split('\n')
        
    def create_vocabulary(self):
        
        pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
        lines = [pattern.findall(line) for line in self.file]
        
        self.distribution = {}

        tokens = 0
        n_words = 0
        for words in lines:
            for i, word in enumerate(words):
                i += tokens
                if word not in self.word_to_index:
                    self.word_to_index[word] = i
                    self.index_to_word[i] = word
                    self.distribution[i] = 1
                else:
                    tokens -= 1
                    self.distribution[self.word_to_index[word]] += 1
                    
            tokens += len(words)
            n_words += len(words)
            
        assert len(self.word_to_index) == len(self.index_to_word)
        self.word_to_index['unk'] = -1
        self.index_to_word[-1] = 'unk'
        
        self.sorted_distribution = dict(sorted(self.distribution.items(), key=lambda x: x[0], reverse=False))
                    
        self.created = True
                        
    def __len__(self):
        return len(self.word_to_index)
    
    def __getitem__(self, item):
        import numpy
        assert isinstance(item, (int, str, numpy.int64))
        if isinstance(item, str):
            if item in self.word_to_index:
                return self.word_to_index[item]
            return self.word_to_index['unk']
        if item in self.index_to_word and item != -1:
            return self.index_to_word[item]
        return 'unk'
