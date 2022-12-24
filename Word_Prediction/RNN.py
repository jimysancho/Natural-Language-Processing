import numpy as np


class RNN: 
    
    def __init__(self, hidden_units, input_units, output_units, lr=0.01, 
                 dtype='many-to-many', cost_type='mse'):
                
        self.type = {'many-to-many': [True, False], 'many-to-one': [False, True]}
        self.cost_type = {'mse', 'negative-log'}
        
        assert cost_type in self.cost_type, f'{cost_type} is not a valid type for the cost function. Try on of {self.cost_type.keys()}'
        assert dtype in self.type, f'{dtype} is not a valid type for the RNN. Try one of {self.type.keys()}'
        
        
        self.cost = cost_type
        self.dtype = self.type[dtype]
        self.hidden_units = hidden_units
        self.output_units = output_units
        self.input_units = input_units
        self.lr = lr
        self.__init_weigths()
        
    def forward(self, input_seq, target_seq):
        
        outputs = {}
        hidden_outputs = {}
        predictions = {}
        hidden_prev = np.zeros_like(self.b)
        hidden_states = {-1: hidden_prev}
        cost = 0
        many_to_many, _ = self.dtype
        if many_to_many:
            for t, (input_t, target_t) in enumerate(zip(input_seq, target_seq)):
                z = self.U.dot(input_t) + self.W.dot(hidden_prev) + self.b
                hidden_outputs[t] = z.copy()
                hidden_prev = self.sigmoid(z)
                hidden_states[t] = hidden_prev.copy()
                
                output = self.V.dot(hidden_prev)
                outputs[t] = output.copy()
                prediction = self.softmax(output)
                predictions[t] = prediction.copy()
                cost += self.compute_cost(output, target_t)
            
            return cost, predictions, outputs, hidden_outputs, hidden_states
        
        for t, input_t in enumerate(input_seq):
            z = self.U.dot(input_t) + self.W.dot(hidden_prev) + self.b
            hidden_outputs[t] = z.copy()
            hidden_prev = self.sigmoid(z)
            hidden_states[t] = hidden_prev.copy()
            
            output = self.V.dot(hidden_prev)
            outputs[t] = output.copy()
            
        prediction = output[t].copy()
        cost = self.compute_cost(output, target_seq[0])            
        return cost, prediction, outputs, hidden_outputs, hidden_states
    
    def backward(self, input_seq, target_seq, predictions, outputs, hidden_outputs, hidden_states):
        
        self.grad_W = np.zeros_like(self.W)
        self.grad_V = np.zeros_like(self.V)
        self.grad_U = np.zeros_like(self.U)
        self.grad_b = np.zeros_like(self.b)
        
        dh_grad_next = np.zeros_like(self.b)
        
        for t in reversed(range(len(input_seq))):
            
            target_t, input_t, output_t, prediction_t, hidden_t = target_seq[t], input_seq[t], predictions[t], outputs[t], hidden_states[t]
            h_t = hidden_outputs[t]
            hidden_prev = hidden_states[t-1]
                        
            if (t+1) in hidden_outputs:
                hidden_t_next = hidden_outputs[t+1]
            else:
                hidden_t_next = np.zeros_like(hidden_outputs[t])
                        
            dy = (prediction_t - target_t)
            
            dh = self.V.T.dot(dy) + self.W.T.dot(self.sigmoid(hidden_t_next, True) * dh_grad_next)
            dh_grad_next = dh.copy()
            
            self.grad_V += dy.dot(hidden_t.T)
            self.grad_W += (dh * self.sigmoid(h_t, True)).dot(hidden_prev.T)
            self.grad_U += (dh * self.sigmoid(h_t, True)).dot(input_t.T)
            self.grad_b += (dh * self.sigmoid(h_t, True))
    
    def update_weigths(self):
        self.U -= self.lr * self.grad_U
        self.V -= self.lr * self.grad_V
        self.b -= self.lr * self.grad_b
        self.W -= self.lr * self.grad_W
        
    def __init_weigths(self):
        
        self.W = np.random.randn(self.hidden_units, self.hidden_units)
        self.U = np.random.randn(self.hidden_units, self.input_units)
        self.V = np.random.randn(self.output_units, self.hidden_units)
        self.b = np.random.randn(self.hidden_units, 1)
        
    def sigmoid(self, z, derivative=False):
        if not derivative:
            return 1.0 / (1.0 + np.exp(-z))
        return self.sigmoid(z, False) * (1.0 - self.sigmoid(z, False))
    
    def compute_cost(self, y_pred, y_target):
       if self.cost == 'mse':
           return np.sum((y_pred - y_target) ** 2)
       return -np.mean(y_target * np.log(y_pred + 1e-12))
    
    def softmax(self, prediction):
        exp = np.exp(prediction - np.max(prediction))
        return exp / np.sum(exp, axis=1, keepdims=True)
    
    def predict(self, input_seq):
        
        outputs = {}
        hidden_outputs = {}
        predictions = {}
        hidden_prev = np.zeros_like(self.b)
        hidden_states = {-1: hidden_prev}      
          
        for t, input_t in enumerate(input_seq):
            z = self.U.dot(input_t) + self.W.dot(hidden_prev) + self.b
            hidden_outputs[t] = z.copy()
            hidden_prev = self.sigmoid(z)
            hidden_states[t] = hidden_prev.copy()
            
            output = self.V.dot(hidden_prev)
            outputs[t] = output.copy()
            prediction = self.softmax(output)
            predictions[t] = prediction.copy()
            
        prediction = [output[1] for output in outputs.items()]
        return prediction
        
            
