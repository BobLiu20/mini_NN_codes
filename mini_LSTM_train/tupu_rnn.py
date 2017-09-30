import numpy as np
import pickle

class tupu_rnn(object):
    def __init__(self, train_data=[], label_data=[]):
        super(tupu_rnn, self).__init__()
        if not train_data:
            # print 'init for predict, not for training.'
            return
        # Training data. should be simple plain text file
        self.train_data = train_data
        self.label_data = label_data
        if len(self.train_data) != 2 * len(self.label_data):
            assert "invalid for your training data and val data."
        # repeat val data. eg: [1,2,3] to [1, 1, 2, 2, 3, 3]
        self.label_data = [i for i in self.label_data for j in range(2)]
        char_set = list(set(self.train_data + self.label_data))
        self.data_size, self.charset_size = len(self.train_data), len(char_set)
        # mapping char and index
        self.char_to_idx = { ch: i for i, ch in enumerate(char_set) }
        self.idx_to_char = { i: ch for i, ch in enumerate(char_set) }

        # temporary parameters
        self.input_state = {}   # x
        self.hidden_state = {}  # h
        self.output_state = {}  # y
        self.prob_state = {}    # probabilities

        # hyper parameters
        self.hidden_size = 100   # size of hidden layer of neurons
        self.seq_length = 6      # number of steps to unroll the RNN for
        self.learning_rate = 0.1 # for Adagrad update

        # model parameters. waiting for learning.(w is mean weight)
        self.w_i2h = np.random.randn(self.hidden_size, self.charset_size)*0.01 # input to hidden
        self.w_h2h = np.random.randn(self.hidden_size, self.hidden_size)*0.01 # hidden to hidden
        self.w_h2o = np.random.randn(self.charset_size, self.hidden_size)*0.01 # hidden to output
        self.b_h = np.zeros((self.hidden_size, 1)) # hidden bias
        self.b_o = np.zeros((self.charset_size, 1)) # output bias

    def forward(self, inputs, targets, hprev):
        '''
        inputs: list of integers. the idx in char set.
        targets: list of integets.
        hprev: Hx1 array of initial hidden state
        '''
        loss = 0
        self.hidden_state[-1] = np.copy(hprev)
        for t in xrange(self.seq_length):
            # encode in 1-of-k representation
            # eg: a is [1,0,0,0] and b is [0,1,0,0]
            self.input_state[t] = np.zeros((self.charset_size, 1))
            self.input_state[t][inputs[t]] = 1
            # update hidden state
            self.hidden_state[t] = np.tanh(np.dot(self.w_i2h, self.input_state[t]) + \
                np.dot(self.w_h2h, self.hidden_state[t-1]) + self.b_h)
            # compute the output vector
            # unnormalized log probabilities for next chars
            self.output_state[t] = np.dot(self.w_h2o, self.hidden_state[t]) + self.b_o
            # probabilities for next chars
            self.prob_state[t] = np.exp(self.output_state[t]) / np.sum(np.exp(self.output_state[t]))
            # softmax (cross-entropy loss)
            loss += -np.log(self.prob_state[t][targets[t], 0])
            # backprop into y.
            self.prob_state[t][targets[t]] -= 1
        return loss

    def backward(self):
        '''
        backward pass: compute gradients going backwards
        '''
        dw_i2h, dw_h2h, dw_h2o = np.zeros_like(self.w_i2h),\
            np.zeros_like(self.w_h2h), np.zeros_like(self.w_h2o)
        db_h, db_o = np.zeros_like(self.b_h), np.zeros_like(self.b_o)
        dh_next = np.zeros_like(self.hidden_state[0])
        for t in reversed(xrange(self.seq_length)):
            do = np.copy(self.prob_state[t])
            dw_h2o += np.dot(do, self.hidden_state[t].T)
            db_o += do
            # backprop into h
            dh = np.dot(self.w_h2o.T, do) + dh_next
            # backprop through tanh nonlinearity
            dhraw = (1 - self.hidden_state[t] * self.hidden_state[t]) * dh
            db_h += dhraw
            dw_i2h += np.dot(dhraw, self.input_state[t].T)
            dw_h2h += np.dot(dhraw, self.hidden_state[t-1].T)
            dh_next = np.dot(self.w_h2h.T, dhraw)
        for dparam in [dw_i2h, dw_h2h, dw_h2o, db_h, db_o]:
            # clip to mitigate exploding gradients
            np.clip(dparam, -5, 5, out=dparam)
        return dw_i2h, dw_h2h, dw_h2o, db_h, db_o, \
            self.hidden_state[self.seq_length-1]

    def training(self):
        # iteration counter and data pointer
        _iter, _p = 0, 0
        # m is memory
        mw_h2i, mw_h2h, mw_h2o = np.zeros_like(self.w_i2h), \
            np.zeros_like(self.w_h2h), np.zeros_like(self.w_h2o)
        # memory bias variables for Adagrad
        mb_h, mb_o = np.zeros_like(self.b_h), np.zeros_like(self.b_o)
        # loss at iteration 0
        smooth_loss = -np.log(1.0/self.charset_size)*self.seq_length

        while True:
            # prepare inputs 
            # (we're sweeping from left to right in steps seq_length long)
            if _p + self.seq_length >= self.data_size or _iter == 0:
                # reset RNN memory in each epoch
                hprev = np.zeros((self.hidden_size, 1))
                _p = 0 # go from start of data
            inputs = [self.char_to_idx[ch] for ch in \
                self.train_data[_p:_p+self.seq_length]]
            targets = [self.char_to_idx[ch] for ch in \
                self.label_data[_p:_p+self.seq_length]]

            # forward seq_length characters through the net
            loss = self.forward(inputs, targets, hprev)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001
            if _iter % 100 == 0:
                print 'iter %d, loss: %f' % (_iter, smooth_loss)

            # backward fetch gradient
            dw_i2h, dw_h2h, dw_h2o, db_h, db_o, hprev = self.backward()

            # perform parameter update with Adagrad
            for param, dparam, mem in zip([self.w_i2h, self.w_h2h, self.w_h2o, self.b_h, self.b_o], 
                                            [dw_i2h, dw_h2h, dw_h2o, db_h, db_o], 
                                            [mw_h2i, mw_h2h, mw_h2o, mb_h, mb_o]):
                mem += dparam * dparam
                # adagrad update
                param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8)

            _p += self.seq_length # move data pointer
            _iter += 1            # iteration counter

            # predict in 500 iters
            if _iter % 500 == 0:
                self.predict(self.train_data)

    def predict(self, input_datas):
        '''
        predict
        '''
        h = np.zeros((self.hidden_size, 1))
        ixes = []
        for ch in input_datas:
            if ch not in self.char_to_idx:
                print "The input data include invalid char."
                return
            x = np.zeros((self.charset_size, 1))
            x[self.char_to_idx[ch]] = 1
            h = np.tanh(np.dot(self.w_i2h, x) + np.dot(self.w_h2h, h) + self.b_h)
            y = np.dot(self.w_h2o, h) + self.b_o
            p = np.exp(y) / np.sum(np.exp(y))
            ix = np.random.choice(range(self.charset_size), p=p.ravel())
            ixes.append(ix)
        res = ''.join(str(self.idx_to_char[ix]) \
            for idx, ix in enumerate(ixes) if idx%2==0)
        print 'predict output: ', res

    def save_model(self, model_name):
        with open(model_name, 'wb') as f:
            data = [self.w_i2h, self.w_h2h, self.w_h2o, self.b_h, self.b_o, 
                self.hidden_size, self.charset_size,
                self.char_to_idx, self.idx_to_char
                ]
            pickle.dump(data, f)

    def load_model(self, model_name):
        with open(model_name, 'rb') as f:
            data = pickle.load(f)
            self.w_i2h, self.w_h2h, self.w_h2o, self.b_h, self.b_o,\
            self.hidden_size, self.charset_size,\
            self.char_to_idx, self.idx_to_char = data
