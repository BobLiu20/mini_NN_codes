#!/usr/bin/env python2.7

from tupu_rnn import tupu_rnn

if __name__ == '__main__':
    rnn = tupu_rnn()
    rnn.load_model('test.model')
    rnn.predict([4, 7, 4, 7, 1, 1])
