#!/usr/bin/env python2.7

import signal
from tupu_rnn import tupu_rnn

def sigint_handler(signum, frame):
    global rnn
    if rnn:
        print '\nsaving test model to test.model...'
        rnn.save_model('test.model')
    exit()

rnn = None

if __name__ == '__main__':
    # save model when ctrl+C or kill me.
    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGHUP, sigint_handler)
    signal.signal(signal.SIGTERM, sigint_handler)

    training_data = [4, 7, 4, 7, 1, 1]
    label_data = [1, 2, 3]

    rnn = tupu_rnn(training_data, label_data)
    rnn.training()
