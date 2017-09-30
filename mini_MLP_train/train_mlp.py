import math

_LABEL = 0 # class = 10
_IMAGE = 1
_FC64_W = 2 #50176
_FC64_B = 3 #64
_FC128_W = 4#8192
_FC128_B = 5#128
_FC10_W = 6 #1280
_FC10_B = 7 #10

def readInput():
    dataList = []
    with open('bp_sample_input.txt') as f:
        for line in f:
            dataList.append(map(float, line.split()))
    dataList[_IMAGE] = [dataList[_IMAGE][i:i+28*28] for i in range(0, len(dataList[_IMAGE]), 28*28)]
    dataList[_LABEL] = [dataList[_LABEL][i:i+10] for i in range(0, len(dataList[_LABEL]), 10)]
    return dataList

def __fcForward(inputBatch, outBatch, weight, bias, inputSize, outSize):
    outBatch[:] = []
    for datas in inputBatch:
        outBatch.append([])
        for i in range(outSize):
            val = 0.0
            for d, data in enumerate(datas):
                # val += data * weight[i*inputSize+d]
                val += data * weight[d*outSize+i]
            val += bias[i]
            val = max(val, 0.0) # relu
            outBatch[-1].append(val)

def __softmaxForward(inputBatch, outBatch):
    outBatch[:] = []
    for batch in inputBatch:
        outBatch.append([])
        shift = max(batch)
        exp_x = [math.exp(y-shift) for y in batch]
        sum_x = sum(exp_x)
        for i, y in enumerate(exp_x):
            outBatch[-1].append(y / sum_x)

def __crossEntropy(inputBatch, labelBatch):
    acc = 0
    for i, l in zip(inputBatch, labelBatch):
        if i.index(max(i)) == l.index(1.0):
            acc += 1
    acc = acc / float(len(inputBatch))
    loss = 0
    for data, label in zip(inputBatch, labelBatch):
        # -sum([p * math.log(q) for q, p in zip(data, label)])
        loss += -math.log(data[label.index(1.0)])
    loss /= len(labelBatch)
    return loss, acc

def __crossEntropyBackward(deltaBatch, inputBatch, labelBatch):
    deltaBatch[:] = []
    for idx, (data, label) in enumerate(zip(inputBatch, labelBatch)):
        deltaBatch.append([])
        cl = label.index(1.0)
        for i, d in enumerate(data):
            deltaBatch[-1].append(-1.0/data[cl])

def __softmaxBackward(deltaBatch, deltaPre, inputBatch, outBatch, labelBatch):
    deltaBatch[:] = []
    for idx, (_in, _out, label) in enumerate(zip(inputBatch, outBatch, labelBatch)):
        deltaBatch.append([])
        cl = label.index(1.0)
        for i, (d, o) in enumerate(zip(_in, _out)):
            if i == cl:
                deltaBatch[-1].append(o * (1.0 - o) * deltaPre[idx][i])
            else:
                deltaBatch[-1].append(- _out[cl] * o * deltaPre[idx][i])

def __fcBackward(deltaBatch, deltaPre, inputBatch, outBatch, weight, bias, lr):
    # input delta
    deltaBatch[:] = []
    inputSize = len(inputBatch[0])
    outSize = len(outBatch[0])
    for idx, data in enumerate(inputBatch):
        deltaBatch.append([])
        for i, d in enumerate(data):
            sumVal = 0.0
            for x in range(len(outBatch[0])):
                if outBatch[idx][x] <= 0.0:
                    deltaPre[idx][x] = 0.0
                sumVal += (deltaPre[idx][x] * weight[i*outSize+x])#weight[x*inputSize+i])
            deltaBatch[-1].append(sumVal)
    # update weight
    for i in range(inputSize):
        for j in range(outSize):
            sumVal = 0.0
            for idx in range(len(inputBatch)):
                sumVal += (deltaPre[idx][j] * inputBatch[idx][i])
            sumVal /= float(len(inputBatch))
            #weight[j*inputSize+i] -= (lr * sumVal)
            weight[i*outSize+j] -= (lr * sumVal)
    # update bias
    for j in range(outSize):
        sumVal = 0.0
        for idx in range(len(inputBatch)):
            sumVal += (deltaPre[idx][j])
        sumVal /= float(len(inputBatch))
        bias[j] -= (lr * sumVal)

def trainStep(dataBatch, labelBatch, dataList, lr):
    outFc64 = []
    __fcForward(dataBatch, outFc64, dataList[_FC64_W], dataList[_FC64_B], 28*28, 64)
    outFc128 = []
    __fcForward(outFc64, outFc128, dataList[_FC128_W], dataList[_FC128_B], 64, 128)
    outFc10 = []
    __fcForward(outFc128, outFc10, dataList[_FC10_W], dataList[_FC10_B], 128, 10)
    outSoftmax = []
    __softmaxForward(outFc10, outSoftmax)
    loss, acc = __crossEntropy(outSoftmax, labelBatch)
    deltaCross = []
    __crossEntropyBackward(deltaCross, outSoftmax, labelBatch)
    deltaSoftmax = []
    __softmaxBackward(deltaSoftmax, deltaCross, outFc10, outSoftmax, labelBatch)
    deltaFc10 = []
    __fcBackward(deltaFc10, deltaSoftmax, outFc128, outFc10, 
        dataList[_FC10_W], dataList[_FC10_B], lr)
    deltaFc128 = []
    __fcBackward(deltaFc128, deltaFc10, outFc64, outFc128,
        dataList[_FC128_W], dataList[_FC128_B], lr)
    deltaFc64 = []
    __fcBackward(deltaFc64, deltaFc128, dataBatch, outFc64,
        dataList[_FC64_W], dataList[_FC64_B], lr)
    return loss, acc

if __name__ == "__main__":
    batchSize = 64
    lr = 0.1
    epochNum = 50
    iterCnt = 0
    dataList = readInput()
    for epoch in range(epochNum):
        for idx in range(0, len(dataList[_LABEL]), batchSize):
            # do bp
            loss, acc = trainStep(dataList[_IMAGE][idx: idx+batchSize],
                dataList[_LABEL][idx: idx+batchSize], dataList, lr)
            iterCnt += 1
            print "epoch:%d iter:%d loss:%f acc:%f"%(epoch, iterCnt, loss, acc)
