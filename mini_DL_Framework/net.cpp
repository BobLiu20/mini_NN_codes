
#include "net.h"
#include "layer.h"
#include "convLayer.h"
#include "dataLayer.h"
#include "tanhLayer.h"
#include "poolingLayer.h"
#include "fcLayer.h"
#include "lossLayer.h"
#include "imageInputLayer.h"

namespace TUPU {

net::net()
{
}

net::~net()
{
    for (auto& iter : netTrain) {
        delete iter;
    }
    for (auto& iter : netTest) {
        delete iter;
    }
    for (auto& iter : netPredict) {
        delete iter;
    }
}

void net::trainMnist(const char* modelPath)
{
    // Hyper parameter
    int epochIter = 60000;          // mnist has 60000 for training
    int maxIter = epochIter * 10 + 1;// Max iter to train
    int displayInterval = 1000;     // Display loss info
    int testInterval = epochIter;   // Testing interval
    int saveInterval = epochIter;   // Auto save model file.
    double lr = 0.001;             // learning rate
    double momentum = 0.9;          // momentum

    // Create training Net
    createNet(netTrain, NULL,
        "datas/train-images-idx3-ubyte",
        "datas/train-labels-idx1-ubyte", 60000);

    // if need to load pre-train model
    if (modelPath) {
        loadModel(netTrain, modelPath);
    }

    // Main loop for training and testing
    double accuracyTrain = 0;
    double averageLoss = 1.0;
    for (int iter = 0; iter < maxIter; iter++) {
        // 1. Feed Forward
        vector<double*> fBottomData, fTopData;
        auto* dataL = (*netTrain.begin());    // Data Layer
        dataL->forward(fBottomData, fTopData);
        vector<double*> fBottomNormal, fTopNormal;
        fBottomNormal.push_back(fTopData[0]); // image data
        for (auto it = (netTrain.begin() + 1); it != (netTrain.end() - 1); ++it)
        {
            (*it)->forward(fBottomNormal, fTopNormal);// Other Layer
            fTopNormal.swap(fBottomNormal);
        }
        vector<double*> fBottomLoss, fTopLoss;
        fBottomLoss.push_back(fBottomNormal[0]); // bottom 0: prev out
        fBottomLoss.push_back(fTopData[1]);      // bottom 1: image label
        auto* lossL = (*(netTrain.end() - 1));   // Loss Layer
        lossL->forward(fBottomLoss, fTopLoss);

        // 2. Loss and Accuracy
        averageLoss = 0.1 * (*fTopLoss[0]) + 0.9 * averageLoss;
        if (iter % displayInterval == 0) {
            cout << "Iter " << iter << "\tLoss = " << averageLoss << endl;
        }
        accuracyTrain += (*fTopLoss[1]);
        if (iter % epochIter == 0 && iter != 0) {
            cout << "Epoch " << (iter / epochIter - 1) << " Done......" << endl;
            cout << "Training Accuracy = " << (accuracyTrain/double(epochIter)) << endl;
            accuracyTrain = 0;
        }

        // 3. Back propagation
        vector<double*> bTop, bBottom;
        for (auto it = netTrain.rbegin(); it != netTrain.rend(); ++it)
        {
            (*it)->backward(bTop, bBottom);
            bTop.swap(bBottom);
        }

        // 4. Update Weight and Bias
        for (auto& iter : netTrain) {
            iter->applyUpdate(lr, momentum);// lr and momentum
        }

        // 5. Testing
        if (iter % testInterval == 0 && iter != 0) {
            testMnist(true);
        }

        // 6. Auto save model
        if (iter % saveInterval == 0 && iter != 0) {
            // Auto save model
            stringstream ss;
            ss << "iter_" << iter << ".model";
            saveModel(netTrain, ss.str());
        }
    }
}

void net::testMnist(bool isTrain, const char* modelPath)
{
    // Create testing Net
    if (netTest.size() == 0) {
        createNet(netTest, NULL,
            "datas/t10k-images-idx3-ubyte",
            "datas/t10k-labels-idx1-ubyte", 10000);
    }

    // Confirm call from training or external
    if (isTrain) {
        assert(netTrain.size() == netTest.size());
        // Training Mode. Sync Weight and Bias.
        for (int i = 0; i < netTrain.size(); i++) {
            netTest[i]->copyParameters(netTrain[i]);
        }
    } else {
        // Only Testing Mode. Load model file
        loadModel(netTest, modelPath);
    }

    // Main loop for Testing
    cout << "Testing running......" << endl;
    int maxIter = 10000; // mnist has 10000 for testing
    double accuracyTest = 0;
    for (int iter = 0; iter < 10000; iter++) {
        // 1. Feed Forward
        vector<double*> fBottomData, fTopData;
        auto* dataL = (*netTest.begin()); // Data Layer
        dataL->forward(fBottomData, fTopData);
        vector<double*> fBottomNormal, fTopNormal;
        fBottomNormal.push_back(fTopData[0]); // image data
        for (auto it = (netTest.begin() + 1); it != (netTest.end() - 1); ++it)
        {
            (*it)->forward(fBottomNormal, fTopNormal);
            fTopNormal.swap(fBottomNormal);
        }
        vector<double*> fBottomLoss, fTopLoss;
        fBottomLoss.push_back(fBottomNormal[0]); // bottom 0: prev out
        fBottomLoss.push_back(fTopData[1]);      // bottom 1: image label
        auto* lossL = (*(netTest.end() - 1));   // Loss Layer
        lossL->forward(fBottomLoss, fTopLoss);
        // 2. Accuracy
        accuracyTest += (*fTopLoss[1]);
    }
    cout << "Testing Done......" << endl;
    cout << "Testing Accuracy = " << (accuracyTest/double(maxIter)) << endl;
    cout << endl;
}

void net::predictImages(const char* modelPath, const char* imagesFolder)
{
    // Create predict net.
    createNet(netPredict, imagesFolder, "", "", -1);
    // Load model
    loadModel(netPredict, modelPath);
    // Main loop for Predict
    cout << "Predict running......" << endl;
    while(true) {
        // 1. Feed Forward
        vector<double*> fBottomData, fTopData;
        auto* dataL = (*netPredict.begin()); // Data Layer
        dataL->forward(fBottomData, fTopData);
        vector<double*> fBottomNormal, fTopNormal;
        fBottomNormal.push_back(fTopData[0]); // image data
        if (*(int*)fTopData[1] == -1) {
            break;
        }
        for (auto it = (netPredict.begin() + 1); it != (netPredict.end() - 1); ++it)
        {
            (*it)->forward(fBottomNormal, fTopNormal);
            fTopNormal.swap(fBottomNormal);
        }
        // 2. Cal predict
        double maxVal = -3.402823466e+38F;
        int maxIdx = -1;
        for (int i = 0; i < 10; i++) {
            if (fBottomNormal[0][i] > maxVal) {
                maxVal = fBottomNormal[0][i];
                maxIdx = i;
            }
        }
        cout << "\tPredict Numer is " << maxIdx << endl;
    }
    cout << "Predict Done......" << endl;
}

void net::createNet(vector<layer*>& net,
    const char* imagesFolder, string imagePath, string labelPath, int total)
{
    // if imagesFolder is NULL : mnist data input to training or testing
    // if imagesFolder not NULL: external images input to predict
    if (!imagesFolder) {
        // Input: mnist       Output: 1x32x32
        net.push_back(new dataLayer(32, 32, 1, total, imagePath, labelPath));
    } else {
        // Input: images      Output: 1x32x32
        net.push_back(new imageInputLayer(32, 32, 1, imagesFolder));
    }
    // Input: 1x32x32     Output: 6x28x28
    net.push_back(new convLayer(32, 32, 1, 5, 6));
    // Input: 6x28x28     Output: 6x28x28
    net.push_back(new tanhLayer(28, 28, 6));
    // Input: 6x28x28     Output: 6x14x14
    net.push_back(new poolingLayer(28, 28, 6));
    // Input: 6x14x14     Output: 16x10x10
    net.push_back(new convLayer(14, 14, 6, 5, 16));
    // Input: 16x10x10    Output: 16x10x10
    net.push_back(new tanhLayer(10, 10, 16));
    // Input: 16x10x10    Output: 16x5x5
    net.push_back(new poolingLayer(10, 10, 16));
    // Input: 16x5x5      Output: 120x1x1
    net.push_back(new convLayer(5, 5, 16, 5, 120));
    // Input: 120x1x1     Output: 120x1x1
    net.push_back(new tanhLayer(1, 1, 120));
    // Input: 120x1x1     Output: 10x1x1
    net.push_back(new fcLayer(1, 1, 120, 10));
    // Input: 10x1x1      Output: 10x1x1
    net.push_back(new tanhLayer(1, 1, 10));
    // Input: 10x1x1      Output: 1
    if (!imagesFolder) {
        net.push_back(new lossLayer(1, 1, 10));
    }
}

void net::saveModel(vector<layer*> net, string modelPath)
{
    FILE* fp = fopen(modelPath.c_str(), "wb");
    if (fp == NULL) {
        cout << "Error: Can not create model file!" << endl;
        return ;
    }
    for (auto& iter : net) {
        vector<pair<int, double*>> param;
        if (iter->getParameters(param)) {
            for (auto& p : param) {
                fwrite(p.second, sizeof(double), p.first, fp);
            }
        }
    }
    fflush(fp);
    fclose(fp);
    cout << "Saving Model File to: " << modelPath << endl << endl;
}

void net::loadModel(vector<layer*> net, string modelPath)
{
    FILE* fp = fopen(modelPath.c_str(), "rb");
    if (fp == NULL) {
        cout << "Error: Can not read model file!" << endl;
        return ;
    }
    for (auto& iter : net) {
        vector<pair<int, double*>> param;
        if (iter->getParameters(param)) {
            for (auto& p : param) {
                fread(p.second, sizeof(double), p.first, fp);
            }
        }
    }
    fflush(fp);
    fclose(fp);
    cout << "Loaded Model File From: " << modelPath << endl << endl;
}

}
