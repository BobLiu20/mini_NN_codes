#include <iostream>
#include <cstring>
#include "net.h"

using namespace std;

void printUsage();

int main(int argc, char const *argv[])
{
    if (argc <= 1) {
        printUsage();
        return 0;
    }
    if (strcmp(argv[1], "train") == 0 and (argc == 2 || argc == 3)) {
        TUPU::net netTrain;
        if (argc == 2) {
            netTrain.trainMnist();          // Train from zero
        } else {
            netTrain.trainMnist(argv[2]);   // Load Pre-Train Model
        }
    } else if (strcmp(argv[1], "test") == 0 and argc == 3) {
        TUPU::net netTest;
        netTest.testMnist(false, argv[2]);
    } else if (strcmp(argv[1], "predict") == 0 and argc == 4) {
        #ifdef WITH_OPENCV  // Request OpenCV to read external images
        TUPU::net netPredict;
        netPredict.predictImages(argv[2], argv[3]);
        #else
        cout << "Predict feature request to build with OpenCV." << endl;
        cout << "1. Note that just using OpenCV to read image." << endl;
        cout << "2. Open MakeFile and set OPENCV_ENABLE=1." << endl;
        cout << "3. make clean && make" << endl;
        #endif
    } else {
        printUsage();
    }
    return 0;
}

void printUsage()
{
    cout << endl;
    cout << "Usage:(Type it in command line.)" << endl;
    cout << endl;
    cout << "* train   :  ./main  train" << endl;
    cout << "* test    :  ./main  test  iter_xxx.model" << endl;
    cout << "* predict :  ./main  predict  iter_xxx.model  images_folder" << endl;
}
