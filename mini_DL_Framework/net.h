#ifndef __NET_H
#define __NET_H

#include <vector>
#include <string>

using namespace std;

namespace TUPU {

class layer;

class net
{
public:
    net();
    ~net();

    void trainMnist(const char* modelPath=NULL);
    void testMnist(bool isTrain, const char* modelPath=NULL);
    void predictImages(const char* modelPath, const char* imagesFolder);

    void saveModel(vector<layer*> net, string modelPath);
    void loadModel(vector<layer*> net, string modelPath);
private:

    void createNet(vector<layer*>& net, const char* imagesFolder, 
        string imagePath, string labelPath, int total);

    vector<layer*> netTrain;
    vector<layer*> netTest;
    vector<layer*> netPredict;

};

}
#endif
