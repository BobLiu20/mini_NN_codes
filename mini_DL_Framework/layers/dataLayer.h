#ifndef __DATALAYER_H
#define __DATALAYER_H

#include "layer.h"

namespace TUPU {

class dataLayer:public layer
{
public:
    dataLayer(int oWidth, int oHeight, int oDepth,
        int totalCount, string imagePath, string labelPath);
    ~dataLayer();

    void forward(vector<double*>& bottom, vector<double*>& top);
    void backward(vector<double*>& top, vector<double*>& bottom);

private:
    double* m_imageData; // train-images.idx3-ubyte [-1, 1]
    double* m_labelData; // train-labels.idx1-ubyte -0.8/ 0.8

    // i = input, o = output
    int m_oWidth;
    int m_oHeight;
    int m_oDepth;

    string m_imagePath;
    string m_labelPath;
    int m_totalCount;

    int m_indexImage;

    int reverseInt(int i);
    void readMnistImages(string fileName, double* dataDst, int numImage);
    void readMnistLabels(string fileName, double* dataDst, int numImage);
};
}

#endif
