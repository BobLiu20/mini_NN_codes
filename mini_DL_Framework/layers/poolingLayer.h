#ifndef __POOLINGLAYER_H
#define __POOLINGLAYER_H
#include "layer.h"


namespace TUPU {

class poolingLayer:public layer
{
public:
    poolingLayer(int iWidth, int iHeight, int iDepth);
    ~poolingLayer();

    void forward(vector<double*>& bottom, vector<double*>& top);
    void backward(vector<double*>& top, vector<double*>& bottom);

private:
    double* m_inputData;

    double* m_neuronData;
    double* m_deltaNeuronData;
    int* m_mask; // max val index for max pooling

    // i = input, o = output
    int m_iWidth;
    int m_iHeight;
    int m_iDepth;
    int m_oWidth;
    int m_oHeight;
    int m_oDepth;
};

}
#endif
