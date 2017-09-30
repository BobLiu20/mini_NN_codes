#ifndef __LOSSLAYER_H
#define __LOSSLAYER_H
#include "layer.h"


namespace TUPU {

class lossLayer:public layer
{
public:
    lossLayer(int iWidth, int iHeight, int iDepth);
    ~lossLayer();

    void forward(vector<double*>& bottom, vector<double*>& top);
    void backward(vector<double*>& top, vector<double*>& bottom);

private:

    double* m_inputData;
    double* m_inputLabel;

    double* m_deltaNeuronData;

    // i = input, o = output
    int m_iWidth;
    int m_iHeight;
    int m_iDepth;

    double m_loss;
    double m_correct;
};

}
#endif
