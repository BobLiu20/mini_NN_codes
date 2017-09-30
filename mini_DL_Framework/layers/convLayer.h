#ifndef __CONVLAYER_H
#define __CONVLAYER_H
#include "layer.h"


namespace TUPU {

class convLayer:public layer
{
public:
    convLayer(int iWidth, int iHeight, int iDepth,
        int kernelSize, int numOutput);
    ~convLayer();

    void forward(vector<double*>& bottom, vector<double*>& top);
    void backward(vector<double*>& top, vector<double*>& bottom);
    void applyUpdate(double lr, double momentum);

    bool getParameters(vector<pair<int, double*>>&);
    bool setParameters(vector<pair<int, double*>>);
    bool copyParameters(layer* src);

private:
    void layerSetup();

    double* m_inputData;

    double* m_neuronData;
    double* m_weight;
    double* m_bias;

    double* m_deltaNeuronData;
    double* m_deltaWeight;
    double* m_deltaBias;

    double* m_deltaWeightHistory;
    double* m_deltaBiasHistory;

    // i = input, o = output
    int m_iWidth;
    int m_iHeight;
    int m_iDepth;
    int m_oWidth;
    int m_oHeight;

    int m_kernelSize;
    int m_numOutput; // depth for output
    int m_stride;
    int m_pad;
};

}
#endif
