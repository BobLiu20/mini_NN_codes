#ifndef __FCLAYER_H
#define __FCLAYER_H
#include "layer.h"


namespace TUPU {

class fcLayer:public layer
{
public:
    fcLayer(int iWidth, int iHeight, int iDepth, int numOutput);
    ~fcLayer();

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
    int m_numOutput;
};

}
#endif
