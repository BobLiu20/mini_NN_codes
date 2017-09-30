
#include "tanhLayer.h"

namespace TUPU {

tanhLayer::tanhLayer(int oWidth, int oHeight, int oDepth)
{
    m_oWidth = oWidth;
    m_oHeight = oHeight;
    m_oDepth = oDepth;

}

tanhLayer::~tanhLayer()
{
}

void tanhLayer::forward(vector<double*>& bottom, vector<double*>& top)
{
    // In-Places
    assert(bottom.size() == 1);
    m_inputData = bottom[0];

    double* iter = m_inputData;
    for (int i = 0; i < m_oWidth * m_oHeight * m_oDepth; i++) {
        double ep = std::exp(*iter);
        double em = std::exp(-*iter);
        *iter = (ep - em) / (ep + em);
        iter++;
    }

    top.clear();
    top.push_back(m_inputData);
}

void tanhLayer::backward(vector<double*>& top, vector<double*>& bottom)
{
    // In-Places
    assert(top.size() == 1);
    int total = m_oWidth * m_oHeight * m_oDepth;

    for (int i = 0; i < total; i++) {
        double val = m_inputData[i];
        top[0][i] = top[0][i] * (1.0 - val * val);
    }

    bottom.clear();
    bottom.push_back(top[0]);
}

} 
