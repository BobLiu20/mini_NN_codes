
#include "lossLayer.h"

namespace TUPU {

lossLayer::lossLayer(int iWidth, int iHeight, int iDepth)
{
    m_iWidth = iWidth;
    m_iHeight = iHeight;
    m_iDepth = iDepth;

    m_deltaNeuronData = new double[m_iWidth * m_iHeight * m_iDepth];
}

lossLayer::~lossLayer()
{
    delete[] m_deltaNeuronData;
}

void lossLayer::forward(vector<double*>& bottom, vector<double*>& top)
{
    assert(bottom.size() == 2);
    m_inputData = bottom[0];
    m_inputLabel = bottom[1];
    // Euclidean distance: y = 1/2 * (a - b)^2
    double label = m_inputLabel[0];
    double loss = 0.0;
    double maxVal = -3.402823466e+38F;
    int maxIdx = -1;
    for (int i = 0; i < m_iWidth * m_iHeight * m_iDepth; i++) {
        double val = (i == label ? 0.8: -0.8);
        val = m_inputData[i] - val;
        loss += (val * val / 2.0);
        // accuracy
        if (maxVal < m_inputData[i]) {
            maxVal = m_inputData[i];
            maxIdx = i;
        }
    }
    m_loss = loss / (m_iWidth * m_iHeight * m_iDepth);
    m_correct = (maxIdx == label ? 1.0: 0.0);

    top.clear();
    top.push_back(&m_loss);
    top.push_back(&m_correct);
}

void lossLayer::backward(vector<double*>& top, vector<double*>& bottom)
{
    int total = m_iWidth * m_iHeight * m_iDepth;
    memset(m_deltaNeuronData, 0.0, 
        sizeof(m_deltaNeuronData) * total);
    double label = m_inputLabel[0];
    for (int i = 0; i < total; i++) {
        double val = (i == label ? 0.8: -0.8);
        m_deltaNeuronData[i] = m_inputData[i] - val;
    }
    bottom.clear();
    bottom.push_back(m_deltaNeuronData);
}
}
