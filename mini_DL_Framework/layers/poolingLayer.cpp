 
#include "poolingLayer.h"

namespace TUPU {

poolingLayer::poolingLayer(int iWidth, int iHeight, int iDepth)
{
    m_iWidth = iWidth;
    m_iHeight = iHeight;
    m_iDepth = iDepth;

    m_oWidth = m_iWidth / 2;
    m_oHeight = m_iHeight / 2;
    m_oDepth = m_iDepth;

    m_neuronData = new double[m_oWidth * m_oHeight * m_oDepth];
    m_deltaNeuronData = new double[m_iWidth * m_iHeight * m_iDepth];
    m_mask = new int[m_oWidth * m_oHeight * m_oDepth];
}

poolingLayer::~poolingLayer()
{
    delete[] m_neuronData;
    delete[] m_deltaNeuronData;
    delete[] m_mask;
}

void poolingLayer::forward(vector<double*>& bottom, vector<double*>& top)
{
    assert(bottom.size() == 1);
    m_inputData = bottom[0];
    memset(m_mask, -1, sizeof(int) * m_oWidth * m_oHeight * m_oDepth);

    // 2x2 MAX Pooling
    int kernelSize = 2;
    for (int c = 0; c < m_oDepth; ++c) {
        for (int ph = 0; ph < m_oHeight; ++ph) {
            for (int pw = 0; pw < m_oWidth; ++pw) {
                int hstart = ph * 2;// TODO: add stride and pad here
                int wstart = pw * 2;// TODO: add stride and pad here
                int hend = min(hstart + kernelSize, m_iHeight);
                int wend = min(wstart + kernelSize, m_iWidth);
                // hstart = max(hstart, 0);
                // wstart = max(wstart, 0);
                const int poolIndex = c * m_oWidth * m_oHeight + ph * m_oWidth + pw;
                m_neuronData[poolIndex] = -3.402823466e+38F;
                for (int h = hstart; h < hend; ++h) {
                    for (int w = wstart; w < wend; ++w) {
                        const int index = c * m_iHeight * m_iWidth + h * m_iWidth + w;
                        if (m_inputData[index] > m_neuronData[poolIndex]) {
                            m_neuronData[poolIndex] = m_inputData[index];
                            // for backward
                            m_mask[poolIndex] = index;
                        }
                    }
                }
            }
        }
    }
    top.clear();
    top.push_back(m_neuronData);
}

void poolingLayer::backward(vector<double*>& top, vector<double*>& bottom)
{
    memset(m_deltaNeuronData, 0.0,
        sizeof(m_deltaNeuronData) * m_iWidth * m_iHeight * m_iDepth);

    for (int c = 0; c < m_oDepth; ++c) {
        for (int ph = 0; ph < m_oHeight; ++ph) {
            for (int pw = 0; pw < m_oWidth; ++pw) {
                const int index = c * m_oWidth * m_oHeight + ph * m_oWidth + pw;
                const int bottom_index = m_mask[index];
                m_deltaNeuronData[bottom_index] += top[0][index];
            }
        }
    }

    bottom.clear();
    bottom.push_back(m_deltaNeuronData);
}

}
