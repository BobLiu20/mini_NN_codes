
#include "convLayer.h"

namespace TUPU {

convLayer::convLayer(int iWidth, int iHeight, int iDepth,
    int kernelSize, int numOutput)
{
    m_iWidth = iWidth;
    m_iHeight = iHeight;
    m_iDepth = iDepth;

    m_kernelSize = kernelSize;
    m_numOutput = numOutput;
    m_stride = 1;
    m_pad = 0;

    m_oWidth = ((iWidth + 2 * m_pad) - kernelSize + 1) / m_stride;
    m_oHeight = ((iHeight + 2 * m_pad) - kernelSize + 1) / m_stride;

    m_neuronData = new double[m_oWidth * m_oHeight * m_numOutput];
    m_weight = new double[m_kernelSize * m_kernelSize * m_iDepth * m_numOutput];
    m_bias = new double[m_numOutput];

    m_deltaNeuronData = new double[m_iWidth * m_iHeight * m_iDepth];
    m_deltaWeight = new double[m_kernelSize * m_kernelSize * m_iDepth * m_numOutput];
    m_deltaBias = new double[m_numOutput];

    m_deltaWeightHistory = new double[
        m_kernelSize * m_kernelSize * m_iDepth * m_numOutput];
    m_deltaBiasHistory = new double[m_numOutput];

    layerSetup();
}

convLayer::~convLayer()
{
    delete[] m_neuronData;
    delete[] m_weight;
    delete[] m_bias;
    delete[] m_deltaNeuronData;
    delete[] m_deltaWeight;
    delete[] m_deltaBias;
    delete[] m_deltaWeightHistory;
    delete[] m_deltaBiasHistory;
}

void convLayer::forward(vector<double*>& bottom, vector<double*>& top)
{
    assert(bottom.size() == 1);
    m_inputData = bottom[0];
    memset(m_neuronData, 0.0, 
        sizeof(double) * m_oWidth * m_oHeight * m_numOutput);

    for (int o = 0; o < m_numOutput; o++) {
        for (int i = 0; i < m_iDepth; i++) {
            int idx1 = getIndex(0, 0, m_iDepth * o + i, m_kernelSize, m_kernelSize,
                m_numOutput * m_iDepth);
            int idx2 = getIndex(0, 0, i, m_iWidth, m_iHeight, m_iDepth);
            int idx3 = getIndex(0, 0, o, m_oWidth, m_oHeight, m_numOutput);

            const double* pw = m_weight + idx1;
            const double* pi = m_inputData + idx2;
            double* pa = m_neuronData + idx3;

            for (int y = 0; y < m_oHeight; y++) {
                for (int x = 0; x < m_oWidth; x++) {
                    const double* ppw = pw;
                    const double* ppi = pi + y * m_iWidth + x;
                    double sum = 0.0;

                    for (int wy = 0; wy < m_kernelSize; wy++) {
                        for (int wx = 0; wx < m_kernelSize; wx++) {
                            sum += *ppw++ * ppi[wy * m_iWidth + wx];
                        }
                    }

                    pa[y * m_oWidth + x] += sum;
                }
            }
        }

        int idx4 = getIndex(0, 0, o, m_oWidth, m_oHeight, m_numOutput);
        double* pa = m_neuronData + idx4;
        double b = m_bias[o];
        for (int y = 0; y < m_oHeight; y++) {
            for (int x = 0; x < m_oWidth; x++) {
                pa[y * m_oWidth + x] += b;
            }
        }
    }

    top.clear();
    top.push_back(m_neuronData);
}

void convLayer::backward(vector<double*>& top, vector<double*>& bottom)
{
    assert(top.size() == 1);
    memset(m_deltaNeuronData, 0.0, 
        sizeof(double) * m_iWidth * m_iHeight * m_iDepth);
    memset(m_deltaWeight, 0.0, 
        sizeof(double) * m_kernelSize * m_kernelSize * m_iDepth * m_numOutput);
    memset(m_deltaBias, 0.0, 
        sizeof(double) * m_numOutput);

    // input delta
    for (int inc = 0; inc < m_iDepth; inc++) {
        for (int outc = 0; outc < m_numOutput; outc++) {
            int idx1 = getIndex(0, 0, m_iDepth * outc + inc, 
                m_kernelSize, m_kernelSize, m_iDepth * m_numOutput);
            int idx2 = getIndex(0, 0, outc, m_oWidth, m_oHeight, m_numOutput);
            int idx3 = getIndex(0, 0, inc, m_iWidth, m_iHeight, m_iDepth);

            const double* pw = m_weight + idx1;
            const double* pdeltaSrc = top[0] + idx2;
            double* pdeltaDst = m_deltaNeuronData + idx3;

            for (int y = 0; y < m_oHeight; y++) {
                for (int x = 0; x < m_oWidth; x++) {
                    const double* ppw = pw;
                    const double ppdeltaSrc = pdeltaSrc[y * m_oWidth + x];
                    double* ppdeltaDst = pdeltaDst + y * m_iWidth + x;

                    for (int wy = 0; wy < m_kernelSize; wy++) {
                        for (int wx = 0; wx < m_kernelSize; wx++) {
                            ppdeltaDst[wy * m_iWidth + wx] += *ppw++ * ppdeltaSrc;
                        }
                    }
                }
            }
        }
    }

    // weight delta
    for (int inc = 0; inc < m_iDepth; inc++) {
        for (int outc = 0; outc < m_numOutput; outc++) {
            for (int wy = 0; wy < m_kernelSize; wy++) {
                for (int wx = 0; wx < m_kernelSize; wx++) {
                    int idx1 = getIndex(wx, wy, inc, m_iWidth, m_iHeight, m_iDepth);
                    int idx2 = getIndex(0, 0, outc, m_oWidth, m_oHeight, m_numOutput);
                    int idx3 = getIndex(wx, wy, m_iDepth * outc + inc, 
                        m_kernelSize, m_kernelSize, m_iDepth * m_numOutput);

                    double dst = 0.0;
                    const double* prevo = m_inputData + idx1;
                    const double* delta = top[0] + idx2;

                    for (int y = 0; y < m_oHeight; y++) {
                        dst += dotProduct(prevo + y * m_iWidth,
                            delta + y * m_oWidth, m_oWidth);
                    }

                    m_deltaWeight[idx3] += dst;
                }
            }
        }
    }

    // bias delta
    for (int outc = 0; outc < m_numOutput; outc++) {
        int idx2 = getIndex(0, 0, outc, m_oWidth, m_oHeight, m_numOutput);
        const double* delta = top[0] + idx2;

        for (int y = 0; y < m_oHeight; y++) {
            for (int x = 0; x < m_oWidth; x++) {
                m_deltaBias[outc] += delta[y * m_oWidth + x];
            }
        }
    }

    bottom.clear();
    bottom.push_back(m_deltaNeuronData);
}

void convLayer::applyUpdate(double lr, double momentum)
{
    for (int i = 0; i < m_kernelSize * m_kernelSize * m_iDepth * m_numOutput; i++) {
        m_deltaWeightHistory[i] = \
            lr * m_deltaWeight[i] + momentum * m_deltaWeightHistory[i];
        m_weight[i] -= m_deltaWeightHistory[i];
    }

    for (int i = 0; i < m_numOutput; i++) {
        m_deltaBiasHistory[i] = \
            lr * m_deltaBias[i] + momentum * m_deltaBiasHistory[i];
        m_bias[i] -= m_deltaBiasHistory[i];
    }
}

bool convLayer::getParameters(vector<pair<int, double*>>& param)
{
    param.push_back(
        make_pair(m_kernelSize * m_kernelSize * m_iDepth * m_numOutput, m_weight));
    param.push_back(make_pair(m_numOutput, m_bias));
    return true;
}

bool convLayer::setParameters(vector<pair<int, double*>> param)
{
    assert(param.size() == 2);
    memcpy(m_weight, param[0].second, sizeof(double) * param[0].first);
    memcpy(m_bias, param[1].second, sizeof(double) * param[1].first);
    return true;
}

bool convLayer::copyParameters(layer* src)
{
    vector<pair<int, double*>> param;
    assert(src->getParameters(param) == true);
    setParameters(param);
    return true;
}

void convLayer::layerSetup()
{
    // init weight and bias
    srand(time(0) + rand());
    const double scale = 6.0;
    double min_ = -std::sqrt(scale / (25.0 + 150.0));
    double max_ = std::sqrt(scale / (25.0 + 150.0));
    uniform_rand(m_weight, m_kernelSize * m_kernelSize * m_iDepth * m_numOutput
        , min_, max_);
    memset(m_bias, 0.0, sizeof(double) * m_numOutput);

    memset(m_deltaWeightHistory, 0.0,
        sizeof(double) * m_kernelSize * m_kernelSize * m_iDepth * m_numOutput);
    memset(m_deltaBiasHistory, 0.0,
        sizeof(double) * m_numOutput);
}

}
