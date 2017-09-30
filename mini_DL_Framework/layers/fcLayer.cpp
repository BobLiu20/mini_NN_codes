
#include "fcLayer.h"

namespace TUPU {

fcLayer::fcLayer(int iWidth, int iHeight, int iDepth, int numOutput)
{
    m_iWidth = iWidth;
    m_iHeight = iHeight;
    m_iDepth = iDepth;

    m_numOutput = numOutput;

    m_neuronData = new double[m_numOutput];
    m_weight = new double[m_iWidth * m_iHeight * m_iDepth * m_numOutput];
    m_bias = new double[m_numOutput];

    m_deltaNeuronData = new double[m_iWidth * m_iHeight * m_iDepth];
    m_deltaWeight = new double[m_iWidth * m_iHeight * m_iDepth * m_numOutput];
    m_deltaBias = new double[m_numOutput];

    m_deltaWeightHistory = new double[
        m_iWidth * m_iHeight * m_iDepth * m_numOutput];
    m_deltaBiasHistory = new double[m_numOutput];
    layerSetup();
}


fcLayer::~fcLayer()
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

void fcLayer::forward(vector<double*>& bottom, vector<double*>& top)
{
    assert(bottom.size() == 1);
    m_inputData = bottom[0];
    memset(m_neuronData, 0.0, sizeof(double) * m_numOutput);
    for (int i = 0; i < m_numOutput; i++) {
        m_neuronData[i] = 0.0;
        for (int c = 0; c < m_iWidth * m_iHeight * m_iDepth; c++) {
            m_neuronData[i] += m_weight[c * m_numOutput + i] * m_inputData[c];
        }
        m_neuronData[i] += m_bias[i];
    }
    top.clear();
    top.push_back(m_neuronData);
}

void fcLayer::backward(vector<double*>& top, vector<double*>& bottom)
{
    assert(top.size() == 1);
    memset(m_deltaNeuronData, 0.0, 
        sizeof(double) * m_iWidth * m_iHeight * m_iDepth);
    memset(m_deltaWeight, 0.0, 
        sizeof(double) * m_iWidth * m_iHeight * m_iDepth * m_numOutput);
    memset(m_deltaBias, 0.0, 
        sizeof(double) * m_numOutput);

    for (int i = 0; i < m_iWidth * m_iHeight * m_iDepth; i++) {
        // input delta
        m_deltaNeuronData[i] = dotProduct(
            top[0], &m_weight[i * m_numOutput], m_numOutput);
        // weight delta
        mulAdd(top[0], m_inputData[i], m_numOutput, m_deltaWeight + i * m_numOutput);
    }
    // bias delta
    for (int i = 0; i < m_numOutput; i++) {
        m_deltaBias[i] = top[0][i];
    }

    bottom.clear();
    bottom.push_back(m_deltaNeuronData);
}

void fcLayer::applyUpdate(double lr, double momentum)
{
    for (int i = 0; i < m_iWidth * m_iHeight * m_iDepth * m_numOutput; i++) {
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

bool fcLayer::getParameters(vector<pair<int, double*>>& param)
{
    param.push_back(
        make_pair(m_iWidth * m_iHeight * m_iDepth * m_numOutput, m_weight));
    param.push_back(make_pair(m_numOutput, m_bias));
    return true;
}

bool fcLayer::setParameters(vector<pair<int, double*>> param)
{
    assert(param.size() == 2);
    memcpy(m_weight, param[0].second, sizeof(double) * param[0].first);
    memcpy(m_bias, param[1].second, sizeof(double) * param[1].first);
    return true;
}

bool fcLayer::copyParameters(layer* src)
{
    vector<pair<int, double*>> param;
    assert(src->getParameters(param) == true);
    setParameters(param);
    return true;
}

void fcLayer::layerSetup()
{
    // init weight and bias
    srand(time(0) + rand());
    const double scale = 6.0;
    double min_ = -std::sqrt(scale / (120.0 + 10.0));
    double max_ = std::sqrt(scale / (120.0 + 10.0));
    uniform_rand(m_weight, m_iWidth * m_iHeight * m_iDepth * m_numOutput , min_, max_);
    memset(m_bias, 0.0, sizeof(double) * m_numOutput);

    memset(m_deltaWeightHistory, 0.0,
        sizeof(double) * m_iWidth * m_iHeight * m_iDepth * m_numOutput);
    memset(m_deltaBiasHistory, 0.0,
        sizeof(double) * m_numOutput);
}

}
