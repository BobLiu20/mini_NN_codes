#ifndef __TANHLAYER_H
#define __TANHLAYER_H
#include "layer.h"


namespace TUPU {

class tanhLayer:public layer
{
public:
    tanhLayer(int oWidth, int oHeight, int oDepth);
    ~tanhLayer();

    void forward(vector<double*>& bottom, vector<double*>& top);
    void backward(vector<double*>& top, vector<double*>& bottom);

private:

    double* m_inputData;

    // i = input, o = output
    int m_oWidth;
    int m_oHeight;
    int m_oDepth;
};

}
#endif
