#ifndef __LAYER__H
#define __LAYER__H

#include <assert.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <numeric>
#include <random>
#include <algorithm>
#include <string>
#include <sstream>
#include <memory.h>

using namespace std;

namespace TUPU {

class layer
{
public:
    layer() {};
    ~layer() {};

    virtual void forward(vector<double*>& bottom, vector<double*>& top) = 0;
    virtual void backward(vector<double*>& top, vector<double*>& bottom) = 0;
    virtual void applyUpdate(double lr, double momentum) {};

    virtual bool getParameters(vector<pair<int, double*>>&) {return false;};
    virtual bool setParameters(vector<pair<int, double*>>) {return false;};
    virtual bool copyParameters(layer* src) {return false;};

protected:
    int getIndex(int x, int y, int channel, int width, int height, int depth) {
        assert(x >= 0 && x < width);
        assert(y >= 0 && y < height);
        assert(channel >= 0 && channel < depth);
        return (height * channel + y) * width + x;
    }
    double uniform_rand(double min, double max) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dst(min, max);
        return dst(gen);
    }
    bool uniform_rand(double* src, int len, double min, double max) {
        for (int i = 0; i < len; i++) {
            src[i] = uniform_rand(min, max);
        }
        return true;
    }
    double dotProduct(const double* s1, const double* s2, int len) {
        double result = 0.0;
        for (int i = 0; i < len; i++) {
            result += s1[i] * s2[i];
        }
        return result;
    }
    bool mulAdd(const double* src, double c, int len, double* dst)
    {
        for (int i = 0; i < len; i++) {
            dst[i] += (src[i] * c);
        }
        return true;
    }
};

}

#endif
