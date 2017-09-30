#ifndef __IMAGEINPUTLAYER_H
#define __IMAGEINPUTLAYER_H

#include "layer.h"

#ifdef WITH_OPENCV
namespace cv {
    class Mat;
}
#endif

namespace TUPU {

class imageInputLayer:public layer
{
public:
    imageInputLayer(int oWidth, int oHeight, int oDepth, string folderSrc);
    ~imageInputLayer();

    void forward(vector<double*>& bottom, vector<double*>& top);
    void backward(vector<double*>& top, vector<double*>& bottom);

private:
    // i = input, o = output
    int m_oWidth;
    int m_oHeight;
    int m_oDepth;

    #ifdef WITH_OPENCV
    cv::Mat* m_currImage;
    #endif
    int m_totalCount;
    int m_indexImage;
    vector<string> m_imagesPath;

    void scanImages(string folderSrc);
};
}

#endif
