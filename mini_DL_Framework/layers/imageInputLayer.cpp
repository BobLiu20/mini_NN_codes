
#include "imageInputLayer.h"
#include <dirent.h>

#ifdef WITH_OPENCV
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace cv;
#endif

namespace TUPU {

imageInputLayer::imageInputLayer(int oWidth, int oHeight, int oDepth, string folderSrc)
{
    m_oWidth = oWidth;
    m_oHeight = oHeight;
    m_oDepth = oDepth;

    m_indexImage = -1;

    #ifdef WITH_OPENCV
    m_currImage = new Mat();
    #endif

    scanImages(folderSrc);
}

imageInputLayer::~imageInputLayer()
{
    #ifdef WITH_OPENCV
    delete m_currImage;
    #endif
}

void imageInputLayer::forward(vector<double*>& bottom, vector<double*>& top)
{
    #ifdef WITH_OPENCV
    while(true) {
        if (++m_indexImage >= m_imagesPath.size()) {
            m_indexImage = -1;
            top.clear();
            top.push_back(NULL);
            top.push_back((double*)&m_indexImage);
            return ;
        }
        *m_currImage = imread(m_imagesPath[m_indexImage], 0);
        if(m_currImage->data) {
            resize(*m_currImage, *m_currImage, Size(m_oWidth, m_oHeight));
            m_currImage->convertTo(*m_currImage, CV_64FC1, 0.0039215684f * 2.0, -1.0);
            cout << "Processing Image: " << m_imagesPath[m_indexImage] << endl;
            top.clear();
            top.push_back((double*)m_currImage->data);
            top.push_back((double*)&m_indexImage);
            return ;
        }
    }
    #else
    assert(0);
    #endif
}

void imageInputLayer::backward(vector<double*>& top, vector<double*>& bottom)
{
    // without backward in imageInputLayer
}

void imageInputLayer::scanImages(string folderSrc)
{
    DIR *d;
    struct dirent *dir;
    d = opendir(folderSrc.c_str());
    if (d) {
        while ((dir = readdir(d)) != NULL) {
            m_imagesPath.push_back(folderSrc + dir->d_name);
        }
        closedir(d);
    }
    assert(m_imagesPath.size() > 0);
}

}
