
#include "dataLayer.h"

namespace TUPU {

dataLayer::dataLayer(int oWidth, int oHeight, int oDepth,
    int totalCount, string imagePath, string labelPath)
{
    m_imagePath = imagePath;
    m_labelPath = labelPath;
    m_totalCount = totalCount;

    m_oWidth = oWidth;
    m_oHeight = oHeight;
    m_oDepth = oDepth;

    m_imageData = new double[m_oWidth * m_oHeight * m_totalCount];
    m_labelData = new double[m_totalCount];

    readMnistImages(m_imagePath, m_imageData, m_totalCount);
    readMnistLabels(m_labelPath, m_labelData, m_totalCount);

    m_indexImage = m_totalCount - 1; // start from 0 after forward
}

dataLayer::~dataLayer()
{
    delete[] m_imageData;
    delete[] m_labelData;
}

void dataLayer::forward(vector<double*>& bottom, vector<double*>& top)
{
    assert(bottom.size() == 0);
    if (++m_indexImage >= m_totalCount) {
        m_indexImage = 0;
    }
    top.clear();
    top.push_back(m_imageData + (m_oWidth * m_oHeight * m_indexImage));
    top.push_back(m_labelData + m_indexImage);
}

void dataLayer::backward(vector<double*>& top, vector<double*>& bottom)
{
    // without backward in data layer
}

int dataLayer::reverseInt(int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void dataLayer::readMnistImages(string fileName,
    double* dataDst, int numImage)
{
    const int widthSrcImage = 28;
    const int heightSrcImage = 28;
    const int xPadding = 2;
    const int yPadding = 2;
    const double scaleMin = -1;
    const double scaleMax = 1;

    ifstream file(fileName, ios::binary);
    assert(file.is_open());

    int magicNumber = 0;
    int numberOfImages = 0;
    int nRows = 0;
    int nCols = 0;
    file.read((char*)&magicNumber, sizeof(magicNumber));
    magicNumber = reverseInt(magicNumber);
    file.read((char*)&numberOfImages, sizeof(numberOfImages));
    numberOfImages = reverseInt(numberOfImages);
    assert(numberOfImages == numImage);
    file.read((char*)&nRows, sizeof(nRows));
    nRows = reverseInt(nRows);
    file.read((char*)&nCols, sizeof(nCols));
    nCols = reverseInt(nCols);
    assert(nRows == heightSrcImage && nCols == widthSrcImage);

    int sizeSingleImage = m_oWidth * m_oHeight;

    for (int i = 0; i < numberOfImages; ++i) {
        int addr = sizeSingleImage * i;

        for (int r = 0; r < nRows; ++r) {
            for (int c = 0; c < nCols; ++c) {
                unsigned char temp = 0;
                file.read((char*)&temp, sizeof(temp));
                dataDst[addr + m_oWidth * (r + yPadding)\
                    + c + xPadding] = (temp / 255.0)\
                    * (scaleMax - scaleMin) + scaleMin;
            }
        }
    }
}

void dataLayer::readMnistLabels(string fileName,
    double* dataDst, int numImage)
{
    ifstream file(fileName, ios::binary);
    assert(file.is_open());

    int magicNumber = 0;
    int numberOfImages = 0;
    file.read((char*)&magicNumber, sizeof(magicNumber));
    magicNumber = reverseInt(magicNumber);
    file.read((char*)&numberOfImages, sizeof(numberOfImages));
    numberOfImages = reverseInt(numberOfImages);
    assert(numberOfImages == numImage);

    for (int i = 0; i < numberOfImages; ++i) {
        unsigned char temp = 0;
        file.read((char*)&temp, sizeof(temp));
        dataDst[i] = temp;
    }
}

}
