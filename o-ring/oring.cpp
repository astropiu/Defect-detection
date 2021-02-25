#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include "detect_defects.h"

using namespace cv;
using namespace std;
int main(){
    defect_detector detector;
    Mat in = imread("cuts.jpg");
    detector.setInput(in);
    detector.equalize();
    detector.laplacian();
    //After performing laplacian , the idea is to mask out the boundaries of the ring
    //so that the resulting image only has defects
    detector.canny();
    //When run on slide.jpg, this gave terrible results and needed modifications in the parameters
    detector.detect_simple_blobs();
    waitKey();
    return 0;

}