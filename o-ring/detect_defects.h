#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <string.h>
#include <math.h>
#include <iostream>
using namespace cv;
using namespace std;
class defect_detector{
private:
    Mat in,eqd,bil,lap,can;
    vector<KeyPoint> keypoints; //keypoints for blobs
public: 
    void setInput(Mat &input){
        in = input;
        eqd=in;
        bil=in;
    }
    void equalize(){
        Mat halved,out;
        cvtColor(in,out,COLOR_BGR2HSV); // converting to HSV colorspace
        vector<Mat> channels(3);
        split(out,channels); //splitting the mat into its corresponding channels
        Ptr<CLAHE> clahe = createCLAHE();
        clahe->setClipLimit(1);
        //channels[2] = channels[2]/2; //this gave better results for slide.jpg
        merge(channels,halved);
        cvtColor(halved,halved,COLOR_HSV2BGR);
        //imshow("halved",halved);
        clahe->setTilesGridSize(Size(4,4)); //set to lower value if image is too bright
        clahe->apply(channels[2],channels[2]); // applying CLAHE to value channel
        merge(channels,out);
        cvtColor(out,out,COLOR_HSV2BGR);
        imshow("original",in);
        //imwrite("cutsclahe.jpg",out);
        imshow("clahe",out);
        eqd = out;
    }
    void getHarriscorners(){
        //Corner detection didnt work as expected because of reflections
        cv::Mat cornerStrength;
        cv::cornerHarris(in, cornerStrength,
                        3,     // neighborhood size
                        3,     // aperture size
                        0.01); 

    // threshold the corner strengths
        cv::Mat harrisCorners;
        double threshold= 0.001; 
        cv::threshold(cornerStrength, harrisCorners,
                    threshold,255,cv::THRESH_BINARY_INV);
        
        cv::namedWindow("Harris");
        cv::imshow("Harris", harrisCorners);

    }
    void bilateral(){
        bilateralFilter(in,bil,15,40,40); // These parameters would also need to be tuned depending on the image
        imshow("bilateral filter",bil);
    }
    void laplacian(){
         Laplacian(eqd,lap,CV_8UC3);
         imshow("laplacian",lap);
    }
    void canny(){
         Canny(lap,can,100,200);
         imshow("Canny",can);
    }
    void gradients(){
        //no luck wiht gradient orientation either
        Mat gx,gy,norm,dir;
        Sobel(in,gx,CV_32F,1,0,3);
        Sobel(in,gy,CV_32F,0,1,3);
        cartToPolar(gx,gy,norm,dir,true);
        //cout<<dir;
    }
    void detect_simple_blobs(){
    SimpleBlobDetector::Params params; //setting parameters 
    params.minThreshold=50;
    params.maxThreshold=200;
    params.minArea=10;
    params.maxArea=3000;
    params.filterByCircularity=true;
    params.minCircularity=0.6f;
    params.filterByConvexity=true;
    params.minConvexity=0.6f;
    Ptr<SimpleBlobDetector> detector= SimpleBlobDetector::create(params);
    detector->detect( in , keypoints);
    Mat im_with_keypoints;
    drawKeypoints( in, keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    namedWindow("keypoints");
    // Show blobs
    imshow("keypoints", im_with_keypoints );
    std::cout<<"no of blobs="<<keypoints.size();
}
};