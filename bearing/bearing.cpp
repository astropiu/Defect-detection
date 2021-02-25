#include <iostream>
#include <filesystem>
#include <opencv2/core/core.hpp>
#include "detectRollers.h"
using namespace cv;
using namespace std;
namespace fs = std::filesystem;

int main(){
    rollerDetector detector; // Instance of the rollerDetector class in detectRoller.h
    int blurs=4,tol_base = 30,tol_refine = 5; //No. of median blurs for preprocessing and tolerance values for moving window
    String path ;
    cout<<"Enter the name of the folder "<<'\n';
    cin>>path;
    path = path + "/";
    //Looping over all images in the 'path' directory
    for (const auto & entry : fs::directory_iterator(path)){
    cout << entry.path() << '\n';
    String im_name=entry.path();
    String name = im_name.substr(path.size(),im_name.size());
    detector.set_name(name);
    Mat in = imread(im_name,0); //bw
    Mat col= imread(im_name);   //col
    resize(in,in,Size(640,480)); // Fix the size for processing 
    resize(col,col,Size(640,480));
    detector.setInput(in,col);
    detector.preprocess1();
    bool found_outer = detector.find_outer_circles(); // If outer circle is found
    detector.setFilterSize(found_outer);
    detector.preprocess2(blurs);
    detector.detect_roller_circles();
    bool possible =detector.find_centre_circle(found_outer);
    if(possible==false){
        cout<<"Sorry could't work on this image ";
        continue;
    }
    detector.detect_rollers(blurs,tol_base);
    detector.refine(tol_refine);
    waitKey();}
    return 0;
}