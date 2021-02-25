#ifndef headergaurd
#define headergaurd

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <string.h>
#include <math.h>
#include <iostream>
#include <time.h> 
// Uncomment the following two lines for visualization and verbose behind steps
// #define vis
// #define verbose
#define PI 3.14159265
#define max_rollers 16
using namespace cv;
using namespace std;


class rollerDetector{

 private:
    String name,write;
    Mat in,col,thresholded,adaptive,Mask;
    Vec3f out_circle,ref_roller;
    array<float,max_rollers> diff;
    vector<Vec3f> rollers,corrected_rollers,centres_circle;
    int result;
    double theta_ref;
    int filter_size1,filter_size2;

 public:
    void set_name(String n){
        name = n;
    }
    void setInput(Mat &input, Mat &colored){
        in = input;
        col = colored;
    }
    //Choosing filter size according to the size of the bearing in the image
    void setFilterSize(bool found_outer){
        if(found_outer && out_circle[2]<60){
            filter_size1=3;
            filter_size2=5;
        }
        else{ filter_size1=5;
            filter_size2=7;}
    }
    // Draw circles on a reference image when circles are given as vectors of Vec3f(x,y,r)
    void draw_circles(Mat &im, vector<Vec3f> circles, Scalar color, int thickness){
        std::vector<Vec3f>::const_iterator itc = circles.begin();
        while (itc!=circles.end()) {
            circle(im, 
            Point((*itc)[0], (*itc)[1]), // circle centre
            (*itc)[2], // circle radius
            color, // color 
            thickness); // thickness
            ++itc;	
        }
    }
    // Linear Algebra approach of projecting to column space to fit a circle for a given set of points lying on a circle
    //Circle in 2D can be represented as linear equation in 3 variables with x=g(centre's x-coord), y=f(y-coord) and z=c
    void fit_circle_LA(vector<Vec2f> &centres){
        centres_circle.clear();
        int len = centres.size();
        Mat original = col.clone();
        Mat a(len,3,CV_32FC1);
        Mat x(3,1,CV_32FC1);
        Mat b(len,1,CV_32FC1); // ax=b
        Mat p(len,1,CV_32FC1); // p is projection of b onto column space of a
        for(int i=0;i<len;i++){
            a.at<float>(i,0)=2*centres[i][0];
            a.at<float>(i,1)=2*centres[i][1];
            a.at<float>(i,2)=1;
            b.at<float>(i,0)= -(pow(centres[i][0],2)+pow(centres[i][1],2));
        }
        p=a*((a.t()*a).inv()*a.t()*b); // projection 
        Mat a_trun = a(Range(0,3),Range(0,3)); // once we get p, we can obtain x using just three equations
        Mat p_trun = p(Range(0,3),Range(0,1));
        x = a_trun.inv()*p_trun; // x = a^-1 * p
        float x_c = -(x.at<float>(0,0));
        float y_c = -(x.at<float>(0,1));
        float r = sqrt( pow(x_c,2) + pow(y_c,2) - x.at<float>(0,2) ); // conversion from one circle representation(g,f,c) to (x_c,y_c,r)
        /*cout<< "b1= "<< b.at<float>(0,0)<< "b2 = " << b.at<float>(1,0)<< '\n';
        cout<< "p1= "<< p.at<float>(0,0)<< "p2 = " << p.at<float>(1,0)<< '\n';*/
        centres_circle.push_back(Vec3f(x_c,y_c,r));
        draw_circles(original,centres_circle,Scalar(255,255,0),2);
        #ifdef verbose
        cout<< "centre of fitted centres-circle , x= "<<x_c<<" y="<<y_c<<" radius="<<r<<'\n';
        #endif
        #ifdef vis
        namedWindow("circle fitted");
        imshow("circle fitted",original);
        #endif
    }
    //Applying some threshold and filters for detecting outer/bigger circle that surrounds the pins
    void preprocess1(){
        threshold(in,thresholded,100,256,THRESH_BINARY);
        medianBlur(thresholded,thresholded,5);
        medianBlur(thresholded,thresholded,5);
        #ifdef vis
        namedWindow("original bw");
        imshow("original bw",in);
        namedWindow("blurred");
        imshow("blurred",thresholded);
        #endif
        }
    //Different preprocessing for detecting rollers using Hough circle transform
    void preprocess2(int blurs){
        medianBlur(in,adaptive,filter_size1);
        medianBlur(adaptive,adaptive,filter_size1);
        adaptiveThreshold(adaptive,adaptive,255,ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY_INV,15,10);
        Mat element(3,3,CV_8U,Scalar(1));
    
        morphologyEx(adaptive,adaptive,MORPH_OPEN,element); // open and close morphology operators for obtaining better contours
        morphologyEx(adaptive,adaptive,MORPH_CLOSE,element);
        
        while(blurs--)
        medianBlur(adaptive,adaptive,filter_size1);
    
        Mat sob;
        Sobel(adaptive,sob,CV_8U,1,1);
        adaptive = adaptive - sob*10; // Sharpening the image using Sobel operator
        #ifdef vis
        imshow("adaptive sharpened",adaptive);
        #endif
    }
    // Finding the bigger circle surrounding the bearings
    bool find_outer_circles(){
        int min_dim= min(thresholded.size[0],thresholded.size[1]);
        int max_radius = min_dim*3/4;
        int min_radius = min_dim*1/8;
        int min_votes = min_dim*1/3;
        vector<Vec3f> circles;
        //Fitting for outer circle whose radius ranges between 1/8th and 3/4th of the smallest dimestion of the image
        // and passes through atleast 1/3rd of the smallest diemsion-number of points
        HoughCircles(thresholded,circles,HOUGH_GRADIENT,2,200,100,min_votes,min_radius,max_radius);
        //The parameters are set such that only a single circle gets detected which is the surrounding circle
        #ifdef verbose
        std::cout << "#outer circles: " << circles.size() << std::endl;
        #endif
        if(circles.size()>0){
        Mat original = col.clone(); 
        out_circle = circles[0];
        draw_circles(original,circles,Scalar(0,255,255),2);
        #ifdef vis
        namedWindow("Detected outer Circle");
        imshow("Detected outer Circle",original);
        #endif
        }
        
        if(circles.size()==1)
        return true;
        return false;
    }
    // Detecting possible rollers using Hough Circle accumulator 
    void detect_roller_circles(){
        int max_r;
        //cout<<out_circle[2]<<'\n';
        if(out_circle[2]>=220)
        max_r=35;
        else max_r=30;
        int min_r=5;
        HoughCircles(adaptive,rollers,HOUGH_GRADIENT,2,15,100,60,min_r,max_r);
        #ifdef verbose
        cout << "#Rollers detected by Hough : " << rollers.size() << std::endl;
        #endif
        if(rollers.size()>0){
        Mat original = col.clone();
        draw_circles(original,rollers,Scalar(0,255,255),2);
        #ifdef vis
        namedWindow("Detected roller Circles");
        imshow("Detected roller Circles",original);
        #endif
    }}
    // Finding the circle passing through the centres of the detected Hough circles for moving window approach
    bool find_centre_circle( bool outer_found){
        vector<Vec2f> centres;
        corrected_rollers.clear();
        Mat blank(col.size(),CV_8UC1,255);
        vector<Vec3f>::iterator it = rollers.begin();
        while (it!=rollers.end()) {
            //Only consider those circle that lie inside the outer circle and beyond half the radius of it
            if( pow(out_circle[0]- (*it)[0],2) + pow(out_circle[1]- (*it)[1],2) < pow(out_circle[2],2)
                && pow(out_circle[0]- (*it)[0],2) + pow(out_circle[1]- (*it)[1],2) > 1/4*pow(out_circle[2],2)
                || outer_found == false){
                // roller centres lie within boundaries
                corrected_rollers.push_back((*it));  
                circle(blank, 
                Point((*it)[0], (*it)[1]), // circle centre
                1, // circle radius
                Scalar(0,0,0), // color 
                1); // thickness
                centres.push_back(Vec2f((*it)[0], (*it)[1]));
                ++it;	
        }}
        #ifdef vis
        namedWindow("roller centres");
        imshow("roller centres",blank);
        #endif
        if(centres.size()>3){
        fit_circle_LA(centres);
        return true;}
        //Its not possible to fit a distinct circle with just two points so can't work on this image any further
        return false;
    }
    //Finding the suitable roller mask/roi for moving window comparision
    void find_ref_roller_and_mask(){
        vector<float> radii;
        float x,y;
        vector<Vec3f>::iterator it = corrected_rollers.begin();
        while(it!=corrected_rollers.end()){
            radii.push_back((*it)[2]);
            it++;
        }
        //Finding the 25th percentile radius as it's the most probable estimate
        size_t n = corrected_rollers.size()/4;
        nth_element(radii.begin(),radii.begin()+n,radii.end());
        float perc25 = radii[n]; 
        it = corrected_rollers.begin();
        while(it!=corrected_rollers.end()){
            if(perc25==(*it)[2]){
                x=(*it)[0];
                y=(*it)[1];
                break; }it++;}
        float side = perc25*sqrt(2);
        //Choosing the mask/filter as the rectangle that is enclosed within the circle with perc25 radius
        Mat roi(thresholded, Rect(x-side/2,y-side/2,side,side));
        Mask = roi; 
        ref_roller = Vec3f(x,y,perc25);
    }
    //Applying convolution with mask/filter for rectangle roi's along the centers_circle fitted using LA 
    int moving_window(int tol,int mode){
        Mat Original = col.clone();
        float r = ref_roller[2];
        float centre_radius = centres_circle[0][2];
        // Start with the point exactly to the right of centre on the circumference
        float in_x = centres_circle[0][0]+centre_radius;    
        float in_y = centres_circle[0][1];
        //mode 0 for base detection
        if(mode==0){
            //finding the angle that the line joining the centre of the reference circle(used for mask) makes
            // with the horizontal line from the from the centre of the centres_circle 
            double yt= ref_roller[1]-in_y;
            double xt= ref_roller[0]-in_x+centre_radius;
            theta_ref = atan2(yt,xt);
        }
    
        float x=0,y=0;
        int theta = 0;
        float thetar=0,theta_prev=0;
        float side = r* sqrt(2);
        float lastx=0,lasty=0;
        Mat dif;
        float error;
        int count = 0;
        diff[0]=0;
        //Looping along the circumference of the centres_circle, theta from 0 to 360
        while(theta<=360){
            theta+=1;
            thetar = theta*PI/180;//converting to radians
            //Finding the point (x,y) that lies on the circumference at an angle of thetar with the horizontal
            x=in_x-centre_radius + centre_radius* cos(thetar);
            y=in_y+ centre_radius * sin(thetar);
            Mat roi(thresholded, Rect(x-side/2,y-side/2,side,side)); // roi/patch
            absdiff(roi,Mask,dif); // comparing the mask/filter to the roi
            error = sum(dif)[0]/255; // As the images are binary(either 0 or 255 intensity), 
            // the number of pixel differences can be found by summing all differences and then dividing by 255
                //tol is the maximum allowed number of pixel differences
            if(error<tol){
                if(pow(lastx-x,2)+pow(lasty-y,2)>4*pow(r,2)){//if this new roi is atleast at a 2r distance from the previous one
                //so that we don't get multiple detections for the same roller
                #ifdef verbose
                cout<<"error= "<<error<<'\n';
                #endif
                circle(Original,Point(x,y),r,Scalar(0,0,255),2);
                //storing the the angle difference between the successive detections
                diff[count] = thetar - theta_prev;
                lastx=x;
                lasty=y;
                theta_prev=thetar;
                count++; //count of positive detections
                }
            }

        }
        diff[0] += 2*PI - theta_prev;
        //for(int i=0;i<count;i++)
        //cout<< "diff in radians = "<<diff[i]<<'\n';
        #ifdef vis
        if(mode==0){
        namedWindow("Base Rollers detection");
        imshow("Base Rollers detection",Original); }
        #endif
        //mode-1 for refined detections
        if(mode==1 && count==result && result<max_rollers){//if bearing is defective and has max detections, display the refined image
        imwrite(write,Original);
        namedWindow("Refined bad bearing");
        imshow("Refined bad bearing",Original); 
        }
        return count; 
    }
    //Base detection of all rollers
    void detect_rollers(int blurs,int tol){
        //Using the preprcess1 image-'thresholded' for sliding window 
        medianBlur(thresholded,thresholded,filter_size2);
        while(blurs--)
        medianBlur(thresholded,thresholded,filter_size1);
        find_ref_roller_and_mask();
        float r = ref_roller[2];
        #ifdef verbose
        cout<<"perc25_radius = "<< r << '\n';
        #endif
        write = "result/" + name;
        result = moving_window(tol,0);//mode 0 for base detection
        cout<<"No of rollers/pins detected = "<< result<<'\n';
        if(result==max_rollers)  cout<<"GOOD! It's a good bearing\n";
        else            cout<<"BAD! It's a defective bearing\n";
    }
    //Refining for obtaining better roller boundaries as the base detection doesnt give exact boundaries
    // becuase of the trade-off between tolerance (base) and detecting all possible rollers 
    // As the rollers/pins are not noise-free, some pins would give small error whereas a high-noise pin
    // requires high tolerance to used
    void refine(int tol){
        while(true){
            //Start with small tolerance and increment unitl max pins are detected
            int detected = moving_window(tol,1); // mode 1 for refining
            if(detected==result){
            //If it's a perfect bearing , working on further refinement for better boundaries
            if(result == max_rollers)
            break;
            else if(result<max_rollers)   
            //Else if it's a defective bearing , not proceeding any further        
            return;}
            tol++;
        }
        //Finding mean of angle differences
        float mean_theta = 0;
        for(int i=0;i<max_rollers;i++){
            if(diff[i]==0)
            continue;
            mean_theta+=diff[i];
            }
        mean_theta/=max_rollers;
        #ifdef verbose
        cout<<"mode="<<mean<<"theta ref"<<theta_ref<<'\n';
        #endif
        float centre_radius = centres_circle[0][2];
        float theta_c = theta_ref - floor(theta_ref/mean_theta)*mean_theta; //angle the centre of first circle detected in moving window
        //makes with horizontal
        float in_x = centres_circle[0][0]+ centre_radius;// Starting with the point similar as 'moving_window'
        float in_y = centres_circle[0][1];
        float r = ref_roller[2];
        float thetar=0;
        float x,y;
        Mat Original = col.clone();
        for(int i=0; i< max_rollers; i++){
            //As a perfect bearing is symmetric , each bearing is at an angle 'mean_theta' apprx from the previous one
            thetar= (theta_c + mean_theta*i);
            x=in_x-centre_radius + centre_radius* cos(thetar);
            y=in_y+ centre_radius * sin(thetar);
            circle(Original,Point(x,y),r,Scalar(0,0,255),2);
         }
         imwrite(write,Original);
         imshow("Refined good bearing",Original);     
         
    }
};
#endif
