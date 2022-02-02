// The imports
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#define KB_ENTER int('\n')

using namespace cv;
using namespace std;

// Variables

// Vector of Seeds
vector<cv::Point2i> germes;
// Vector of Vectors of regions
vector<vector<cv::Point2i>> regions;
// Vector of regions
vector<cv::Point2i> reg;
// List of seeds already used
vector<cv::Point2i> usedSeed;
// Seed declaration
cv::Point2i seed;
// The homogeneity threshold
int seuil = 35;
// Vector of colors
Vec4b color ;
vector<Vec4b> SeedsColors;
// Matrices
Mat img3 ;
Mat img4;
Mat outputImg ;


// Methods

// get_random_color
int getColor(){
    int x;
    x = rand()%255;
    return x;
}

// use_mouse
void onMouse(int event, int x, int y, int flags, void* param)
{
    Mat &xyz = *((Mat*)param);//cast and deref the param

    unsigned char * xyzr = xyz.data;

    if (event == CV_EVENT_LBUTTONDOWN)
    {
        cout<< "chosing germes : "<< "x = " << x << " y = " << y <<  endl;
        int val;

        val = (xyz.at< cv::Vec3b >(y,x)[0] + xyz.at< cv::Vec3b >(y,x)[1] + xyz.at< cv::Vec3b >(y,x)[2])/3;
        cout<<"with vec : "<<val<<endl;

        int val2;
        unsigned long k;
        k = xyz.channels()*(y*xyz.cols +x);
        val2 = (xyzr[k] + xyzr[k+1]+xyzr[k+2])/3;
        cout<<"with k : "<<val2<<endl;

        germes.push_back(cv::Point2i(x, y));
        cout<<"la liste de tous les germes est:" <<germes<<endl;
    }
}

// Get_8_neighbors of The Pixel
vector<cv::Point2i> getNeighbors(cv::Point2i pc){
    vector<cv::Point2i> neighbors;
    int rows = img3.rows;// Line
    int cols = img3.cols;//cologne

    if(pc.x-1>=0){
        neighbors.push_back(cv::Point2i(pc.x-1,pc.y));
    }
    if(pc.x-1>=0 && pc.y+1<=cols){
        neighbors.push_back(cv::Point2i(pc.x-1,pc.y+1));
    }
    if(pc.y+1<=cols){
        neighbors.push_back(cv::Point2i(pc.x,pc.y+1));
    }
    if(pc.x+1<=rows && pc.y+1<=cols){
        neighbors.push_back(cv::Point2i(pc.x+1,pc.y+1));
    }
    if(pc.x+1<=rows){
        neighbors.push_back(cv::Point2i(pc.x+1,pc.y));
    }
    if(pc.x+1<=rows && pc.y-1>=0){
        neighbors.push_back(cv::Point2i(pc.x+1,pc.y-1));
    }
    if(pc.y-1>=0){
        neighbors.push_back(cv::Point2i(pc.x,pc.y-1));
    }
    if(pc.x-1>=0 && pc.y-1>=0){
        neighbors.push_back(cv::Point2i(pc.x-1,pc.y-1));
    }
    return neighbors;
}


// get_gray_value of the pixel
int grayValue(cv::Point2i pc){
    unsigned char * imr = img3.data;
    int x;

    unsigned long kk;
    kk = img3.channels()*(pc.y*img3.cols +pc.x);
    x = (imr[kk] + imr[kk+1]+imr[kk+2])/3;

    return x;
}

// get the average gray value of the region
int RegionMoyGrayValue(vector<cv::Point2i> reg){

    int val;
    int moyVal;

    val = (img3.at< cv::Vec3b >(reg[0].y,reg[0].x)[0] + img3.at< cv::Vec3b >(reg[0].y,reg[0].x)[1] + img3.at< cv::Vec3b >(reg[0].y,reg[0].x)[2])/3 ;

    for(int i = 1; i < reg.size(); i++ ){
        val += (img3.at< cv::Vec3b >(reg[i].y,reg[i].x)[0] + img3.at< cv::Vec3b >(reg[i].y,reg[i].x)[1] + img3.at< cv::Vec3b >(reg[i].y,reg[i].x)[2])/3 ;
    }
    moyVal = val/reg.size();
    return moyVal;
}

// compare the homogeneity
bool isHomogene(cv::Point2i pc,vector<cv::Point2i> reg){
    int val1;
    int val2;
    val1 = grayValue(pc);
    val2 = RegionMoyGrayValue(reg);
    if(std::abs(val1-val2)<= seuil){
        return true;
    }else{
        return false;
    }
}


// Check is pixel in region_vector
bool isPixelInRegions (cv::Point2i pi,vector<vector<cv::Point2i>> regs){
    for(int i = 0; i< regs.size();i++){
        for(int j =0;j<regs[i].size();j++){
            if(regs[i][j].x == pi.x && regs[i][j].y == pi.y){
                return true;
            }
        }
    }
    return false;
}

// Check is pixel in region
bool isPixelInReg (cv::Point2i pi,vector<cv::Point2i> regs){
    for(int i = 0; i< regs.size();i++){
        if(regs[i].x == pi.x && regs[i].y == pi.y){
            return true;
        }
    }
     return false;
}

// compare the homogeneity between the seed and neighbors
bool isNeighbHomogeneWithSeed(Point2i neighbor,Point2i currentSeed){
    int val1;
    int val2;

    val1 =  grayValue(neighbor);
    val2 = grayValue(currentSeed);
    if(std::abs(val1-val2)<= seuil){
        return true;
    }
    return false;
}

//compare the homogeneity with the old Seeds
bool IsSeedHomogeneWithPreviousSeed(Point2i pc){
    for (int i = 0;i<usedSeed.size();i++ ){
        if(isNeighbHomogeneWithSeed(usedSeed[i],pc)){
            color[0]= SeedsColors[i][0];
            color[1]= SeedsColors[i][1];
            color[2]= SeedsColors[i][2];
            color[3]= SeedsColors[i][3];
            return true;
        }
    }
    return false;
}

// The growing region, The pillar method
void growing_region(Point2i currentSeed , vector<Point2i> neighbs,void* outputParam){

    Mat &matt2 = *((Mat*)outputParam);
    unsigned char * outputImgr2 = matt2.data;
    unsigned long d;
        if(neighbs.empty())
            return;
        for(int i =0;i<neighbs.size();i++){
            if(!isPixelInRegions(neighbs[i],regions)){
                if(isNeighbHomogeneWithSeed(neighbs[i],currentSeed) && !isPixelInReg(neighbs[i],reg) ){

                reg.push_back(neighbs[i]);

                d = matt2.channels()*(neighbs[i].y*matt2.cols +neighbs[i].x);
                outputImgr2[d] = color[1];
                outputImgr2[d+1] = color[2];
                outputImgr2[d+2] = color[3];
                imshow( "Image Gray2", matt2 );
                waitKey(1);
                vector<Point2i> newNeighbs;
                newNeighbs = getNeighbors(neighbs[i]);
                growing_region(currentSeed,newNeighbs,(void*)&matt2);
            }
        }
    }
}

// Operators
ostream& operator<<(ostream& os, const vector<vector<cv::Point2i>>& v){
    os<<v.size()<<"\n";
    os << "[";
    for (int i = 0; i < v.size(); ++i) {
            os << "(";
            for(int j=0;j<v[i].size();++j){
                os << v[i][j];
                if (j != v[i].size() - 1)
                    os << ", ";}
                os << ")\n";
                if (i != v.size() - 1)
                    os << ", ";

    }
    os << "]\n";
    return os;
}

// The main
int main(int argc, char** argv)
{
    // reading the picture
    Mat img1 = imread( "C:/Users/PC/Desktop/projects/Artificial_Vision/Floor_Occupation/img.jpeg", 1 );
    // if the image is not found display error
    if(img1.empty()){
        cout<<"Erreur de lecture"<<endl;
        return -1;
    }

    // duplicate the image
    // transfer image to gray
    namedWindow( "Image Original", cv::WINDOW_NORMAL );
    imshow( "Image Original", img1 );

    cvtColor(img1, img3, COLOR_RGB2GRAY );

    namedWindow( "Image Gray", cv::WINDOW_NORMAL );
    imshow( "Image Gray", img3 );

    // Call Mouse
    setMouseCallback("Image Gray", onMouse, (void*)&img3);

    waitKey( 0 );

    cout<<"press ENTRER ... "<<endl;

    unsigned long k;

    if(cin.get() == KB_ENTER){

        cvtColor(img3, outputImg, CV_GRAY2RGB);
        unsigned char * outputImgr = outputImg.data;

        namedWindow( "Image Gray2", cv::WINDOW_NORMAL );
        imshow( "Image Gray2", outputImg );

        cout<<"Starting Process ..."<<endl;
        while(!germes.empty()){

            seed = cv::Point2i(germes[0].x, germes[0].y);
            if(!usedSeed.empty()){
                if(IsSeedHomogeneWithPreviousSeed(seed)){
                    //
                }else{
                    color[0]= grayValue(seed);
                    color[1]= getColor();
                    color[2]= getColor();
                    color[3]= getColor();
                }
            }else{
                color[0]= grayValue(seed);
                color[1]= getColor();
                color[2]= getColor();
                color[3]= getColor();
            }

            if (isPixelInRegions(seed,regions)){
                germes.erase(germes.begin());
            }else{
                reg.push_back(seed);
                k = outputImg.channels()*(seed.y*outputImg.cols +seed.x);
                outputImgr[k] = color[1];
                outputImgr[k+1] = color[2];
                outputImgr[k+2] = color[3];

                SeedsColors.push_back(color);
                imshow( "Image Gray2", outputImg );
                waitKey(1);

                vector<Point2i> neighbours;
                neighbours = getNeighbors(seed);

                growing_region(seed,neighbours,(void*)&outputImg);

                usedSeed.push_back(seed);
                germes.erase(germes.begin());
                regions.push_back(reg);
                reg.clear();
            }
        }
    }
        imwrite( "C:/Users/PC/Desktop/projects/Artificial_Vision/Floor_Occupation/output_img.jpeg", outputImg );
        cout<<"Process finished !"<<endl;
}
