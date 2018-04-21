#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;

const int rNum = 10; //row of SOM lattice
const int cNum = 10; //column of SOM lattice
const int nNeuron = 100; //number of SOM neurons
const int dTrnPat = 784;
const int epochNumber = 2000;
const int TrnPatNum = 5000;
const string iFileName = "data.txt"; // name of dataset
const string rFileName = "SOM.Train.Model"; // name of model
float sigma_0 = sqrt((0.5 * rNum) * (0.5 * rNum) + (0.5 * cNum) * (0.5 * cNum));
float sigma_t = sigma_0;
float tau_1 = 1000 / log(sigma_0);
float eta_0 = 0.1;
float eta_t = 0;
int tau_2 = 1000;

//random function used by random_shuffle
int randomFunc(int i) { return rand()%i; }
//generate random number
float randomNumber(){return (rand()/(float)(RAND_MAX));}
//Init som map
void randomWeight(Mat& m){
    for(int i = 0; i < m.rows; i++){
        for(int j = 0; j < m.cols; j++){
            m.at<float>(i,j) = randomNumber();
        }
    }
}
//Plot weight of som
void plotWeight(Mat& weight, int idx){

    //
    Mat tmpImg(280, 280, CV_32F);
    Mat somImg(280, 280, CV_32F);

    for(int i = 0; i < weight.cols; i++){
        Mat img;
        weight.col(i).copyTo(img);
        img = img.reshape(0,28);
        img.copyTo(tmpImg.rowRange(i / 10 * 28, i / 10 * 28 + 28).colRange(i % 10 *28, i % 10 * 28 + 28));
    }
    tmpImg.convertTo(tmpImg, CV_8UC1, 255.0);
    string index = "../finalImage/digit_" + to_string(idx)  + ".bmp";
    somImg = tmpImg.t();
    imwrite(index, somImg);
}

//find the winner
int findWinner(Mat& map, Mat input){
    float result = 0;
    float winner = 0;
    int winnerIdx = 0;
    for(int i = 0; i < map.cols; i++){
        result = norm(map.col(i), input);
        if(result < winner){
            winner = result;
            winnerIdx = i;
        }
    }
    return winnerIdx;
}
void cooperProcess(int winner, Mat map, float sigmaT, vector<float>& hNeighbor){
    //Get the position of winner in the map
    int winnerX = winner / 10;
    int winnerY = winner % 10;
    vector<float> disFromWinner;
    for(int i = 0; i < map.cols; i++){
        int x = i / 10;
        int y = i % 10;
        float distance = (winnerX - x) ^ 2 + (winnerY - y) ^ 2;
        disFromWinner.push_back(distance);
    }
    for(int i = 0; i < disFromWinner.size(); i++){
        float tmp = exp(-disFromWinner[i] / (2 * sigmaT * sigmaT));
        hNeighbor.push_back(tmp);
    }
}
int main() {
    cout << "start ******************** " << endl;
    fstream fout(rFileName, ios::out);
    Mat imageData; // store images in dataset
    float tmp; // temperate variable
    ifstream fin;
    fin.open(iFileName);
    if (!fin.good()) cout << "File does not exist" << endl;
    cout << "MNIST data are successfully loaded" << endl;

    //Read Images
    while (fin >> tmp) imageData.push_back(tmp);
    //split Images set into single image
    imageData = imageData.reshape(0, 784);

    Mat somWeight(dTrnPat, nNeuron, CV_32F);
    randomWeight(somWeight);
    cout << somWeight.size() << endl;
    plotWeight(somWeight, 0);
    eta_t = eta_0;

    Mat randomImageData(imageData);
    vector<int> imageIndex;
    srand(unsigned(time(0)));

    for (int i = 0; i < imageData.cols; i++)
        imageIndex.push_back(i);

    for (int i = 0; i < epochNumber; i++) {

        float weightDiff = 0;
        random_shuffle(imageIndex.begin(), imageIndex.end(), randomFunc);
        for (int j = 0; j < imageData.cols; j++)
            imageData.col(imageIndex[j]).copyTo(randomImageData.col(j));

        cout << "Now in epoch " << i << endl;
        for (int k = 0; k < TrnPatNum; k++) {
            int winnerIdx = findWinner(somWeight, randomImageData.col(k));
            vector<float> h_winNeuron;
            cooperProcess(winnerIdx, randomImageData, sigma_t, h_winNeuron);
            for (int n = 0; n < nNeuron; n++) {
                Mat weightTmp;
                somWeight.col(n).copyTo(weightTmp);
                somWeight.col(n) += eta_t * h_winNeuron[n] * (randomImageData.col(k) - somWeight.col(n));
                weightDiff += norm(weightTmp, somWeight.col(n));
            }
            eta_t = max(eta_0 * exp(-epochNumber / tau_2), 0.01);
            sigma_t = sigma_0 * exp(-i / tau_1);
            weightDiff = weightDiff / 100;
        }
        fout << weightDiff << endl;
        plotWeight(somWeight, i + 1);
    }
    fout.close();
    cout << "finish! ***************************" << endl;
    return 0;
}
    /*//Read image
    Mat =
    //define som map
    int numNeurons = rNum * cNum;
    vector<double> twoLatticeRows(rNum,0);
    vector<vector<double> >twoLattice;
    twoLattice.assign(cNum,twoLatticeRows);

    //define som map weight
    vector<double >weight(784,0);
    vector<vector<double> >weightArray;
    weightArray.assign(numNeurons, weight);
    //for checking the convergence
    vector<vector<double> >oldWeightArray(weightArray);
    //define the epoch numbers for two phrases
    int epochNumOrd = 100;
    int epochNumCov = 100;

    //initialise the learning rate and define the rate in each epoch t
    float lrnRate0 = 0.1;
    float lrnRate1 = lrnRate0;

    //define the attenuation speed of the learning rate with epoches
    int tau_2 = 1000;*/