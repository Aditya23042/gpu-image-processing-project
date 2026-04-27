#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <iostream>
#include <filesystem>
#include <chrono>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

int main() {

    cv::ocl::setUseOpenCL(true);

    cout << "OpenCL Enabled: " << cv::ocl::useOpenCL() << endl;

    fs::create_directory("output");

    int count = 0;

    auto start = chrono::high_resolution_clock::now();

    for (auto &file : fs::directory_iterator("images")) {

        Mat img = imread(file.path().string());

        if (img.empty()) continue;

        UMat gpuImg, gray, blurImg;

        img.copyTo(gpuImg);

        cvtColor(gpuImg, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, blurImg, Size(7,7), 1.5);

        Mat result;
        blurImg.copyTo(result);

        imwrite("output/out_" + to_string(count) + ".jpg", result);

        count++;
    }

    auto stop = chrono::high_resolution_clock::now();

    cout << "Processed: " << count << " images\n";
    cout << "Time(ms): "
         << chrono::duration_cast<chrono::milliseconds>(stop-start).count()
         << endl;

    return 0;
}