#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "omp.h"

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "merge.h"

using namespace cv;
using namespace std;

const char *QUERY_ARG = "-q";
/*const char *TRACE_ARG = "-t";
const char *NUMERIC_ARG = "-n";
const char *ALPHA_ARG = "-a";
const char *CONVERGENCE_ARG = "-c";
const char *SIZE_ARG = "-s";
const char *DELIM_ARG = "-d";
const char *ITER_ARG = "-m";*/

int main( int argc, char** argv ) {
  // retrieve image representation by EM - GMM distribution model on each images *4 clusters
  // merge image model into catergory model
  // predict and retrieve highest image in data set
  char query[20];
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], QUERY_ARG)) {
      printf("QUERY: %s\n", argv[i + 1]);
    }
  }

  // download query images from Google
  char command[100], tempDir[20] = "./temp/";
  sprintf(command, "./script/crawler.py -n 50 -q %s -o %s", argv[2], tempDir);
  printf("execute %s\n", command);
//  system(command);
  
  // read all images
  // convert images from BGR to Lab color space
  // extract 5d features (L, a, b, x, y) for each images
  // L, a, b in (0, 255)
  Mat images[50], samples[50], logLikelihoods[50], means[50];
  # pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
    char filename[20];
    sprintf(filename, "%s%d.jpg", tempDir, i);
    Mat img = imread(filename, CV_LOAD_IMAGE_COLOR);
    cvtColor(img, images[i], CV_BGR2Lab);
    
    printf("read %d\n", i);
    samples[i] = Mat(10000, 5, CV_64F);
    # pragma omp parallel for collapse(2)
    for (int r = 0; r < 100; ++r) {
      for (int c = 0; c < 100; ++c) {
        int j = r * 100 + c;
        int y = r * images[i].rows / 100;
        int x = c * images[i].cols / 100;
        Vec3b value = images[i].at<Vec3b>(y, x);
        samples[i].at<double>(j, 0) = value[0] / 256.0;
        samples[i].at<double>(j, 1) = value[1] / 256.0;
        samples[i].at<double>(j, 2) = value[2] / 256.0;
        samples[i].at<double>(j, 3) = r / 100.0;
        samples[i].at<double>(j, 4) = c / 100.0;
      }
    }

    EM em = EM(4, EM::COV_MAT_DIAGONAL);
    em.train(samples[i], logLikelihoods[i], noArray(), noArray());

    means[i] = em.get<Mat>("means");
    cout << means[i] << endl;

//    draw(filename + 7, means[i]);
  }

  vector<Mat> sample, logl;
  sample.assign(samples, samples + 10);
  logl.assign(logLikelihoods, logLikelihoods + 10);
  int ret = merge(sample, logl);
  printf("retur %d\n", ret);
//  imshow(filename, images[i]);
//  waitKey(0);

  return 0;
}
