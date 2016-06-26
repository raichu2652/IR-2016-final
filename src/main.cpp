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

#define DATA_SIZE 10
#define SIZE 100
#define SIZE_F 100.0

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
  Mat images[SIZE], samples[SIZE], logLikelihoods[SIZE], means[SIZE];
  # pragma omp parallel for
  for (int i = 0; i < DATA_SIZE; ++i) {
    char filename[20];
    sprintf(filename, "%s%d.jpg", tempDir, i);
    Mat img = imread(filename, CV_LOAD_IMAGE_COLOR);
    cvtColor(img, images[i], CV_BGR2Lab);
    
    printf("read %d\n", i);
    samples[i] = Mat(SIZE*SIZE, 5, CV_64F);
    # pragma omp parallel for collapse(2)
    for (int r = 0; r < SIZE; ++r) {
      for (int c = 0; c < SIZE; ++c) {
        int j = r * SIZE + c;
        int y = r * images[i].rows / SIZE;
        int x = c * images[i].cols / SIZE;
        Vec3b value = images[i].at<Vec3b>(y, x);
        samples[i].at<double>(j, 0) = value[0] / 256.0;
        samples[i].at<double>(j, 1) = value[1] / 256.0;
        samples[i].at<double>(j, 2) = value[2] / 256.0;
        samples[i].at<double>(j, 3) = r / SIZE_F;
        samples[i].at<double>(j, 4) = c / SIZE_F;
      }
    }

    EM em = EM(4, EM::COV_MAT_DIAGONAL);
    em.train(samples[i], logLikelihoods[i], noArray(), noArray());

    means[i] = em.get<Mat>("means");
//    draw(filename + 7, means[i]);
  }

  // retrieve image representation by EM - GMM distribution model on each images *4 clusters
  // merge image model into catergory model
  vector<Mat> sample, logl;
  sample.assign(samples, samples + DATA_SIZE);
  logl.assign(logLikelihoods, logLikelihoods + DATA_SIZE);
  int ret = merge(sample, logl);
  for (int i = 0; i < ret; ++i)
    printf("[%d]:%d\n", ret, sample[i].rows);
  
  // predict and retrieve highest image in data set
  //
  
//  imshow(filename, images[i]);
//  waitKey(0);

  return 0;
}
