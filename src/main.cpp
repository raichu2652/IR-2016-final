#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "omp.h"

#include <algorithm>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>
#include <dirent.h>

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
  // download query images from Google
  char command[100], tempDir[20] = "./temp/";
  if (argc == 2) {
    sprintf(command, "./script/crawler.py -n %d -q %s -o %s", DATA_SIZE, argv[1], tempDir);
    printf("execute %s\n", command);
    system(command);
  }
  
  // read all images
  // convert images from BGR to Lab color space
  // extract 5d features (L, a, b, x, y) for each images
  // L, a, b in (0, 255)
  Mat images[DATA_SIZE], labels[DATA_SIZE], means[DATA_SIZE];
  vector<Mat> samples(DATA_SIZE);
  vector<Mat> likelihoods(DATA_SIZE);
  EM em[DATA_SIZE];
  # pragma omp parallel for
  for (int i = 0; i < DATA_SIZE; ++i) {
    char filename[20];
    sprintf(filename, "%s%d.jpg", tempDir, i);
    Mat img, raw = imread(filename, CV_LOAD_IMAGE_COLOR);
    resize(raw, img, Size(SIZE, SIZE));
    cvtColor(img, images[i], CV_BGR2Lab);
    
    printf("read[%d] %s\n", i, filename);
    samples[i] = Mat(SIZE*SIZE, 5, CV_64F);
    # pragma omp parallel for collapse(2)
    for (int r = 0; r < SIZE; ++r) {
      for (int c = 0; c < SIZE; ++c) {
        int j = r * SIZE + c;
        Vec3b value = images[i].at<Vec3b>(r, c);
        samples[i].at<double>(j, 0) = value[0] / 256.0;
        samples[i].at<double>(j, 1) = value[1] / 256.0;
        samples[i].at<double>(j, 2) = value[2] / 256.0;
        samples[i].at<double>(j, 3) = r / SIZE_F;
        samples[i].at<double>(j, 4) = c / SIZE_F;
      }
    }

    em[i] = EM(4, EM::COV_MAT_DIAGONAL);
    em[i].train(samples[i], likelihoods[i], labels[i], noArray());

    means[i] = em[i].get<Mat>("means");
//    draw(filename + 7, images[i], labels[i], means[i]);
  }

  // retrieve image representation by EM - GMM distribution model on each images *4 clusters
  // merge image model into catergory model
  vector<vector<int> > groups;
  int ret = merge(samples, likelihoods, groups, em);
  for (int i = 0; i < ret; ++i) {
    printf("[%d]:%d\n", i, samples[i].rows);
    for (int j = 0; j < groups[i].size(); ++j) {
      printf("%d ", groups[i][j]);
    }
    printf("\n");
  }
  
  // predict and retrieve highest image in data set
  DIR *directory;
  struct dirent *file;
  char localDir[20] = "./local/";
  char candidates[100][50] = {{}};
  Mat features[100];
  directory = opendir(localDir);
  int n = 0;
  if (directory) {
    while ((file = readdir(directory)) != NULL) {
      if (file->d_type == DT_DIR) continue;

      sprintf(candidates[n], "%s%s", localDir, file->d_name);
      printf("[%d]: %s\n", n, candidates[n]);
      Mat img, lab, raw = imread(candidates[n], CV_LOAD_IMAGE_COLOR);
      resize(raw, img, Size(SIZE, SIZE));
      cvtColor(img, lab, CV_BGR2Lab);
      
      features[n] = Mat(SIZE*SIZE, 5, CV_64F);
      # pragma omp parallel for collapse(2)
      for (int r = 0; r < SIZE; ++r) {
        for (int c = 0; c < SIZE; ++c) {
          int j = r * SIZE + c;
          Vec3b value = lab.at<Vec3b>(r, c);
          features[n].at<double>(j, 0) = value[0] / 256.0;
          features[n].at<double>(j, 1) = value[1] / 256.0;
          features[n].at<double>(j, 2) = value[2] / 256.0;
          features[n].at<double>(j, 3) = r / SIZE_F;
          features[n].at<double>(j, 4) = c / SIZE_F;
        }
      }

      ++n;
    }

    closedir(directory);
  }

  map<int, double> list;
  for (int i = 0; i < n; ++i) {
    double minDistance = 1.0;
    Mat likelihood;
    EM emc = EM(4, EM::COV_MAT_DIAGONAL);
    emc.train(features[i], likelihood, noArray(), noArray());

    for (int j = 0; j < ret; ++j) {
      int c1 = likelihood.rows;
      int c2 = likelihoods[j].rows;
      double sum = c1 + c2;

      Mat predict1, predict2;
      #pragma omp parallel sections
      {
        #pragma omp section
        {
          predict1 = Mat(c2, 1, CV_64FC1);
          for (int k = 0; k < c2; ++k) {
            predict1.at<double>(k, 0) = emc.predict(samples[j].row(k))[0];
          }
        }
        #pragma omp section
        {
          predict2 = Mat(c1, 1, CV_64FC1);
          for (int k = 0; k < c1; ++k) {
            double lsum = 0.0;
            for (int g = 0; g < groups[j].size(); ++g) {
              lsum += em[groups[j][g]].predict(features[i].row(k))[0];
            }
            double csum = 0.0;
            for (int g = 0; g < groups[j].size(); ++g) {
              double ck = SIZE;
              csum += ck * pow(2, lsum - em[groups[j][g]].predict(features[i].row(k))[0]);
            }
            predict2.at<double>(k, 0) = log2(csum) + lsum - log2(c1);
          }
        }
      }
      Mat first = (likelihood * (c1 / sum)) + (predict2 * (c2 / sum));
      Mat second = (likelihoods[j] * (c2 / sum)) + (predict1 * (c1 / sum));
      Mat clikelihood = first.clone();
      clikelihood.push_back(second);

      double distance = kl_distance(SIZE*SIZE*(DATA_SIZE+1), likelihood, likelihoods[j], clikelihood);

      if (minDistance > distance) {
        printf("minDistance(%d, %d) = %lf\n", i, j, distance);

        list[i] = distance;
        minDistance = distance;
      }
    }
  }

  for (map<int, double>::iterator it = list.begin(); it != list.end(); ++it) {
    printf("%lf: %s\n", it->second, candidates[it->first]);
  }
  
//  imshow(filename, images[i]);
//  waitKey(0);

  return 0;
}
