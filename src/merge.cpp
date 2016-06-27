#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "omp.h"

#include <iostream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "merge.h"

using namespace cv;
using namespace std;

int merge(vector<Mat> &images, vector<Mat> &likelihoods, vector<vector<int> > &collections, EM em[]) {
  int total = images[0].rows * images.size();
  double threshold = THRESHOLD;
  double minDistance = 0.0;
  int index1 = -1, index2 = -1;

  // Initialize combination log-likelihood vector
  int length = images.size();
  vector<Mat> categories(images);
  vector<vector<int> > groups(length, vector<int>());
  vector<Mat> glikelihoods(length);
  vector<vector<Mat> > clikelihoods(length, vector<Mat>(length));
  #pragma omp parallel for
  for (int i = 0; i < length; ++i) {
    groups[i].push_back(i);
    glikelihoods[i] = likelihoods[i].clone();
  }

  while (minDistance < threshold && length > 4) {
    minDistance = 1.0;

    // update combination log-likelihood vector
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < length; ++i) {
      for (int j = 0; j < length; ++j) {
        if (i >= j) continue;

        if (index1 == -1 || j == index1 || i == index1) {
          int c1 = glikelihoods[i].rows;
          int c2 = glikelihoods[j].rows;
          double sum = c1 + c2;

          Mat predict1, predict2;
          #pragma omp parallel sections
          {
            #pragma omp section
            {
              predict1 = Mat(c2, 1, CV_64FC1);
              for (int k = 0; k < c2; ++k) {
                double lsum = 0.0;
                for (int g = 0; g < groups[i].size(); ++g) {
                  lsum += em[groups[i][g]].predict(categories[j].row(k))[0];
                }
                double csum = 0.0;
                for (int g = 0; g < groups[i].size(); ++g) {
                  double ck = images[groups[i][g]].rows;
                  csum += ck * pow(2, lsum - em[groups[i][g]].predict(categories[j].row(k))[0]);
                }
                predict1.at<double>(k, 0) = log(csum) + lsum - log(c2);
              }
            }
            #pragma omp section
            {
              predict2 = Mat(c1, 1, CV_64FC1);
              for (int k = 0; k < c1; ++k) {
                double lsum = 0.0;
                for (int g = 0; g < groups[j].size(); ++g) {
                  lsum += em[groups[j][g]].predict(categories[i].row(k))[0];
                }
                double csum = 0.0;
                for (int g = 0; g < groups[j].size(); ++g) {
                  double ck = images[groups[j][g]].rows;
                  csum += ck * pow(2, lsum - em[groups[j][g]].predict(categories[i].row(k))[0]);
                }
                predict2.at<double>(k, 0) = log2(csum) + lsum - log2(c1);
              }
            }
          } 
          Mat first = (glikelihoods[i] * (c1 / sum)) + (predict2 * (c2 / sum));
          Mat second = (glikelihoods[j] * (c2 / sum)) + (predict1 * (c1 / sum));
          clikelihoods[i][j] = first.clone();
          clikelihoods[i][j].push_back(second);
        }
      }
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < length; ++i) {
      for (int j = 0; j < length; ++j) {
        if (i >= j) continue;

        double distance = kl_distance(total, glikelihoods[i], glikelihoods[j], clikelihoods[i][j]);
//        printf("distance(%d, %d) = %lf\n", i, j, distance);

        #pragma omp critical
        if (minDistance > distance) {
          minDistance = distance;
          index1 = i;
          index2 = j;
        }
      }
    }
    printf("Merge(%d, %d): %lf\n", index1, index2, minDistance);

    // merge categories
    categories[index1].push_back(categories[index2]);
    categories.erase(categories.begin() + index2);

    // merge glikelihoods index2 into index1
    clikelihoods[index1][index2].copyTo(glikelihoods[index1]);
    glikelihoods.erase(glikelihoods.begin() + index2);

    // merge groups
    groups[index1].insert(groups[index1].end(), groups[index2].begin(), groups[index2].end());
    groups.erase(groups.begin() + index2);

    // erase index2 in clikelihoods vector
    clikelihoods.erase(clikelihoods.begin() + index2);
    #pragma omp parallel for
    for (int i = 0; i < clikelihoods.size(); ++i) {
      clikelihoods[i].erase(clikelihoods[i].begin() + index2);
    }
    
    length = categories.size();
  }

  images = categories;
  likelihoods = glikelihoods;
  collections = groups;

  return length;
}

double kl_distance(int total, Mat logL1, Mat logL2, Mat logL) {
  double dist1 = 0.0, dist2 = 0.0;
  #pragma omp parallel sections
  {
    #pragma omp section
    {
      #pragma omp parallel for reduction(+:dist1)
      for (int i = 0; i < logL1.rows; ++i) {
        dist1 += logL1.at<double>(i, 0) - logL.at<double>(i, 0);
      }
      dist1 /= logL1.rows;
    }

    #pragma omp section
    {
      #pragma omp parallel for reduction(+:dist2)
      for (int i = 0; i < logL2.rows; ++i) {
        dist2 += logL2.at<double>(i, 0) - logL.at<double>(i + logL1.rows, 0);
      }
      dist2 /= logL2.rows;
    }
  }
  
  return ((logL1.rows * dist1) + (logL2.rows * dist2)) / total;
}

void draw(char* filename, Mat image, Mat label, Mat mean) {
  int count[4] = {0, 0, 0, 0};
  for (int i = 0; i < SIZE; ++i) {
    for (int j = 0; j < SIZE; ++j) {
      int l = label.at<int>(i*SIZE+j, 0);
      int r = mean.at<double>(l, 3) * 100;
      int c = mean.at<double>(l, 4) * 100;
      image.at<Vec3b>(i, j) = image.at<Vec3b>(r, c);

      count[l]++;
//      Point pt(i, j);
//      circle(clustered, pt, 1, color, -1);
    }
  }
  imwrite(filename, image);

  printf("Clusters[%d, %d, %d, %d]\n", count[0], count[1], count[2], count[3]);
}
