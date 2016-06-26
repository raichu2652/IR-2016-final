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

int merge(vector<Mat> &categories, vector<Mat> &likelihoods) {
  int total = categories[0].rows * categories.size();
  double threshold = 0.25;
  double minDistance = 0.0;

  // Initialize combination log-likelihood vector
  int length = categories.size();
  vector<vector<Mat> > clikelihoods(length, vector<Mat>(length));
  while (minDistance < threshold && length > 4) {
    minDistance = 1.0;
    int index1 = -1, index2 = -1;

    // update combination log-likelihood vector
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < length; ++i) {
      for (int j = 0; j < length; ++j) {
        if (i >= j) continue;

        if (index1 == -1 || j == index1 || i == index1) {
          EM em = EM(4, EM::COV_MAT_DIAGONAL);
          Mat combination = categories[i].clone();
          combination.push_back(categories[j]);
          em.train(combination, clikelihoods[i][j], noArray(), noArray());
          
          printf("%d, %d\n", i, j);
        }
      }
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < length; ++i) {
      for (int j = 0; j < length; ++j) {
        if (i >= j) continue;
//        index = (2 * categories.size() - i - 2) * (i + 1) / 2 + j - i;
        double distance = kl_distance(total, likelihoods[i], likelihoods[j], clikelihoods[i][j]);
        printf("distance(%d, %d) = %lf\n", i, j, distance);

        #pragma omp critical
        if (minDistance > distance) {
          minDistance = distance;
          index1 = i;
          index2 = j;
        }
      }
    }
    printf("minDistance(%d, %d) = %lf\n", index1, index2, minDistance);

    // merge categories
    categories[index1].push_back(categories[index2]);
    categories.erase(categories.begin() + index2);
/*    #pragma omp parallel for
    for (int r = 0; r < categories[index1].rows; ++r) {
      categories[index1].at<double>(r, 0) = (categories[index1].at<double>(r, 0) + categories[index2].at<double>(r, 0)) / 2;
      categories[index1].at<double>(r, 1) = (categories[index1].at<double>(r, 1) + categories[index2].at<double>(r, 1)) / 2;
      categories[index1].at<double>(r, 2) = (categories[index1].at<double>(r, 2) + categories[index2].at<double>(r, 2)) / 2;
      categories[index1].at<double>(r, 3) = (categories[index1].at<double>(r, 3) + categories[index2].at<double>(r, 3)) / 2;
      categories[index1].at<double>(r, 4) = (categories[index1].at<double>(r, 4) + categories[index2].at<double>(r, 4)) / 2;
    }*/

    // erase index2 in clikelihoods vector
    clikelihoods.erase(clikelihoods.begin() + index2);
    #pragma omp parallel for
    for (int i = 0; i < clikelihoods.size(); ++i) {
      clikelihoods[i].erase(clikelihoods[i].begin() + index2);
    }
    
    length = categories.size();
  }

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

void draw(char* filename, Mat mean) {
  Mat clustered = Mat::zeros(Size(100, 100), CV_8UC3);
  for (int j = 0; j < 4; j++) {
    Point pt(cvRound(mean.at<double>(j, 3)*100), cvRound(mean.at<double>(j, 4)*100));
    circle(clustered, pt, 10, Scalar(0, 255, 0), -1);
  }
  imwrite(filename, clustered);
//  imshow("Clustered", clustered);
//  waitKey(0);
}
