#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <vector>

#define DATA_SIZE 20
#define SIZE 20
#define SIZE_F 20.0
#define THRESHOLD 0.2

using namespace cv;
using namespace std;

/**
 * Merge image model into category model,
 * until the KL distance is less than a threshold.
 * Return the number of catergories
 */
int merge(vector<Mat> &categories, vector<Mat> &likelihoods, vector<vector<int> > &collections, EM em[]);

/**
 * Calculate KL distance between distributions,
 * used to determine either merging two models togather or not.
 * order of logL is logL1 + logL2
 */
double kl_distance(int total, Mat logL1, Mat logL2, Mat logL);

void draw(char* filename, Mat image, Mat label, Mat mean);
