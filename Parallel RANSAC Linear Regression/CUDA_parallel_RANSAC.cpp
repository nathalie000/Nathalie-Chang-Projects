#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

using namespace std;

// Structure to represent a 2D point
struct Point2D {
    double x, y;

    Point2D(double x, double y) : x(x), y(y) {}
};

// Function to perform 2D linear regression using RANSAC
pair<double, double> linearRegressionRANSAC(const vector<Point2D>& points, int iterations, double threshold) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(0, points.size() - 1);

    double best_slope, best_intercept;
    int best_inliers = 0;

    for (int i = 0; i < iterations; ++i) {
        // Randomly select two points
        int idx1 = dist(gen);
        int idx2 = dist(gen);

        double x1 = points[idx1].x;
        double y1 = points[idx1].y;
        double x2 = points[idx2].x;
        double y2 = points[idx2].y;

        // Calculate slope and intercept
        double slope = (y2 - y1) / (x2 - x1);
        double intercept = y1 - slope * x1;

        // Count inliers
        int inliers = 0;
        for (const auto& point : points) {
            double expected_y = slope * point.x + intercept;
            double error = fabs(point.y - expected_y);

            if (error < threshold) {
                inliers++;
            }
        }

        // Update best model if we found more inliers
        if (inliers > best_inliers) {
            best_inliers = inliers;
            best_slope = slope;
            best_intercept = intercept;
        }
    }

    return make_pair(best_slope, best_intercept);
}

int main() {
    // Generate some example data
    vector<Point2D> points;
    for (double x = 0.0; x < 10.0; x += 0.5) {
        double y = 2.0 * x + 1.0 + (rand() % 20 - 10) / 10.0; // Adding noise
        points.emplace_back(x, y);
    }

    // Perform linear regression using RANSAC
    int iterations = 1000;
    double threshold = 1.0;
    auto result = linearRegressionRANSAC(points, iterations, threshold);

    // Display the result
    cout << "Estimated Slope: " << result.first << endl;
    cout << "Estimated Intercept: " << result.second << endl;

    return 0;
}
