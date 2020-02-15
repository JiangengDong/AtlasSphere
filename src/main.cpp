#include <Eigen/Dense>
#include "SpherePlanning.h"

int main(int argc, char **argv) {
    ompl::msg::setLogLevel(ompl::msg::LOG_INFO);
    SpherePlanning a;
    Eigen::Vector3d start, goal;
    start(2) = 1;
    goal(2) = -1;
    a.planOnce(start, goal);
}
