#include <Eigen/Dense>
#include "SpherePlanning.h"

int main(int argc, char **argv) {
    ompl::msg::setLogLevel(ompl::msg::LOG_DEBUG);
    SpherePlanning spherePlanning;
    Eigen::Vector3d start, goal;
    start(2) = 1;
    goal(2) = -1;
    bool solved = spherePlanning.planOnce(start, goal);

    if(solved) {
        spherePlanning.exportAtlas("data/atlas0");
        spherePlanning.exportTree("data/tree0");
    }
    spherePlanning.clear();
}
