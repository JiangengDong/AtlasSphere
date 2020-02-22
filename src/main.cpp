#include <Eigen/Dense>
#include "SpherePlanning.h"

int main(int argc, char **argv) {
    ompl::msg::setLogLevel(ompl::msg::LOG_INFO);
    SpherePlanning spherePlanning;
    Eigen::Vector3d start, goal;

    do {
        while (true) {
            start.setRandom();
            start.normalize();
            if (spherePlanning.isValid(start))
                break;
        }
        while (true) {
            goal.setRandom();
            goal.normalize();
            if (spherePlanning.isValid(goal))
                break;
        }
    } while ((start-goal).norm() < 1.9);

    bool solved = spherePlanning.planOnce(start, goal);
    if (solved) {
        spherePlanning.exportAll();
    }
    spherePlanning.clear();
}