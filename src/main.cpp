#include <Eigen/Dense>
#include "SpherePlanning.h"

int main(int argc, char **argv) {
    ompl::msg::setLogLevel(ompl::msg::LOG_INFO);
    SpherePlanning spherePlanning;

    std::vector<Eigen::Vector3d> collisions;
    int num_pairs = 200;
    Eigen::Vector3d start, goal;
    bool valid, solved;
    for(int i=0; i<num_pairs;i++) {
        while(true) {
            start.setRandom();
            start.normalize();
            valid = spherePlanning.isValid(start);
            if (valid) {
                break;
            } else {
                collisions.emplace_back(start);
            }
        }
        while(true) {
            goal.setRandom();
            goal.normalize();
            valid = spherePlanning.isValid(goal);
            if (valid) {
                break;
            } else {
                collisions.emplace_back(goal);
            }
        }
        std::cout << "Planning: " << i << std::endl;
        solved = spherePlanning.planOnce(start, goal);
        if(solved) {
            spherePlanning.exportPath("data/path.txt");
        }
        spherePlanning.clear();
    }
}