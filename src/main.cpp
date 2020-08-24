#include "Parameter.h"
#include "SpherePlanning.h"
#include "MPNetPlanner.h"
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
    ompl::msg::setLogLevel(ompl::msg::LOG_NONE);
    Parameter param;
    param.planner = Parameter::RRTstar;
    param.space = Parameter::ATLAS;

    srand(0);
    SpherePlanning instance(param);
    Eigen::Vector3d start, goal;

    std::ofstream len_file("/workspaces/AtlasSphere/data/result/rrtstar-mpnet-time.csv");
    for (size_t iter=0; iter<50; iter++){
        // sample start and goal
        do{
            start = Eigen::Vector3d::Random();
            start /= start.norm();
        }while(!instance.isValid(start));
        do {
            goal = Eigen::Vector3d::Random();
            goal /= goal.norm();
        }while(!instance.isValid(goal));
        instance.setStartGoal(start, goal);

        // repeat 2 sec planning for 10 times
        ompl::base::PlannerStatus status;
        double total_time=0;
        double length = std::numeric_limits<double>::infinity();
        std::cout << "experiment " << iter << std::endl;
        for (size_t i = 0; i< 5; i++){
            status = instance._simple_setup->solve(2);
            total_time += instance._simple_setup->getLastPlanComputationTime();
            if (status == ompl::base::PlannerStatus::EXACT_SOLUTION) {
                length = instance._simple_setup->getSolutionPath().length();
            }
            std::cout << "\ttime: " << total_time << " length: " << length << std::endl;
            len_file << (status == ompl::base::PlannerStatus::EXACT_SOLUTION?total_time:std::numeric_limits<double>::infinity()) << ", " << length;
        }
        len_file << std::endl;
        instance.clear();
    }
    len_file.close();
}
