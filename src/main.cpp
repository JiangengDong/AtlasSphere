#include "Parameter.h"
#include "SpherePlanning.h"
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
    // get config
    std::string config_path = "./data/config/compnet-atlas-config.txt";
    if (argc > 1) {
        config_path = argv[1];
    }
    std::ifstream param_file(config_path);
    if(!param_file.good()){
        std::cerr << "Cannot find config file. " << std::endl;
        return 0;
    }
    Parameter param(param_file);
    param_file.close();

    srand(param.seed);
    ompl::msg::setLogLevel(param.log_level);

    std::ofstream output_csv;
    if (param.csv_path != "") {
        output_csv.open(param.csv_path);
        output_csv << "no, time, cost" << std::endl;
    }

    // main sampling part
    for (unsigned int i = 0; i < param.num_iter; i++) {
        std::cout << "Sample " << i << std::endl;
        SpherePlanning spherePlanning(param);
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
        } while ((start - goal).norm() < 1.9);

        bool solved = spherePlanning.planOnce(start, goal);
        double planning_time = solved ? spherePlanning.getTime() : INFINITY;
        if (output_csv.is_open()) {
            output_csv << i << ", " << planning_time << ", " << spherePlanning.getPathLength() << std::endl;
            std::cout << "Result save to csv. " << std::endl;
        }

        if (param.path_path != ""){
            spherePlanning.exportPath(param.path_path+std::to_string(i));
        }
        if (param.smooth_path != "") {
            spherePlanning.exportSmoothPath(param.smooth_path+std::to_string(i));
        }
        if (param.graph_path != "") {
            spherePlanning.exportTree(param.graph_path+std::to_string(i));
        }
        if(param.atlas_path != "") {
            spherePlanning.exportAtlas(param.atlas_path+std::to_string(i));
        }
    }
    if (output_csv.is_open())
        output_csv.close();
}