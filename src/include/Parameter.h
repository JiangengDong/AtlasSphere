#ifndef ATLASSPHERE_PARAMETER_H
#define ATLASSPHERE_PARAMETER_H

#include <fstream>
#include <iostream>
#include <ompl/util/Console.h>
#include <string>

class Parameter {
public:
    Parameter(std::istream &config_stream) {
        std::string key;
        while (!config_stream.eof()) {
            config_stream >> key;
            if (key == "num_iter") {
                config_stream >> num_iter;
            } else if (key == "space") {
                std::string temp;
                config_stream >> temp;
                if (temp == "proj")
                    space = proj;
                else if (temp == "atlas")
                    space = atlas;
                else if (temp == "tb")
                    space = tb;
            } else if (key == "planner") {
                std::string temp;
                config_stream >> temp;
                if (temp == "rrtconnect")
                    planner = RRTConnect;
                else if (temp == "compnet")
                    planner = CoMPNet;
                else if (temp == "rrtstar")
                    planner = RRTstar;
            } else if (key == "seed") {
                config_stream >> seed;
            } else if (key == "log_level") {
                std::string temp;
                config_stream >> temp;
                if (temp == "info") {
                    log_level = ompl::msg::LOG_INFO;
                } else if (temp == "debug") {
                    log_level = ompl::msg::LOG_DEBUG;
                } else if (temp == "error") {
                    log_level = ompl::msg::LOG_ERROR;
                }
            } else if (key == "csv_path") {
                config_stream >> csv_path;
            } else if (key == "path_path") {
                config_stream >> path_path;
            } else if (key == "graph_path") {
                config_stream >> graph_path;
            } else if (key == "atlas_path") {
                config_stream >> atlas_path;
            } else if (key== "pnet_path") {
                config_stream >> pnet_path;
            } else if (key=="voxel_path") {
                config_stream >> voxel_path;
            } else if (key=="smooth_path") {
                config_stream >> smooth_path;
            }
        }
    }

    unsigned int num_iter = 1;
    enum SpaceType { proj = 0,
                     atlas,
                     tb } space = proj;
    enum PlannerType { RRTConnect = 0,
                       CoMPNet,
                       RRTstar } planner = RRTConnect;
    unsigned int seed = 0;
    ompl::msg::LogLevel log_level = ompl::msg::LOG_DEBUG;
    std::string csv_path = "";
    std::string path_path = "";
    std::string graph_path = "";
    std::string atlas_path = "";
    std::string pnet_path = "";
    std::string voxel_path = "";
    std::string smooth_path = "";
};

#endif // ATLASSPHERE_PARAMETER_H