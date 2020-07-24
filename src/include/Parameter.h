#ifndef ATLASSPHERE_PARAMETER_H
#define ATLASSPHERE_PARAMETER_H

#include "yaml-cpp/yaml.h"
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <fstream>
#include <iostream>
#include <ompl/util/Console.h>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <time.h>

namespace po = boost::program_options;

class Parameter {
public:
    Parameter(int argc, char **argv);

    bool set_space(std::string space) {
        boost::algorithm::to_lower(space);
        if (space == "proj" or space == "projection")
            this->space = proj;
        else if (space == "atlas")
            this->space = atlas;
        else if (space == "tb" or space == "tangent-bundle" or space == "tangent_bundle")
            this->space = tb;
        else
            OMPL_ERROR("Unsupported space type.");
        return true;
    }

    bool set_planner(std::string planner) {
        boost::algorithm::to_lower(planner);
        if (planner == "rrtconnect")
            this->planner = RRTConnect;
        else if (planner == "compnet")
            this->planner = CoMPNet;
        else if (planner == "rrtstar")
            this->planner = RRTstar;
        else
            OMPL_ERROR("Unsupported planner type.");
        return true;
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

    friend std::ostream &operator<<(std::ostream &o, SpaceType s);

    friend std::ostream &operator<<(std::ostream &o, PlannerType p);

    friend std::ostream &operator<<(std::ostream &o, const Parameter &p);
};

#endif // ATLASSPHERE_PARAMETER_H
