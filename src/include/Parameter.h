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
            this->space = PROJ;
        else if (space == "atlas")
            this->space = ATLAS;
        else if (space == "tb" or space == "tangent-bundle" or space == "tangent_bundle")
            this->space = TB;
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

    bool set_task(std::string task) {
        boost::algorithm::to_lower(task);
        if(task == "generate_path")
            this->task = GENERATE_PATH;
        else if (task == "generate_smooth")
            this->task = GENERATE_SMOOTH;
        else if (task == "generate_visual")
            this->task = GENERATE_VISUAL;
        else
            OMPL_ERROR("Unsupported task.");
        return true;
    }

    unsigned int num_iter;
    enum SpaceType { PROJ = 0,
                     ATLAS,
                     TB } space;
    enum PlannerType { RRTConnect = 0,
                       CoMPNet,
                       RRTstar } planner;
    unsigned int seed;
    ompl::msg::LogLevel log_level;
    boost::filesystem::path input_dir;
    boost::filesystem::path output_dir;
    enum TaskType { GENERATE_PATH = 0, GENERATE_SMOOTH, GENERATE_VISUAL} task;

    friend std::ostream &operator<<(std::ostream &o, SpaceType s);

    friend std::ostream &operator<<(std::ostream &o, PlannerType p);

    friend std::ostream &operator<<(std::ostream &o, const Parameter &p);
};

#endif // ATLASSPHERE_PARAMETER_H
