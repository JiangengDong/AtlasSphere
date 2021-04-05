#ifndef ATLASSPHERE_PARAMETER_H
#define ATLASSPHERE_PARAMETER_H

#include <Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>
#include <ompl/util/Console.h>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <torch/script.h>

class Parameter {
public:
    enum SpaceType { PROJ = 0,
                     ATLAS,
                     TB } space;
    enum PlannerType { RRTConnect = 0,
                       CoMPNet,
                       RRTstar,
                       BITstar,
                       FMT } planner;

    bool is_brick_env = true;
    Eigen::Matrix2Xd brick_configs;

    torch::jit::script::Module pnet;
    torch::Tensor voxel;

    Parameter() = default;
};

#endif // ATLASSPHERE_PARAMETER_H
