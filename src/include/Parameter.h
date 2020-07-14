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
    Parameter(int argc, char **argv) {
        auto pwd = boost::filesystem::current_path();

        po::positional_options_description p;
        p.add("task", 1);

        po::options_description desc;
        desc.add_options()                                                                                                                                                       //
            ("space", po::value<std::string>()->default_value("proj")->notifier([this](std::string space_str) { this->set_space(space_str); }), "type of constrained space")     //
            ("planner", po::value<std::string>()->default_value("rrtconnect")->notifier([this](std::string planner_str) { this->set_planner(planner_str); }), "type of planner") //
            ("num_iter", po::value<unsigned int>(&this->num_iter)->default_value(1), "num of iterations")                                                                        //
            ("seed", po::value<unsigned int>(&this->seed)->default_value(time(NULL)), "random seed")                                                                             //
            ("log_level", po::value<unsigned int>()->default_value(0)->notifier([this](unsigned int level) {                                                                     //
                this->log_level = static_cast<ompl::msg::LogLevel>(level);
            }))                                                                             //
            ("csv_path", po::value<std::string>()->notifier([this, &pwd](std::string csv) { //
                auto temp_path = boost::filesystem::path(csv);
                this->csv_path = temp_path.is_absolute() ? temp_path.string() : (pwd / temp_path).string();
            }));

        po::options_description cmdline;
        cmdline.add_options()                                                      //
            ("help", "produce help message")                                       //
            ("config_file", po::value<std::string>(), "type of constrained space") //
            ;
        cmdline.add(desc);

        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).options(cmdline).positional(p).run(), vm);
        if (vm.count("help")) {
            std::cout << cmdline << std::endl;
            exit(0);
        }
        if (vm.count("config_file")) {
            auto filename = vm["config_file"].as<std::string>();
            if (!boost::filesystem::is_regular_file(filename))
                OMPL_ERROR("The config file does not exist.");
            std::ifstream config_stream(filename);
            if (!config_stream)
                OMPL_ERROR("Unable to open the config file.");
            po::store(po::parse_config_file(config_stream, desc), vm);
            config_stream.close();
        }
        po::notify(vm);
    }

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
};

#endif // ATLASSPHERE_PARAMETER_H