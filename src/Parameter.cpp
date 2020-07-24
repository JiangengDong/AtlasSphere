#include "Parameter.h"

Parameter::Parameter(int argc, char **argv) {
    auto pwd = boost::filesystem::current_path();

    po::positional_options_description p;
    p.add("task", 1);

    po::options_description desc;
    desc.add_options()                                                                                   //
        ("num_iter", po::value<unsigned int>(&this->num_iter)->default_value(1), "num of iterations")    //
        ("seed", po::value<unsigned int>(&this->seed)->default_value(time(NULL)), "random seed")         //
        ("log_level", po::value<unsigned int>()->default_value(0)->notifier([this](unsigned int level) { //
            this->log_level = static_cast<ompl::msg::LogLevel>(level);
        }))                                                                                                                                                                  //
        ("space", po::value<std::string>()->default_value("proj")->notifier([this](std::string space_str) { this->set_space(space_str); }), "type of constrained space")     //
        ("planner", po::value<std::string>()->default_value("rrtconnect")->notifier([this](std::string planner_str) { this->set_planner(planner_str); }), "type of planner") //
        ("output.csv_path", po::value<std::string>()->notifier([this, &pwd](std::string value) {                                                                             //
            if (value == "") return;
            auto temp_path = boost::filesystem::path(value);
            this->csv_path = temp_path.is_absolute() ? temp_path.string() : (pwd / temp_path).normalize().string();
        }))                                                                                       //
        ("output.path_path", po::value<std::string>()->notifier([this, &pwd](std::string value) { //
            if (value == "") return;
            auto temp_path = boost::filesystem::path(value);
            this->path_path = temp_path.is_absolute() ? temp_path.string() : (pwd / temp_path).normalize().string();
        }))                                                                                        //
        ("output.graph_path", po::value<std::string>()->notifier([this, &pwd](std::string value) { //
            if (value == "") return;
            auto temp_path = boost::filesystem::path(value);
            this->graph_path = temp_path.is_absolute() ? temp_path.string() : (pwd / temp_path).normalize().string();
        }))                                                                                        //
        ("output.atlas_path", po::value<std::string>()->notifier([this, &pwd](std::string value) { //
            if (value == "") return;
            auto temp_path = boost::filesystem::path(value);
            this->atlas_path = temp_path.is_absolute() ? temp_path.string() : (pwd / temp_path).normalize().string();
        }))                                                                                         //
        ("output.smooth_path", po::value<std::string>()->notifier([this, &pwd](std::string value) { //
            if (value == "") return;
            auto temp_path = boost::filesystem::path(value);
            this->smooth_path = temp_path.is_absolute() ? temp_path.string() : (pwd / temp_path).normalize().string();
        }))                                                                                      //
        ("input.pnet_path", po::value<std::string>()->notifier([this, &pwd](std::string value) { //
            if (value == "") return;
            auto temp_path = boost::filesystem::path(value);
            this->pnet_path = temp_path.is_absolute() ? temp_path.string() : (pwd / temp_path).normalize().string();
        }))                                                                                       //
        ("input.voxel_path", po::value<std::string>()->notifier([this, &pwd](std::string value) { //
            if (value == "") return;
            auto temp_path = boost::filesystem::path(value);
            this->voxel_path = temp_path.is_absolute() ? temp_path.string() : (pwd / temp_path).normalize().string();
        })) //
        ;

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
        auto config_dir = boost::filesystem::path(filename).parent_path();
        pwd = config_dir.is_absolute()?config_dir.string():(pwd/config_dir).normalize().string();
    }
    po::notify(vm);
}

std::ostream &operator<<(std::ostream &o, Parameter::SpaceType s) {
    switch (s) {
    case Parameter::proj:
        o << "proj";
        break;
    case Parameter::atlas:
        o << "atlas";
        break;
    case Parameter::tb:
        o << "tb";
        break;
    default:
        break;
    }
    return o;
}

std::ostream &operator<<(std::ostream &o, Parameter::PlannerType p) {
    switch (p) {
    case Parameter::RRTConnect:
        o << "RRTConnect";
        break;
    case Parameter::CoMPNet:
        o << "CoMPNet";
        break;
    case Parameter::RRTstar:
        o << "RRTstar";
        break;
    default:
        break;
    }
    return o;
}

std::ostream &operator<<(std::ostream &o, const Parameter &p) {
    o << "num_iter=" << p.num_iter << std::endl
      << "seed=" << p.seed << std::endl
      << "log_level=" << p.log_level << std::endl
      << "space=" << p.space << std::endl
      << "planner=" << p.planner << std::endl;

    o << std::endl
      << "[output]" << std::endl
      << "csv_path=" << p.csv_path << std::endl
      << "path_path=" << p.path_path << std::endl
      << "graph_path=" << p.graph_path << std::endl
      << "atlas_path=" << p.atlas_path << std::endl
      << "smooth_path=" << p.smooth_path << std::endl;

    o << std::endl
      << "[input]" << std::endl
      << "pnet_path=" << p.pnet_path << std::endl
      << "voxel_path=" << p.voxel_path;
    return o;
}
