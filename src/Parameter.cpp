#include "Parameter.h"

Parameter::Parameter(int argc, char **argv) {
    // define options
    po::options_description desc;
    desc.add_options()                                                                                                      //
        ("num_iter", po::value<unsigned int>()->default_value(1))                                                           //
        ("seed", po::value<unsigned int>()->default_value(time(NULL)))                                                      //
        ("log_level", po::value<unsigned int>()->default_value(0))                                                          //
        ("space", po::value<std::string>()->default_value("proj"), "type of constrained space: proj, atlas, tb")            //
        ("planner", po::value<std::string>()->default_value("rrtconnect"), "type of planner: rrtconnect, compnet, rrtstar") //
        ;
    po::options_description cmdline;
    cmdline.add_options()                                                                                             //
        ("help", "produce help message")                                                                              //
        ("config_file", po::value<std::string>())                                                                     //
        ("task", po::value<std::string>()->required(), "task to do: generate_path, generate_smooth, generate_visual") //
        ("output_dir", po::value<std::string>()->default_value("./"))                                                 //
        ("input_dir", po::value<std::string>()->default_value("./"))                                                  //
        ;
    cmdline.add(desc);

    // read conmand line
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, cmdline), vm);
    if (vm.count("help")) {
        std::cout << cmdline << std::endl;
        exit(0);
    }

    // read config file
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

    // store values into the instance
    this->set_task(vm["task"].as<std::string>());
    this->num_iter = vm["num_iter"].as<unsigned int>();
    this->seed = vm["seed"].as<unsigned int>();
    this->log_level = static_cast<ompl::msg::LogLevel>(vm["log_level"].as<unsigned int>());
    this->set_space(vm["space"].as<std::string>());
    this->set_planner(vm["planner"].as<std::string>());
    boost::filesystem::path output_dir(vm["output_dir"].as<std::string>()),
        input_dir(vm["input_dir"].as<std::string>()),
        pwd = boost::filesystem::current_path();
    this->output_dir = (output_dir.is_absolute() ? output_dir : pwd / output_dir).normalize();
    this->input_dir = (input_dir.is_absolute() ? input_dir : pwd / input_dir).normalize();
}

std::ostream &operator<<(std::ostream &o, Parameter::SpaceType s) {
    switch (s) {
    case Parameter::PROJ:
        o << "proj";
        break;
    case Parameter::ATLAS:
        o << "atlas";
        break;
    case Parameter::TB:
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
      << "planner=" << p.planner << std::endl
      << "input_dir=" << p.input_dir.string() << std::endl
      << "output_dir=" << p.output_dir.string();
    return o;
}
