#include "Parameter.h"
#include "SpherePlanning.h"
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
    // get config
    Parameter param(argc, argv);
    std::cout << param << std::endl;
}
