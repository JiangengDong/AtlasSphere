#include <Eigen/Dense>
#include <boost/format.hpp>
#include <boost/math/constants/constants.hpp>
#include <cnpy/cnpy.h>
#include <math.h>

Eigen::Matrix2Xd GenerateBrickConfig(unsigned int seed, unsigned int N) {
    srand(seed);
    Eigen::ArrayXXd phi(1, N), theta(1, N);
    phi.setRandom();
    phi = Eigen::acos(phi * 0.75);
    theta.setRandom();
    theta *= boost::math::double_constants::pi;

    Eigen::Matrix2Xd data(2, N);
    data.row(0) = phi;
    data.row(1) = theta;

    return data;
}

int main(int argc, char **argv) {
    for (unsigned int i = 40; i < 50; i++) {
        auto data = GenerateBrickConfig(i, 500);
        std::string brick_config_path = (boost::format("./data/brick_config/env%d.npy") % i).str();
        cnpy::npy_save(brick_config_path, data.data(), {500, 2}, "w");
    }
}
