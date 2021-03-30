//
// Created by jiangeng on 2/13/20.
//

#ifndef ATLASSPHERE_SPHEREVALIDITYCHECKER_H
#define ATLASSPHERE_SPHEREVALIDITYCHECKER_H

#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/StateValidityChecker.h>
#include <ompl/base/spaces/constraint/ConstrainedStateSpace.h>

class SphereValidityChecker : public ompl::base::StateValidityChecker {
public:
    explicit SphereValidityChecker(const ompl::base::SpaceInformationPtr &si) : StateValidityChecker(si) {
    }

    bool isValid(const ompl::base::State *state) const override {
        auto &&x = *state->as<ompl::base::ConstrainedStateSpace::StateType>();

        if (-0.80 < x[2] && x[2] < -0.6) {
            if (-0.05 < x[1] && x[1] < 0.05)
                return x[0] > 0;
            return false;
        } else if (-0.1 < x[2] && x[2] < 0.1) {
            if (-0.05 < x[0] && x[0] < 0.05)
                return x[1] < 0;
            return false;
        } else if (0.6 < x[2] && x[2] < 0.80) {
            if (-0.05 < x[1] && x[1] < 0.05)
                return x[0] < 0;
            return false;
        }

        return true;
    }
};

class BrickValidityChecker : public ompl::base::StateValidityChecker {
public:
    double brick_halfsize;
    std::vector<Eigen::Matrix3d> brick_rots;

    /* Create a brick environment from brick configs. Brick configs are the sphere coordinate the bricks. */
    BrickValidityChecker(const ompl::base::SpaceInformationPtr &si,
                         const Eigen::Matrix2Xd &brick_configs,
                         const double brick_size = 0.1) : StateValidityChecker(si) {
        this->brick_halfsize = brick_size / 2.0;

        auto N = brick_configs.cols();
        brick_rots.resize(N);

        for (size_t i = 0; i < N; i++) {
            const auto &phi = brick_configs(0, i);
            const auto &theta = brick_configs(1, i);

            // rotate the center of brick to (1, 0, 0)
            brick_rots[i] << sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi),
                -sin(theta), cos(theta), 0,
                -cos(phi) * cos(theta), -cos(phi) * sin(theta), sin(phi);
        }
    }

    bool isValid(const ompl::base::State *state) const override {
        auto &&x = *state->as<ompl::base::ConstrainedStateSpace::StateType>();

        const auto center = Eigen::Vector3d(1.0, 0.0, 0.0);
        auto x_transformed = Eigen::Vector3d();

        for (const auto &brick_rot : brick_rots) {
            x_transformed = brick_rot * x - center;
            // if inside a brick, the config is invalid
            if (x_transformed[0] > -brick_halfsize and
                x_transformed[0] < brick_halfsize and
                x_transformed[1] > -brick_halfsize and
                x_transformed[1] < brick_halfsize and
                x_transformed[2] > -brick_halfsize and
                x_transformed[2] < brick_halfsize) {
                return false;
            }
        }

        return true;
    }
};

#endif //ATLASSPHERE_SPHEREVALIDITYCHECKER_H
