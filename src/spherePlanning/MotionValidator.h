#ifndef ATLASSPHERE_MOTIONVALIDATOR_H
#define ATLASSPHERE_MOTIONVALIDATOR_H

#include <Eigen/Dense>
#include <ompl/base/MotionValidator.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/State.h>
#include <ompl/base/spaces/constraint/ConstrainedStateSpace.h>

class BrickMotionValidator : public ompl::base::MotionValidator {
public:
    double brick_halfsize;
    std::vector<Eigen::Matrix3d> brick_rots;

    /* Create a brick environment from brick configs. Brick configs are the sphere coordinate the bricks. */
    BrickMotionValidator(const ompl::base::SpaceInformationPtr &si,
                         const Eigen::Matrix2Xd &brick_configs,
                         const double brick_size = 0.1) : MotionValidator(si) {
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

    // s1 and s2 are assumed valid
    bool checkMotion(const ompl::base::State *s1, const ompl::base::State *s2) const override {
        const auto center = Eigen::Vector3d(1.0, 0.0, 0.0);
        Eigen::Vector3d s1_, s2_;

        for (const auto &brick_rot : brick_rots) {
            s1_ = brick_rot * (*s1->as<ompl::base::ConstrainedStateSpace::StateType>()) - center;
            s2_ = brick_rot * (*s2->as<ompl::base::ConstrainedStateSpace::StateType>()) - center;
            if (s1_[0] > -brick_halfsize and
                s1_[0] < brick_halfsize and
                s1_[1] > -brick_halfsize and
                s1_[1] < brick_halfsize and
                s1_[2] > -brick_halfsize and
                s1_[2] < brick_halfsize) {
                return false;
            }
            if (s2_[0] > -brick_halfsize and
                s2_[0] < brick_halfsize and
                s2_[1] > -brick_halfsize and
                s2_[1] < brick_halfsize and
                s2_[2] > -brick_halfsize and
                s2_[2] < brick_halfsize) {
                return false;
            }
            const double dx = s2_[0] - s1_[0], dy = s2_[1] - s1_[1], dz = s2_[2] - s1_[2];
            const unsigned int code = ((abs(dx) < 1e-5) ? 0b100 : 0b000) | ((abs(dy) < 1e-5) ? 0b10 : 0b00) | ((abs(dz) < 1e-5) ? 0b1 : 0b0);
            switch (code) {
                case 0b111: {
                    break;
                }
                case 0b110: {
                    if ((s1_[2] > brick_halfsize && s2_[2] > brick_halfsize) ||
                        (s1_[2] < brick_halfsize && s2_[2] < brick_halfsize)) {
                        break;
                    } else {
                        return false;
                    }
                }
                case 0b101: {
                    if ((s1_[1] > brick_halfsize && s2_[1] > brick_halfsize) ||
                        (s1_[1] < brick_halfsize && s2_[1] < brick_halfsize)) {
                        break;
                    } else {
                        return false;
                    }
                }
                case 0b011: {
                    if ((s1_[0] > brick_halfsize && s2_[0] > brick_halfsize) ||
                        (s1_[0] < brick_halfsize && s2_[0] < brick_halfsize)) {
                        break;
                    } else {
                        return false;
                    }
                }
                case 0b100: {
                    const double y_lower = (-s1_[1] - brick_halfsize) / dy,
                                 y_upper = (-s1_[1] + brick_halfsize) / dy,
                                 z_lower = (-s1_[2] - brick_halfsize) / dz,
                                 z_upper = (-s1_[2] + brick_halfsize) / dz;
                    if (std::max(y_lower, z_lower) < std::min(y_upper, z_upper)) {
                        return false;
                    } else {
                        break;
                    }
                }
                case 0b010: {
                    const double x_lower = (-s1_[0] - brick_halfsize) / dx,
                                 x_upper = (-s1_[0] + brick_halfsize) / dx,
                                 z_lower = (-s1_[2] - brick_halfsize) / dz,
                                 z_upper = (-s1_[2] + brick_halfsize) / dz;
                    if (std::max(x_lower, z_lower) < std::min(x_upper, z_upper)) {
                        return false;
                    } else {
                        break;
                    }
                }
                case 0b001: {
                    const double x_lower = (-s1_[0] - brick_halfsize) / dx,
                                 x_upper = (-s1_[0] + brick_halfsize) / dx,
                                 y_lower = (-s1_[1] - brick_halfsize) / dy,
                                 y_upper = (-s1_[1] + brick_halfsize) / dy;
                    if (std::max(y_lower, x_lower) < std::min(y_upper, x_upper)) {
                        return false;
                    } else {
                        break;
                    }
                }
                case 0b000: {
                    const double x_lower = (-s1_[0] - brick_halfsize) / dx,
                                 x_upper = (-s1_[0] + brick_halfsize) / dx,
                                 y_lower = (-s1_[1] - brick_halfsize) / dy,
                                 y_upper = (-s1_[1] + brick_halfsize) / dy,
                                 z_lower = (-s1_[2] - brick_halfsize) / dz,
                                 z_upper = (-s1_[2] + brick_halfsize) / dz;
                    if (std::max(std::max(x_lower, y_lower), z_lower) < std::min(std::min(x_upper, y_upper), z_upper)) {
                        return false;
                    } else {
                        break;
                    }
                }
                default: {
                    break;
                }
            }
        }

        return true;
    }

    bool checkMotion(const ompl::base::State *s1, const ompl::base::State *s2, std::pair<ompl::base::State *, double> &lastValid) const override {
        const auto center = Eigen::Vector3d(1.0, 0.0, 0.0);
        Eigen::Vector3d s1_, s2_;

        for (const auto &brick_rot : brick_rots) {
            s1_ = brick_rot * (*s1->as<ompl::base::ConstrainedStateSpace::StateType>()) - center;
            s2_ = brick_rot * (*s2->as<ompl::base::ConstrainedStateSpace::StateType>()) - center;
            if (s1_[0] > -brick_halfsize and
                s1_[0] < brick_halfsize and
                s1_[1] > -brick_halfsize and
                s1_[1] < brick_halfsize and
                s1_[2] > -brick_halfsize and
                s1_[2] < brick_halfsize) {
                return false;
            }
            if (s2_[0] > -brick_halfsize and
                s2_[0] < brick_halfsize and
                s2_[1] > -brick_halfsize and
                s2_[1] < brick_halfsize and
                s2_[2] > -brick_halfsize and
                s2_[2] < brick_halfsize) {
                return false;
            }
            const double dx = s2_[0] - s1_[0], dy = s2_[1] - s1_[1], dz = s2_[2] - s1_[2];
            const unsigned int code = ((abs(dx) < 1e-5) ? 0b100 : 0b000) | ((abs(dy) < 1e-5) ? 0b10 : 0b00) | ((abs(dz) < 1e-5) ? 0b1 : 0b0);
            switch (code) {
                case 0b111: {
                    break;
                }
                case 0b110: {
                    if ((s1_[2] > brick_halfsize && s2_[2] > brick_halfsize) ||
                        (s1_[2] < brick_halfsize && s2_[2] < brick_halfsize)) {
                        break;
                    } else {
                        return false;
                    }
                }
                case 0b101: {
                    if ((s1_[1] > brick_halfsize && s2_[1] > brick_halfsize) ||
                        (s1_[1] < brick_halfsize && s2_[1] < brick_halfsize)) {
                        break;
                    } else {
                        return false;
                    }
                }
                case 0b011: {
                    if ((s1_[0] > brick_halfsize && s2_[0] > brick_halfsize) ||
                        (s1_[0] < brick_halfsize && s2_[0] < brick_halfsize)) {
                        break;
                    } else {
                        return false;
                    }
                }
                case 0b100: {
                    const double y_lower = (-s1_[1] - brick_halfsize) / dy,
                                 y_upper = (-s1_[1] + brick_halfsize) / dy,
                                 z_lower = (-s1_[2] - brick_halfsize) / dz,
                                 z_upper = (-s1_[2] + brick_halfsize) / dz;
                    if (std::max(y_lower, z_lower) < std::min(y_upper, z_upper)) {
                        return false;
                    } else {
                        break;
                    }
                }
                case 0b010: {
                    const double x_lower = (-s1_[0] - brick_halfsize) / dx,
                                 x_upper = (-s1_[0] + brick_halfsize) / dx,
                                 z_lower = (-s1_[2] - brick_halfsize) / dz,
                                 z_upper = (-s1_[2] + brick_halfsize) / dz;
                    if (std::max(x_lower, z_lower) < std::min(x_upper, z_upper)) {
                        return false;
                    } else {
                        break;
                    }
                }
                case 0b001: {
                    const double x_lower = (-s1_[0] - brick_halfsize) / dx,
                                 x_upper = (-s1_[0] + brick_halfsize) / dx,
                                 y_lower = (-s1_[1] - brick_halfsize) / dy,
                                 y_upper = (-s1_[1] + brick_halfsize) / dy;
                    if (std::max(y_lower, x_lower) < std::min(y_upper, x_upper)) {
                        return false;
                    } else {
                        break;
                    }
                }
                case 0b000: {
                    const double x_lower = (-s1_[0] - brick_halfsize) / dx,
                                 x_upper = (-s1_[0] + brick_halfsize) / dx,
                                 y_lower = (-s1_[1] - brick_halfsize) / dy,
                                 y_upper = (-s1_[1] + brick_halfsize) / dy,
                                 z_lower = (-s1_[2] - brick_halfsize) / dz,
                                 z_upper = (-s1_[2] + brick_halfsize) / dz;
                    if (std::max(std::max(x_lower, y_lower), z_lower) < std::min(std::min(x_upper, y_upper), z_upper)) {
                        return false;
                    } else {
                        break;
                    }
                }
                default: {
                    break;
                }
            }
        }

        return true;
    }
};

#endif //ATLASSPHERE_MOTIONVALIDATOR_H
