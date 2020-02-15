//
// Created by jiangeng on 2/13/20.
//

#ifndef ATLASSPHERE_SPHERECONSTRAINT_H
#define ATLASSPHERE_SPHERECONSTRAINT_H

#include <ompl/base/Constraint.h>

class SphereConstraint : public ompl::base::Constraint {
public:
    SphereConstraint() : Constraint(3, 1) {
    }

    void function(const Eigen::Ref<const Eigen::VectorXd> &x, Eigen::Ref<Eigen::VectorXd> out) const override {
        out[0] = x.squaredNorm() - 1;
    }

    void jacobian(const Eigen::Ref<const Eigen::VectorXd> &x, Eigen::Ref<Eigen::MatrixXd> out) const override {
        out = 2*x.transpose();
    }
};


#endif //ATLASSPHERE_SPHERECONSTRAINT_H
