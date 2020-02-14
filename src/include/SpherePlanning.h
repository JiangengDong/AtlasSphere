//
// Created by jiangeng on 2/13/20.
//

#ifndef ATLASSPHERE_SPHEREPLANNING_H
#define ATLASSPHERE_SPHEREPLANNING_H

#include <ompl/base/StateSpace.h>
#include <ompl/base/Constraint.h>
#include <ompl/base/spaces/constraint/ConstrainedStateSpace.h>
#include <ompl/base/spaces/constraint/AtlasStateSpace.h>
#include <ompl/base/ConstrainedSpaceInformation.h>
#include <ompl/base/StateValidityChecker.h>
#include <ompl/base/Planner.h>
#include <ompl/geometric/SimpleSetup.h>

class SpherePlanning {
public:
    SpherePlanning();

    // TODO: implement this
    bool planOnce(const Eigen::Ref<const Eigen::VectorXd> &start, const Eigen::Ref<const Eigen::VectorXd> &goal);

    bool clear();

private:
    ompl::base::StateSpacePtr _space;
    ompl::base::ConstraintPtr _constraint;
    ompl::base::ConstrainedStateSpacePtr _constrained_space;
    ompl::base::ConstrainedSpaceInformationPtr _constrained_space_info;
    ompl::base::StateValidityCheckerPtr _state_validity_checker;
    ompl::base::PlannerPtr _planner;
    ompl::geometric::SimpleSetupPtr _simple_setup;
};


#endif //ATLASSPHERE_SPHEREPLANNING_H
