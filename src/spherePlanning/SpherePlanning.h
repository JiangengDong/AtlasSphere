//
// Created by jiangeng on 2/13/20.
//

#ifndef ATLASSPHERE_SPHEREPLANNING_H
#define ATLASSPHERE_SPHEREPLANNING_H

#include <ompl/base/ConstrainedSpaceInformation.h>
#include <ompl/base/Constraint.h>
#include <ompl/base/Planner.h>
#include <ompl/base/PlannerData.h>
#include <ompl/base/StateSpace.h>
#include <ompl/base/StateValidityChecker.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/spaces/constraint/AtlasStateSpace.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>

#include "Parameter.h"

class SpherePlanning {
public:
    typedef std::shared_ptr<SpherePlanning> Ptr;
    SpherePlanning(Parameter param);

    bool planOnce(const Eigen::Ref<const Eigen::VectorXd> &start, const Eigen::Ref<const Eigen::VectorXd> &goal);

    bool clear();

    Eigen::MatrixXd getPath() const;

    Eigen::MatrixXd getSmoothPath(unsigned int n_smooth = 100) const;

    double getPathLength() const;

    double getTime() const;

    bool isValid(const Eigen::Ref<const Eigen::VectorXd> &state) const;

    bool init;

    // private:
    std::shared_ptr<ompl::base::RealVectorStateSpace> _space;
    ompl::base::ConstraintPtr _constraint;
    ompl::base::ConstrainedStateSpacePtr _constrained_space;
    ompl::base::ConstrainedSpaceInformationPtr _constrained_space_info;
    ompl::base::StateValidityCheckerPtr _state_validity_checker;
    ompl::base::PlannerPtr _planner;
    ompl::geometric::SimpleSetupPtr _simple_setup;
    Parameter _param;

    // temporary variable
    ompl::base::PlannerDataPtr _planner_data;

    bool setAmbientStateSpace();

    bool setConstraint();

    bool setConstrainedStateSpace();

    bool setStateValidityChecker();

    bool setPlanner();

    bool setSimpleSetup();

    bool setStartGoal(const Eigen::Ref<const Eigen::VectorXd> &start, const Eigen::Ref<const Eigen::VectorXd> &goal);
};

#endif //ATLASSPHERE_SPHEREPLANNING_H
