//
// Created by jiangeng on 2/13/20.
//

#include "SpherePlanning.h"

#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/spaces/constraint/AtlasStateSpace.h>
#include <ompl/base/ConstrainedSpaceInformation.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/geometric/SimpleSetup.h>

#include "SphereConstraint.h"
#include "SphereValidityChecker.h"

SpherePlanning::SpherePlanning() {
    _space = std::make_shared<ompl::base::RealVectorStateSpace>(3);
    _constraint = std::make_shared<SphereConstraint>();
    _constrained_space = std::make_shared<ompl::base::AtlasStateSpace>(_space, _constraint);
    _constrained_space_info = std::make_shared<ompl::base::ConstrainedSpaceInformation>(_constrained_space);
    _state_validity_checker = std::make_shared<SphereValidityChecker>(_constrained_space_info);
    _planner = std::make_shared<ompl::geometric::RRTConnect>(_constrained_space_info);
    _simple_setup = std::make_shared<ompl::geometric::SimpleSetup>(_constrained_space_info);
    // TODO: add parameter setting

    _simple_setup->setPlanner(_planner);
    _simple_setup->setStateValidityChecker(_state_validity_checker);
}

bool SpherePlanning::planOnce(const Eigen::Ref<const Eigen::VectorXd> &start, const Eigen::Ref<const Eigen::VectorXd> &goal) {
    ompl::base::ScopedState<> sstart(_constrained_space);
    ompl::base::ScopedState<> sgoal(_constrained_space);

    sstart->as<ompl::base::ConstrainedStateSpace::StateType>()->copy(start);
    sgoal->as<ompl::base::ConstrainedStateSpace::StateType>()->copy(goal);

    _constrained_space->as<ompl::base::AtlasStateSpace>()->anchorChart(sstart.get());
    _constrained_space->as<ompl::base::AtlasStateSpace>()->anchorChart(sgoal.get());

    _simple_setup->setStartAndGoalStates(sstart, sgoal);
    _simple_setup->solve(5);
    // TODO: do something after a path is found
}

bool SpherePlanning::clear() {
    _constrained_space->clear();
    _simple_setup->clear();
}
