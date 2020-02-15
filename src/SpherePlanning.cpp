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
    _space->as<ompl::base::RealVectorStateSpace>()->setBounds(-2, 2);

    _constraint = std::make_shared<SphereConstraint>();
    _constrained_space = std::make_shared<ompl::base::AtlasStateSpace>(_space, _constraint);
    _constrained_space_info = std::make_shared<ompl::base::ConstrainedSpaceInformation>(_constrained_space);
    _constraint->setTolerance(1e-4);
    _constraint->setMaxIterations(50);

    _constrained_space->setDelta(0.05);
    _constrained_space->setLambda(2.0);

    _constrained_space->setRho(0.2);
    _constrained_space->setEpsilon(0.001);
    _constrained_space->setAlpha(0.3);
    _constrained_space->setExploration(0.5);
    _constrained_space->setMaxChartsPerExtension(500);
    _constrained_space->setSeparated(true);
    auto &&atlas = _constrained_space;
    _constrained_space->setBiasFunction([atlas](ompl::base::AtlasChart *c) -> double {
        return (atlas->getChartCount() - c->getNeighborCount()) + 1;
    });

    _state_validity_checker = std::make_shared<SphereValidityChecker>(_constrained_space_info);
    _planner = std::make_shared<ompl::geometric::RRTConnect>(_constrained_space_info);
    _planner->setRange(0.05);
    _simple_setup = std::make_shared<ompl::geometric::SimpleSetup>(_constrained_space_info);

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
    _simple_setup->setup();
    _simple_setup->solve(10);
    // TODO: do something after a path is found
}

bool SpherePlanning::clear() {
    _constrained_space->clear();
    _simple_setup->clear();
}
