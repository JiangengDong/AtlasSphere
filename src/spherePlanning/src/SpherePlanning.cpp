//
// Created by jiangeng on 2/13/20.
//

#include "SpherePlanning.h"

#include <algorithm>
#include <fstream>
#include <memory>
#include <ompl/base/ConstrainedSpaceInformation.h>
#include <ompl/base/PlannerData.h>
#include <ompl/base/PlannerStatus.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/spaces/constraint/AtlasStateSpace.h>
#include <ompl/base/spaces/constraint/ProjectedStateSpace.h>
#include <ompl/base/spaces/constraint/TangentBundleStateSpace.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/prm/PRMstar.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/util/Console.h>

#include "MotionValidator.h"
#include "Parameter.h"
#include "SphereConstraint.h"
#include "SphereValidityChecker.h"
#include "planner/MPNetPlanner.h"

SpherePlanning::SpherePlanning(Parameter param) : _param(param) {
    init = setAmbientStateSpace() &&
           setConstraint() &&
           setConstrainedStateSpace() &&
           setStateValidityChecker() &&
           setMotionValidator() &&
           setPlanner() &&
           setSimpleSetup();
    if (!init) {
        OMPL_ERROR("!!! Construction failed!!!");
    } else {
        _planner_data = std::make_shared<ompl::base::PlannerData>(_constrained_space_info);
    }
}

bool SpherePlanning::planOnce(const Eigen::Ref<const Eigen::VectorXd> &start,
                              const Eigen::Ref<const Eigen::VectorXd> &goal) {
    if (!init) {
        OMPL_ERROR("This planner is not valid!");
        return false;
    }

    setStartGoal(start, goal);
    ompl::base::PlannerStatus status = ompl::base::PlannerStatus::TIMEOUT;
    status = _simple_setup->solve(10);
    _simple_setup->getPlannerData(*_planner_data);
    if (status == ompl::base::PlannerStatus::EXACT_SOLUTION) {
        return true;
    } else {
        return false;
    }
    // TODO: different logic for RRT*
}

bool SpherePlanning::clear() {
    _constrained_space->clear();
    _planner->clear();
    _simple_setup->clear();
    return true;
}

/** Caution: the matrix returned is column-major, i.e. `M[i, j] = *(M.data() + i
 * + j * num_rows)`. Hence the returned shape is `3*N`. */
Eigen::MatrixXd SpherePlanning::getPath() const {
    auto &path = _simple_setup->getSolutionPath();
    std::size_t n = path.getStateCount();
    Eigen::MatrixXd pathM(3, n);
    std::vector<double> temp;
    for (std::size_t i = 0; i < n; i++) {
        _constrained_space->copyToReals(temp, path.getState(i));
        pathM(0, i) = temp[0];
        pathM(1, i) = temp[1];
        pathM(2, i) = temp[2];
    }
    return pathM;
}

/** Caution: the matrix returned is column-major, i.e. `M[i, j] = *(M.data() + i
 * + j * num_rows)`. Hence the returned shape is `3*N`. */
Eigen::MatrixXd SpherePlanning::getSmoothPath(unsigned int n_smooth) const {
    OMPL_INFORM("Smoothing the path.");
    auto constrained_space = _constrained_space->as<ompl::base::ConstrainedStateSpace>();
    const auto &path = _simple_setup->getSolutionPath();
    std::vector<ompl::base::State *> old_path;
    std::vector<ompl::base::State *> new_path;
    std::vector<ompl::base::State *> geodesic;
    // copy old path to avoid double free
    old_path.resize(path.getStateCount());
    for (unsigned int idx_copy = 0; idx_copy < path.getStateCount(); idx_copy++) {
        old_path[idx_copy] = constrained_space->allocState();
        constrained_space->copyState(old_path[idx_copy], path.getState(idx_copy));
    }
    // smooth: try to find shortcut between any two states on the path
    for (unsigned int idx_smooth = 0; idx_smooth < n_smooth; idx_smooth++) {
        new_path.clear();
        unsigned int n = old_path.size();
        unsigned int idx_Start = 0;
        unsigned int idx_Goal;
        while (idx_Start < n - 1) {
            for (idx_Goal = n - 1; idx_Goal > idx_Start; idx_Goal--) {
                bool connected = constrained_space->discreteGeodesic(old_path[idx_Start], old_path[idx_Goal], false, &geodesic);
                if (connected) {
                    bool still_connected = true;
                    for (unsigned int i = 0; i < geodesic.size() - 1; i++) {
                        still_connected &= _constrained_space_info->checkMotion(geodesic[i], geodesic[i + 1]);
                    }
                    if (still_connected) {
                        double distance = 0;
                        double new_distance = 0;
                        for (unsigned int idx_Temp = idx_Start; idx_Temp < idx_Goal; idx_Temp++) {
                            distance += constrained_space->distance(old_path[idx_Temp], old_path[idx_Temp + 1]);
                        }
                        for (unsigned int idx_Temp = 0; idx_Temp < geodesic.size() - 1; idx_Temp++) {
                            new_distance += constrained_space->distance(geodesic[idx_Temp], geodesic[idx_Temp + 1]);
                        }
                        if (new_distance < distance - 1e-5) {
                            // always copy a state, so that there will be no memory leak
                            for (const auto &geo_state : geodesic) {
                                auto new_state = constrained_space->allocState();
                                constrained_space->copyState(new_state, geo_state);
                                new_path.emplace_back(new_state);
                            }
                            for (const auto &geo_state : geodesic) {
                                constrained_space->freeState(geo_state);
                            }
                            geodesic.clear();
                            break;
                        }
                    }
                }
                // free geodesic
                for (const auto &geo_state : geodesic) {
                    constrained_space->freeState(geo_state);
                }
                geodesic.clear();
            }
            if (idx_Goal == idx_Start) { // no shortcut found
                auto new_state = constrained_space->allocState();
                constrained_space->copyState(new_state, old_path[idx_Start]);
                new_path.emplace_back(new_state);
                idx_Start++;
            } else { // find shortcut
                idx_Start = idx_Goal;
            }
        }
        // ensure the last state in the new path is goal
        auto new_state = constrained_space->allocState();
        constrained_space->copyState(new_state, old_path.back());
        new_path.emplace_back(new_state);
        // free old path
        for (const auto &old_state : old_path) {
            constrained_space->freeState(old_state);
        }
        old_path = new_path;
    }

    std::size_t n = new_path.size();
    Eigen::MatrixXd pathM(3, n);
    std::vector<double> temp;
    for (std::size_t i = 0; i < n; i++) {
        _constrained_space->copyToReals(temp, new_path[i]);
        pathM(0, i) = temp[0];
        pathM(1, i) = temp[1];
        pathM(2, i) = temp[2];
    }

    // free new path
    for (const auto &new_state : new_path) {
        constrained_space->freeState(new_state);
    }
    OMPL_DEBUG("Smoothing done.");
    return pathM;
}

double SpherePlanning::getPathLength() const {
    return _simple_setup->getSolutionPath().length();
}

double SpherePlanning::getTime() const {
    return _simple_setup->getLastPlanComputationTime();
}

bool SpherePlanning::isValid(
    const Eigen::Ref<const Eigen::VectorXd> &state) const {
    ompl::base::ScopedState<> sstate(_constrained_space);

    sstate->as<ompl::base::ConstrainedStateSpace::StateType>()->copy(state);
    return _state_validity_checker->isValid(sstate.get());
}

/* private method below */

bool SpherePlanning::setAmbientStateSpace() {
    _space = std::make_shared<ompl::base::RealVectorStateSpace>(3);
    _space->setBounds(-1, 1);

    if (_space == nullptr) {
        OMPL_ERROR("Failed to construct ambient state space!");
        return false;
    }

    auto bounds = _space->getBounds();
    // print result
    OMPL_INFORM("Constructed ambient state space successfully.");
    std::stringstream ss;
    ss << std::endl;
    ss << "\tDimension: " << _space->getDimension() << std::endl;
    ss << "\tUpper bound:";
    for (auto hb : bounds.high) {
        ss << " " << hb;
    }
    ss << std::endl;
    ss << "\tLower bound:";
    for (auto lb : bounds.low) {
        ss << " " << lb;
    }
    OMPL_DEBUG(ss.str().c_str());
    return true;
}

bool SpherePlanning::setConstraint() {
    _constraint = std::make_shared<SphereConstraint>();
    _constraint->setTolerance(1e-4);
    _constraint->setMaxIterations(50);

    if (_constraint == nullptr) {
        OMPL_ERROR("Failed to construct constraint!");
        return false;
    }

    OMPL_INFORM("Constructed constraint successfully.");
    std::stringstream ss;
    ss << std::endl;
    ss << "\tTolerance: " << _constraint->getTolerance() << std::endl;
    ss << "\tMax projection iteration: " << _constraint->getMaxIterations()
       << std::endl;
    ss << "\tAmbient space dimension: " << _constraint->getAmbientDimension()
       << std::endl;
    ss << "\tManifold dimension: " << _constraint->getManifoldDimension();
    OMPL_DEBUG(ss.str().c_str());
    return true;
}

bool SpherePlanning::setConstrainedStateSpace() {
    switch (_param.space) {
        case Parameter::PROJ: {
            _constrained_space =
                std::make_shared<ompl::base::ProjectedStateSpace>(_space, _constraint);
            _constrained_space_info =
                std::make_shared<ompl::base::ConstrainedSpaceInformation>(
                    _constrained_space);
            break;
        }
        case Parameter::ATLAS: {
            _constrained_space =
                std::make_shared<ompl::base::AtlasStateSpace>(_space, _constraint);
            _constrained_space_info =
                std::make_shared<ompl::base::ConstrainedSpaceInformation>(
                    _constrained_space);
            break;
        }
        case Parameter::TB: {
            _constrained_space = std::make_shared<ompl::base::TangentBundleStateSpace>(
                _space, _constraint);
            _constrained_space_info =
                std::make_shared<ompl::base::TangentBundleSpaceInformation>(
                    _constrained_space);
            break;
        }
    }

    _constrained_space->setDelta(0.05);
    _constrained_space->setLambda(2.0);

    if (_param.space !=
        Parameter::PROJ) { // Parameters for Atlas and Tangent Bundle state space
        auto atlas_space = _constrained_space->as<ompl::base::AtlasStateSpace>();
        atlas_space->setRho(0.3);
        atlas_space->setEpsilon(0.001);
        atlas_space->setAlpha(0.3);
        atlas_space->setExploration(0.5);
        atlas_space->setMaxChartsPerExtension(50);
        atlas_space->setSeparated(true);
        auto &&atlas = atlas_space;
        atlas_space->setBiasFunction([atlas](ompl::base::AtlasChart *c) -> double {
            return (atlas->getChartCount() - c->getNeighborCount()) + 1;
        });
    }

    if (_constrained_space == nullptr || _constrained_space_info == nullptr) {
        OMPL_ERROR("Failed to construct constrained state space!");
        return false;
    }
    OMPL_INFORM("Constructed constrained state space successfully.");
    std::stringstream ss;
    ss << std::endl;
    ss << "\tSpace type: " << _constrained_space->getName() << std::endl;
    ss << "\tAmbient space dimension: "
       << _constrained_space->getAmbientDimension() << std::endl;
    ss << "\tManifold dimension: " << _constrained_space->getManifoldDimension()
       << std::endl;
    ss << "\tDelta (Step-size for discrete geodesic on manifold): "
       << _constrained_space->getDelta() << std::endl;
    ss << "\tLambda (Maximum `wandering` allowed during traversal): "
       << _constrained_space->getLambda() << std::endl;
    if (_param.space != Parameter::PROJ) {
        auto atlas_space = _constrained_space->as<ompl::base::AtlasStateSpace>();
        ss << "\tParameters of atlas state space: " << std::endl;
        ss << "\t\tExploration (tunes balance of refinement and exploration in "
              "atlas sampling): "
           << atlas_space->getExploration() << std::endl;
        ss << "\t\tRho (max radius for an atlas chart): " << atlas_space->getRho()
           << std::endl;
        ss << "\t\tEpsilon (max distance from manifold to chart): "
           << atlas_space->getEpsilon() << std::endl;
        ss << "\t\tAlpha (max angle between chart and manifold): "
           << atlas_space->getAlpha() << std::endl;
        ss << "\t\tMax chart generated during a traversal: "
           << atlas_space->getMaxChartsPerExtension();
    }
    OMPL_DEBUG(ss.str().c_str());
    return true;
}

bool SpherePlanning::setStateValidityChecker() {
    if (_param.is_brick_env) {
        _state_validity_checker = std::make_shared<BrickValidityChecker>(_constrained_space_info, _param.brick_configs);
    } else {
        _state_validity_checker = std::make_shared<SphereValidityChecker>(_constrained_space_info);
    }
    if (_state_validity_checker == nullptr) {
        OMPL_ERROR("Failed to construct state validity checker!");
        return false;
    }
    OMPL_INFORM("Constructed state validity checker successfully.");
    return true;
}

bool SpherePlanning::setMotionValidator() {
    std::shared_ptr<BrickMotionValidator> motion_validator;
    if (_param.is_brick_env) {
        motion_validator = std::make_shared<BrickMotionValidator>(_constrained_space_info, _param.brick_configs);
        _constrained_space_info->setMotionValidator(motion_validator);
    }
    return true;
}

bool SpherePlanning::setPlanner() {
    switch (_param.planner) {
        case Parameter::RRTConnect: {
            _planner = std::make_shared<ompl::geometric::RRTConnect>(_constrained_space_info);
            _planner->as<ompl::geometric::RRTConnect>()->setRange(0.05);
            break;
        }
        case Parameter::CoMPNet: {
            _planner = std::make_shared<ompl::geometric::MPNetPlanner>(
                _constrained_space_info,
                _param.pnet,
                _param.voxel);
            _planner->as<ompl::geometric::MPNetPlanner>()->setRange(0.05);
        }
        default: {
            break;
        }
    }

    if (_planner == nullptr) {
        OMPL_ERROR("Failed to construct planner!");
        return false;
    }

    OMPL_INFORM("Constructed planner successfully.");
    std::stringstream ss;
    ss << std::endl;
    ss << "\tPlanner: " << _planner->getName() << std::endl;
    OMPL_DEBUG(ss.str().c_str());
    return true;
}

bool SpherePlanning::setSimpleSetup() {
    _simple_setup =
        std::make_shared<ompl::geometric::SimpleSetup>(_constrained_space_info);
    _simple_setup->setPlanner(_planner);
    _simple_setup->setStateValidityChecker(_state_validity_checker);

    if (_simple_setup == nullptr) {
        OMPL_ERROR("Failed to construct simple setup!");
        return false;
    }

    OMPL_INFORM("Constructed simple setup successfully.");
    return true;
}

bool SpherePlanning::setStartGoal(
    const Eigen::Ref<const Eigen::VectorXd> &start,
    const Eigen::Ref<const Eigen::VectorXd> &goal) {
    ompl::base::ScopedState<> sstart(_constrained_space);
    ompl::base::ScopedState<> sgoal(_constrained_space);

    sstart->as<ompl::base::ConstrainedStateSpace::StateType>()->copy(start);
    sgoal->as<ompl::base::ConstrainedStateSpace::StateType>()->copy(goal);

    if (_param.space != Parameter::PROJ) {
        _constrained_space->as<ompl::base::AtlasStateSpace>()->newChart(
            sstart->as<ompl::base::AtlasStateSpace::StateType>());
        _constrained_space->as<ompl::base::AtlasStateSpace>()->newChart(
            sgoal->as<ompl::base::AtlasStateSpace::StateType>());
    }

    _simple_setup->setStartAndGoalStates(sstart, sgoal);
    OMPL_INFORM("Set start and goal successfully.");
    OMPL_DEBUG("\n\tStart: %f %f %f\n\tGoal: %f %f %f\n", start[0], start[1], start[2], goal[0], goal[1], goal[2]);

    return true;
}
