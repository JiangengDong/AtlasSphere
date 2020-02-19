//
// Created by jiangeng on 2/13/20.
//

#include "SpherePlanning.h"

#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/spaces/constraint/AtlasStateSpace.h>
#include <ompl/base/ConstrainedSpaceInformation.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/base/PlannerData.h>
#include <fstream>

#include "SphereConstraint.h"
#include "SphereValidityChecker.h"

SpherePlanning::SpherePlanning() {
    ompl::msg::setLogLevel(ompl::msg::LOG_INFO);
    init = setAmbientStateSpace()
           && setConstraint()
           && setConstrainedStateSpace()
           && setStateValidityChecker()
           && setPlanner()
           && setSimpleSetup();
    if (!init) {
        OMPL_ERROR("!!! Construction failed!!!");
    } else {
        _planner_data = std::make_shared<ompl::base::PlannerData>(_constrained_space_info);
    }
}

bool SpherePlanning::planOnce(const Eigen::Ref<const Eigen::VectorXd> &start, const Eigen::Ref<const Eigen::VectorXd> &goal) {
    if (!init) {
        OMPL_ERROR("This planner is not valid!");
        return false;
    }

    setStartGoal(start, goal);
    auto status = _simple_setup->solve(10);
    if (status == ompl::base::PlannerStatus::EXACT_SOLUTION) {
        _simple_setup->getPlannerData(*_planner_data);
        return true;
    } else
        return false;
}

bool SpherePlanning::clear() {
    _constrained_space->clear();
    _planner->clear();
    _simple_setup->clear();
    return true;
}

bool SpherePlanning::printStat() const {
    std::stringstream ss;
    ss << std::endl;
    ss << "\tNum of charts in atlas: " << _constrained_space->getChartCount() << std::endl;
    ss << "\tFrontier charts percentage in atlas: " << _constrained_space->estimateFrontierPercent() << std::endl;
    ss << "\tNodes in trees: " << _planner_data->numVertices() << std::endl;
    OMPL_DEBUG(ss.str().c_str());
    return true;
}

bool SpherePlanning::exportAtlas(const std::string &filename) const {
    std::ofstream file(filename + ".ply");
    _constrained_space->printPLY(file);
    file.close();
    OMPL_INFORM("Atlas has been written to file %s. ", filename.c_str());
    return true;
}

bool SpherePlanning::exportTree(const std::string &filename, int filetype) const {
    switch (filetype) {
        case 0: {
            std::ofstream gmlFile(filename + ".graphml");
            _planner_data->printGraphML(gmlFile);
            gmlFile.close();
            break;
        }
        case 1: {
            std::ofstream plyFile(filename + ".ply");
            _planner_data->printPLY(plyFile);
            plyFile.close();
            break;
        }
        case 2: {
            std::ofstream gvzFile(filename + ".dot");
            _planner_data->printGraphviz(gvzFile);
            gvzFile.close();
            break;
        }
        default:
            break;
    }
    OMPL_INFORM("Tree has been written to file %s. ", filename.c_str());
    return true;
}

bool SpherePlanning::exportPath(const std::string &filename, std::ios_base::openmode mode) const {
    auto path = _simple_setup->getSolutionPath();
    std::ofstream file(filename, mode);
    path.printAsMatrix(file);
    file.close();
    return true;
}

Eigen::MatrixXd SpherePlanning::getPath() {
    auto path = _simple_setup->getSolutionPath();
    std::size_t n = path.getStateCount();
    Eigen::MatrixXd pathM(n, 3);
    std::vector<double> temp;
    for(std::size_t i=0; i<n;i++) {
        _constrained_space->copyToReals(temp, path.getState(i));
        pathM(i, 0) = temp[0];
        pathM(i, 1) = temp[1];
        pathM(i, 2) = temp[2];
    }
    return pathM;
}

bool SpherePlanning::isValid(const Eigen::Ref<const Eigen::VectorXd> &state) const {
    ompl::base::ScopedState<> sstate(_constrained_space);

    sstate->as<ompl::base::ConstrainedStateSpace::StateType>()->copy(state);
    return _state_validity_checker->isValid(sstate.get());
}

bool SpherePlanning::setAmbientStateSpace() {
    _space = std::make_shared<ompl::base::RealVectorStateSpace>(3);
    _space->setBounds(-2, 2);

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
    for (auto hb:bounds.high) {
        ss << " " << hb;
    }
    ss << std::endl;
    ss << "\tLower bound:";
    for (auto lb:bounds.low) {
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
    ss << "\tMax projection iteration: " << _constraint->getMaxIterations() << std::endl;
    ss << "\tAmbient space dimension: " << _constraint->getAmbientDimension() << std::endl;
    ss << "\tManifold dimension: " << _constraint->getManifoldDimension();
    OMPL_DEBUG(ss.str().c_str());
    return true;
}

bool SpherePlanning::setConstrainedStateSpace() {
    _constrained_space = std::make_shared<ompl::base::AtlasStateSpace>(_space, _constraint);
    _constrained_space_info = std::make_shared<ompl::base::ConstrainedSpaceInformation>(_constrained_space);

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

    if (_constrained_space == nullptr || _constrained_space_info == nullptr) {
        OMPL_ERROR("Failed to construct constrained state space!");
        return false;
    }

    OMPL_INFORM("Constructed constrained state space successfully.");
    std::stringstream ss;
    ss << std::endl;
    ss << "\tAmbient space dimension: " << _constrained_space->getAmbientDimension() << std::endl;
    ss << "\tManifold dimension: " << _constrained_space->getManifoldDimension() << std::endl;
    ss << "\tParameters of atlas state space: " << std::endl;
    ss << "\t\tDelta (Step-size for discrete geodesic on manifold): " << _constrained_space->getDelta() << std::endl;
    ss << "\t\tLambda (Maximum `wandering` allowed during traversal): " << _constrained_space->getLambda() << std::endl;
    ss << "\t\tExploration (tunes balance of refinement and exploration in atlas sampling): " << _constrained_space->getExploration() << std::endl;
    ss << "\t\tRho (max radius for an atlas chart): " << _constrained_space->getRho() << std::endl;
    ss << "\t\tEpsilon (max distance from manifold to chart): " << _constrained_space->getEpsilon() << std::endl;
    ss << "\t\tAlpha (max angle between chart and manifold): " << _constrained_space->getAlpha() << std::endl;
    ss << "\t\tMax chart generated during a traversal: " << _constrained_space->getMaxChartsPerExtension();
    OMPL_DEBUG(ss.str().c_str());
    return true;
}

bool SpherePlanning::setStateValidityChecker() {
    _state_validity_checker = std::make_shared<SphereValidityChecker>(_constrained_space_info);
    if (_state_validity_checker == nullptr) {
        OMPL_ERROR("Failed to construct state validity checker!");
        return false;
    }
    OMPL_INFORM("Constructed state validity checker successfully.");
    return true;
}

bool SpherePlanning::setPlanner() {
    _planner = std::make_shared<ompl::geometric::RRTConnect>(_constrained_space_info);
    _planner->setRange(0.05);

    if (_planner == nullptr) {
        OMPL_ERROR("Failed to construct planner!");
        return false;
    }

    OMPL_INFORM("Constructed planner successfully.");
    std::stringstream ss;
    ss << std::endl;
    ss << "\tPlanner: " << _planner->getName() << std::endl;
    ss << "\t\tRange: " << _planner->getRange();
    OMPL_DEBUG(ss.str().c_str());
    return true;
}

bool SpherePlanning::setSimpleSetup() {
    _simple_setup = std::make_shared<ompl::geometric::SimpleSetup>(_constrained_space_info);
    _simple_setup->setPlanner(_planner);
    _simple_setup->setStateValidityChecker(_state_validity_checker);

    if (_simple_setup == nullptr) {
        OMPL_ERROR("Failed to construct simple setup!");
        return false;
    }

    OMPL_INFORM("Constructed simple setup successfully.");
    return true;
}

bool SpherePlanning::setStartGoal(const Eigen::Ref<const Eigen::VectorXd> &start, const Eigen::Ref<const Eigen::VectorXd> &goal) {
    ompl::base::ScopedState<> sstart(_constrained_space);
    ompl::base::ScopedState<> sgoal(_constrained_space);

    sstart->as<ompl::base::ConstrainedStateSpace::StateType>()->copy(start);
    sgoal->as<ompl::base::ConstrainedStateSpace::StateType>()->copy(goal);

    _constrained_space->as<ompl::base::AtlasStateSpace>()->anchorChart(sstart.get());
    _constrained_space->as<ompl::base::AtlasStateSpace>()->anchorChart(sgoal.get());

    _simple_setup->setStartAndGoalStates(sstart, sgoal);
    return true;
}