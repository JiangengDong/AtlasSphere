//
// Created by jiangeng on 2/13/20.
//

#include "SpherePlanning.h"

#include <algorithm>
#include <fstream>
#include <ompl/base/ConstrainedSpaceInformation.h>
#include <ompl/base/PlannerData.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/spaces/constraint/AtlasStateSpace.h>
#include <ompl/base/spaces/constraint/ProjectedStateSpace.h>
#include <ompl/base/spaces/constraint/TangentBundleStateSpace.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/prm/PRMstar.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/geometric/planners/bitstar/BITstar.h>
#include "FMT.h"

#include "MPNetPlanner.h"
#include "Parameter.h"
#include "SphereConstraint.h"
#include "SphereValidityChecker.h"
#include "RRTstar.h"

SpherePlanning::SpherePlanning(Parameter param) : _param(param) {
    init = setAmbientStateSpace() && setConstraint() && setConstrainedStateSpace() && setStateValidityChecker() && setPlanner() && setSimpleSetup();
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
    ompl::base::PlannerStatus status = ompl::base::PlannerStatus::TIMEOUT;
    status = _simple_setup->solve(5);
    _simple_setup->getPlannerData(*_planner_data);
    return true;
    // TODO: different logic for RRT*
}

bool SpherePlanning::clear() {
    _constrained_space->clear();
    _planner->clear();
    _simple_setup->clear();
    return true;
}

bool SpherePlanning::exportAtlas(const std::string &filename) const {
    if (_param.space != Parameter::PROJ) {
        std::ofstream file(filename);
        _constrained_space->as<ompl::base::AtlasStateSpace>()->printPLY(file);
        file.close();
        OMPL_INFORM("Atlas has been written to file %s. ", filename.c_str());
        return true;
    } else {
        OMPL_WARN("This is a projection state space which does not has a atlas.");
    }
}

bool SpherePlanning::exportTree(const std::string &filename) const {
    std::ofstream file(filename);
    _planner_data->printGraphML(file);
    file.close();
    OMPL_INFORM("Tree has been written to file %s. ", filename.c_str());
    return true;
}

bool SpherePlanning::exportPath(const std::string &filename, std::ios_base::openmode mode) const {
    auto path = _simple_setup->getSolutionPath();
    std::ofstream file(filename, mode);
    path.printAsMatrix(file);
    file.close();
    OMPL_INFORM("Path has been written to file %s. ", filename.c_str());
    return true;
    // TODO: add hdf5 support
}

Eigen::MatrixXd SpherePlanning::getPath() {
    auto &path = _simple_setup->getSolutionPath();
    std::size_t n = path.getStateCount();
    Eigen::MatrixXd pathM(n, 3);
    std::vector<double> temp;
    for (std::size_t i = 0; i < n; i++) {
        _constrained_space->copyToReals(temp, path.getState(i));
        pathM(i, 0) = temp[0];
        pathM(i, 1) = temp[1];
        pathM(i, 2) = temp[2];
    }
    return pathM;
}

double SpherePlanning::getPathLength() {
    return _simple_setup->getSolutionPath().length();
}

double SpherePlanning::getTime() {
    return _simple_setup->getLastPlanComputationTime();
}

bool SpherePlanning::exportSmoothPath(std::string filename) {
    auto constrained_space = _constrained_space->as<ompl::base::ConstrainedStateSpace>();
    std::vector<ompl::base::State *> path = _simple_setup->getSolutionPath().getStates();
    std::vector<ompl::base::State *> new_path;
    std::vector<ompl::base::State *> geodesic;
    for (unsigned int smooth_iter = 0; smooth_iter < 100; smooth_iter++) {
        unsigned int n = path.size();
        unsigned int iStart = 0;
        unsigned int iGoal;
        while (iStart < n - 1) {
            for (iGoal = path.size() - 1; iGoal > iStart; iGoal--) {
                bool connected = constrained_space->discreteGeodesic(path[iStart], path[iGoal], false, &geodesic);
                if (connected) {
                    double distance = 0;
                    double new_distance = 0;
                    for (unsigned int iTemp = iStart; iTemp < iGoal; iTemp++) {
                        distance += constrained_space->distance(path[iTemp], path[iTemp + 1]);
                    }
                    for (unsigned int iTemp = 0; iTemp < geodesic.size() - 1; iTemp++) {
                        new_distance += constrained_space->distance(geodesic[iTemp], geodesic[iTemp + 1]);
                    }
                    if (new_distance < distance - 1e-5) {
                        for (auto pState : geodesic) {
                            new_path.emplace_back(pState);
                        }
                        break;
                    }
                }
            }
            if (iGoal == iStart) {
                new_path.emplace_back(path[iStart]);
                iStart++;
            } else {
                iStart = iGoal;
            }
        }
        new_path.emplace_back(path.back());
        path = new_path;
        new_path.clear();
    }
    std::vector<double> real_state;
    std::ofstream file(filename);
    for (auto pState : path) {
        constrained_space->copyToReals(real_state, pState);
        file << real_state[0] << " " << real_state[1] << " " << real_state[2] << std::endl;
    }
    file.close();
    OMPL_INFORM("Path has been written to file %s. ", filename.c_str());
    return true;
}

bool SpherePlanning::isValid(const Eigen::Ref<const Eigen::VectorXd> &state) const {
    ompl::base::ScopedState<> sstate(_constrained_space);

    sstate->as<ompl::base::ConstrainedStateSpace::StateType>()->copy(state);
    return _state_validity_checker->isValid(sstate.get());
}

/* private method below */

bool SpherePlanning::setAmbientStateSpace() {
    _space = std::make_shared<ompl::base::RealVectorStateSpace>(3);
    _space->setBounds(-100, 100);

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
    ss << "\tMax projection iteration: " << _constraint->getMaxIterations() << std::endl;
    ss << "\tAmbient space dimension: " << _constraint->getAmbientDimension() << std::endl;
    ss << "\tManifold dimension: " << _constraint->getManifoldDimension();
    OMPL_DEBUG(ss.str().c_str());
    return true;
}

bool SpherePlanning::setConstrainedStateSpace() {
    switch (_param.space) {
    case Parameter::PROJ: {
        _constrained_space = std::make_shared<ompl::base::ProjectedStateSpace>(_space, _constraint);
        _constrained_space_info = std::make_shared<ompl::base::ConstrainedSpaceInformation>(_constrained_space);
        break;
    }
    case Parameter::ATLAS: {
        _constrained_space = std::make_shared<ompl::base::AtlasStateSpace>(_space, _constraint);
        _constrained_space_info = std::make_shared<ompl::base::ConstrainedSpaceInformation>(_constrained_space);
        break;
    }
    case Parameter::TB: {
        _constrained_space = std::make_shared<ompl::base::TangentBundleStateSpace>(_space, _constraint);
        _constrained_space_info = std::make_shared<ompl::base::TangentBundleSpaceInformation>(_constrained_space);
        break;
    }
    }

    _constrained_space->setDelta(0.05);
    _constrained_space->setLambda(2.0);

    if (_param.space != Parameter::PROJ) { // Parameters for Atlas and Tangent Bundle state space
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
    ss << "\tAmbient space dimension: " << _constrained_space->getAmbientDimension() << std::endl;
    ss << "\tManifold dimension: " << _constrained_space->getManifoldDimension() << std::endl;
    ss << "\tDelta (Step-size for discrete geodesic on manifold): " << _constrained_space->getDelta() << std::endl;
    ss << "\tLambda (Maximum `wandering` allowed during traversal): " << _constrained_space->getLambda() << std::endl;
    if (_param.space != Parameter::PROJ) {
        auto atlas_space = _constrained_space->as<ompl::base::AtlasStateSpace>();
        ss << "\tParameters of atlas state space: " << std::endl;
        ss << "\t\tExploration (tunes balance of refinement and exploration in atlas sampling): " << atlas_space->getExploration() << std::endl;
        ss << "\t\tRho (max radius for an atlas chart): " << atlas_space->getRho() << std::endl;
        ss << "\t\tEpsilon (max distance from manifold to chart): " << atlas_space->getEpsilon() << std::endl;
        ss << "\t\tAlpha (max angle between chart and manifold): " << atlas_space->getAlpha() << std::endl;
        ss << "\t\tMax chart generated during a traversal: " << atlas_space->getMaxChartsPerExtension();
    }
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
    switch (_param.planner) {
    case Parameter::RRTConnect: {
        _planner = std::make_shared<ompl::geometric::RRTConnect>(_constrained_space_info);
        _planner->as<ompl::geometric::RRTConnect>()->setRange(0.05);
        break;
    }
    case Parameter::CoMPNet: {
        _planner = std::make_shared<ompl::geometric::MPNetPlanner>(_constrained_space_info, 
            "/workspaces/AtlasSphere/data/pytorch_models/pnet.pt", 
            "/workspaces/AtlasSphere/data/pytorch_models/voxel.csv");
        break;
    }
    case Parameter::RRTstar: {
        _planner = std::make_shared<ompl::geometric::RRTstar>(_constrained_space_info, 
            "/workspaces/AtlasSphere/data/pytorch_models/pnet.pt", 
            "/workspaces/AtlasSphere/data/pytorch_models/voxel.csv");
        _planner->as<ompl::geometric::RRTstar>()->setRange(0.05);
        break;
    }
    case Parameter::FMT: {
        _planner = std::make_shared<ompl::geometric::FMT>(_constrained_space_info, 
            "/workspaces/AtlasSphere/data/pytorch_models/pnet.pt", 
            "/workspaces/AtlasSphere/data/pytorch_models/voxel.csv");
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

    if (_param.space != Parameter::PROJ) {
        _constrained_space->as<ompl::base::AtlasStateSpace>()->newChart(sstart->as<ompl::base::AtlasStateSpace::StateType>());
        _constrained_space->as<ompl::base::AtlasStateSpace>()->newChart(sgoal->as<ompl::base::AtlasStateSpace::StateType>());
    }

    _simple_setup->setStartAndGoalStates(sstart, sgoal);
    return true;
}
