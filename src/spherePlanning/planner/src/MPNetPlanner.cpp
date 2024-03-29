/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2008, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

/* Author: Ioan Sucan */

#include "planner/MPNetPlanner.h"

#include <ompl/base/goals/GoalSampleableRegion.h>
#include <ompl/base/spaces/constraint/AtlasStateSpace.h>
#include <ompl/base/spaces/constraint/ProjectedStateSpace.h>
#include <ompl/tools/config/SelfConfig.h>
#include <ompl/util/String.h>

ompl::geometric::MPNetPlanner::MPNetPlanner(const base::SpaceInformationPtr &si, const torch::jit::script::Module &pnet, const torch::Tensor &voxel)
    : base::Planner(si, "MPNetPlanner") {
    specs_.recognizedGoal = base::GOAL_SAMPLEABLE_REGION;
    specs_.directed = true;

    Planner::declareParam<double>("range", this, &MPNetPlanner::setRange, &MPNetPlanner::getRange, "0.:1.:10000.");

    connectionPoint_ = std::make_pair<base::State *, base::State *>(nullptr, nullptr);
    distanceBetweenTrees_ = std::numeric_limits<double>::infinity();
    mpnet_sampler_ = std::make_shared<AtlasMPNet::MPNetSampler>(si_->getStateSpace().get(), pnet, voxel);
    simple_sampler_ = si_->allocStateSampler();
}

ompl::geometric::MPNetPlanner::~MPNetPlanner() {
    freeMemory();
}

void ompl::geometric::MPNetPlanner::setup() {
    Planner::setup();
    tools::SelfConfig sc(si_, getName());
    sc.configurePlannerRange(maxDistance_);

    if (!tStart_)
        tStart_.reset(tools::SelfConfig::getDefaultNearestNeighbors<Motion *>(this));
    if (!tGoal_)
        tGoal_.reset(tools::SelfConfig::getDefaultNearestNeighbors<Motion *>(this));
    tStart_->setDistanceFunction([this](const Motion *a, const Motion *b) { return distanceFunction(a, b); });
    tGoal_->setDistanceFunction([this](const Motion *a, const Motion *b) { return distanceFunction(a, b); });
}

void ompl::geometric::MPNetPlanner::freeMemory() {
    std::vector<Motion *> motions;

    if (tStart_) {
        tStart_->list(motions);
        for (auto &motion : motions) {
            if (motion->state != nullptr)
                si_->freeState(motion->state);
            delete motion;
        }
    }

    if (tGoal_) {
        tGoal_->list(motions);
        for (auto &motion : motions) {
            if (motion->state != nullptr)
                si_->freeState(motion->state);
            delete motion;
        }
    }
}

void ompl::geometric::MPNetPlanner::clear() {
    Planner::clear();
    freeMemory();
    if (tStart_)
        tStart_->clear();
    if (tGoal_)
        tGoal_->clear();
    connectionPoint_ = std::make_pair<base::State *, base::State *>(nullptr, nullptr);
    distanceBetweenTrees_ = std::numeric_limits<double>::infinity();
}

ompl::geometric::MPNetPlanner::GrowState ompl::geometric::MPNetPlanner::growTree(TreeData &tree, TreeGrowingInfo &tgi, Motion *rmotion) {
    /* find closest state in the tree */
    Motion *nmotion = tree->nearest(rmotion);

    // this is designed for AtlasStateSpace only.
    std::vector<ompl::base::State *> stateList;
    bool reach = si_->getStateSpace()->as<ompl::base::ConstrainedStateSpace>()->discreteGeodesic(nmotion->state, rmotion->state, false, &stateList);

    if (stateList.empty() ||                                  // did not traverse at all
        si_->equalStates(nmotion->state, stateList.back()) || // did not make a progress
        !si_->checkMotion(nmotion->state, stateList.front())  // collide at the first step
    ) {
        si_->freeStates(stateList);
        return TRAPPED;
    }

    Motion *motion = nullptr;
    for (auto dstate : stateList) {
        if (!si_->satisfiesBounds(dstate) ||
            !si_->checkMotion(nmotion->state, dstate))
            break;
        motion = new Motion(si_);
        si_->copyState(motion->state, dstate);
        motion->parent = nmotion;
        motion->root = nmotion->root;
        tree->add(motion);
        nmotion = motion;
    }
    tgi.xmotion = motion;

    si_->freeStates(stateList);

    return reach ? REACHED : ADVANCED;
}

ompl::geometric::MPNetPlanner::Motion *ompl::geometric::MPNetPlanner::nearest(const TreeData &tree, const Motion *center) {
    std::vector<Motion *> motions;
    tree->list(motions);

    double min_dist = INFINITY;
    Motion *best;
    double dist = 0.0;

    for (const auto &motion : motions) {
        dist = distanceFunction(motion, center);
        if (dist < min_dist) {
            best = motion;
            dist = min_dist;
        }
    }
    return best;
}

ompl::base::PlannerStatus ompl::geometric::MPNetPlanner::solve(const base::PlannerTerminationCondition &ptc) {
    checkValidity();

    Motion *start_motion, *goal_motion;
    auto *goal = dynamic_cast<base::GoalSampleableRegion *>(pdef_->getGoal().get());

    if (goal == nullptr) {
        OMPL_ERROR("%s: Unknown type of goal", getName().c_str());
        return base::PlannerStatus::UNRECOGNIZED_GOAL_TYPE;
    }

    while (const base::State *st = pis_.nextStart()) {
        start_motion = new Motion(si_);
        si_->copyState(start_motion->state, st);
        start_motion->root = start_motion->state;
        tStart_->add(start_motion);
    }

    if (tStart_->size() == 0) {
        OMPL_ERROR("%s: Motion planning start tree could not be initialized!", getName().c_str());
        return base::PlannerStatus::INVALID_START;
    }

    if (!goal->couldSample()) {
        OMPL_ERROR("%s: Insufficient states in sampleable goal region", getName().c_str());
        return base::PlannerStatus::INVALID_GOAL;
    }

    OMPL_INFORM("%s: Starting planning with %d states already in datastructure", getName().c_str(), (int)(tStart_->size() + tGoal_->size()));

    TreeGrowingInfo tgi{};

    Motion *approxsol = nullptr;
    double approxdif = std::numeric_limits<double>::infinity();
    auto *rmotion = new Motion(si_);
    base::State *rstate = rmotion->state;
    bool startTree = true;
    bool solved = false;
    srand(0);

    while (!ptc) {
        TreeData &tree = startTree ? tStart_ : tGoal_;
        TreeData &otherTree = startTree ? tGoal_ : tStart_;
        Motion *&motionA = startTree ? start_motion : goal_motion;
        Motion *&motionB = startTree ? goal_motion : start_motion;
        tgi.start = startTree;
        startTree = !startTree;

        if (tGoal_->size() == 0 || pis_.getSampledGoalsCount() < tGoal_->size() / 2) {
            const base::State *st = tGoal_->size() == 0 ? pis_.nextGoal(ptc) : pis_.nextGoal();
            if (st != nullptr) {
                goal_motion = new Motion(si_);
                si_->copyState(goal_motion->state, st);
                goal_motion->root = goal_motion->state;
                tGoal_->add(goal_motion);
            }

            if (tGoal_->size() == 0) {
                OMPL_ERROR("%s: Unable to sample any valid states for goal tree", getName().c_str());
                break;
            }
        }

        /* sample random state */
        if (use_mpnet)
            mpnet_sampler_->sample(motionA->state, motionB->state, rstate);
        else
            simple_sampler_->sampleUniform(rstate);
        std::vector<double> temp_vec;
        si_->getStateSpace()->copyToReals(temp_vec, rstate);
        samples_memory_.emplace_back(temp_vec);
        GrowState gs = growTree(tree, tgi, rmotion);

        if (gs != TRAPPED) {
            motionA = tgi.xmotion;
            /* attempt to connect trees */
            tgi.start = startTree;
            GrowState gsc = growTree(otherTree, tgi, motionA);
            if (gsc != TRAPPED) {
                motionB = tgi.xmotion;
            }

            /* update distance between trees */
            const double newDist = tree->getDistanceFunction()(start_motion, goal_motion);
            if (newDist < distanceBetweenTrees_) {
                distanceBetweenTrees_ = newDist;
                // OMPL_INFORM("Estimated distance to go: %f", distanceBetweenTrees_);
            }
            /* if we connected the trees in a valid way (start and goal pair is valid)*/
            if (gsc == REACHED && goal->isStartGoalPairValid(start_motion->root, goal_motion->root)) {
                // it must be the case that either the start tree or the goal tree has made some progress
                // so one of the parents is not nullptr. We go one step 'back' to avoid having a duplicate state
                // on the solution path
                if (start_motion->parent != nullptr)
                    start_motion = start_motion->parent;
                else
                    goal_motion = goal_motion->parent;

                connectionPoint_ = std::make_pair(start_motion->state, goal_motion->state);

                /* construct the solution path */
                Motion *solution = start_motion;
                std::vector<Motion *> mpath1;
                while (solution != nullptr) {
                    mpath1.push_back(solution);
                    solution = solution->parent;
                }

                solution = goal_motion;
                std::vector<Motion *> mpath2;
                while (solution != nullptr) {
                    mpath2.push_back(solution);
                    solution = solution->parent;
                }

                auto path(std::make_shared<PathGeometric>(si_));
                path->getStates().reserve(mpath1.size() + mpath2.size());
                for (int i = mpath1.size() - 1; i >= 0; --i)
                    path->append(mpath1[i]->state);
                for (auto &i : mpath2)
                    path->append(i->state);

                pdef_->addSolutionPath(path, false, 0.0, getName());
                solved = true;
                break;
            } else {
                // We didn't reach the goal, but if we were extending the start
                // tree, then we can mark/improve the approximate path so far.
                if (!startTree) {
                    // We were working from the startTree.
                    double dist = 0.0;
                    goal->isSatisfied(tgi.xmotion->state, &dist);
                    if (dist < approxdif) {
                        approxdif = dist;
                        approxsol = tgi.xmotion;
                    }
                }
            }
        }
    }

    si_->freeState(rstate);
    delete rmotion;

    OMPL_INFORM("%s: Created %u states (%u start + %u goal)", getName().c_str(), tStart_->size() + tGoal_->size(),
                tStart_->size(), tGoal_->size());

    if (approxsol && !solved) {
        /* construct the solution path */
        std::vector<Motion *> mpath;
        while (approxsol != nullptr) {
            mpath.push_back(approxsol);
            approxsol = approxsol->parent;
        }

        auto path(std::make_shared<PathGeometric>(si_));
        for (int i = mpath.size() - 1; i >= 0; --i)
            path->append(mpath[i]->state);
        pdef_->addSolutionPath(path, true, approxdif, getName());
        return base::PlannerStatus::APPROXIMATE_SOLUTION;
    }

    return solved ? base::PlannerStatus::EXACT_SOLUTION : base::PlannerStatus::TIMEOUT;
}

void ompl::geometric::MPNetPlanner::getPlannerData(base::PlannerData &data) const {
    Planner::getPlannerData(data);

    std::vector<Motion *> motions;
    if (tStart_)
        tStart_->list(motions);

    for (auto &motion : motions) {
        if (motion->parent == nullptr)
            data.addStartVertex(base::PlannerDataVertex(motion->state, 1));
        else {
            data.addEdge(base::PlannerDataVertex(motion->parent->state, 1), base::PlannerDataVertex(motion->state, 1));
        }
    }

    motions.clear();
    if (tGoal_)
        tGoal_->list(motions);

    for (auto &motion : motions) {
        if (motion->parent == nullptr)
            data.addGoalVertex(base::PlannerDataVertex(motion->state, 2));
        else {
            // The edges in the goal tree are reversed to be consistent with start tree
            data.addEdge(base::PlannerDataVertex(motion->state, 2), base::PlannerDataVertex(motion->parent->state, 2));
        }
    }

    // Add the edge connecting the two trees
    data.addEdge(data.vertexIndex(connectionPoint_.first), data.vertexIndex(connectionPoint_.second));

    // Add some info.
    data.properties["approx goal distance REAL"] = ompl::toString(distanceBetweenTrees_);
}

void ompl::geometric::MPNetPlanner::exportSamples(std::string filename) const {
    std::ofstream out(filename);
    std::stringstream v;
    std::size_t vcount = 0;
    for (const auto &sample : samples_memory_) {
        for (const auto &val : sample) {
            v << val << " ";
        }
        v << "\n";
        vcount++;
    }

    out << "ply\n";
    out << "format ascii 1.0\n";
    out << "element vertex " << vcount << "\n";
    out << "property float x\n";
    out << "property float y\n";
    out << "property float z\n";
    out << "end_header\n";
    out << v.str();

    out.close();

    OMPL_INFORM("%d samples have been written to file %s. ", vcount, filename.c_str());
}
