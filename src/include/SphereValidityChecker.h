//
// Created by jiangeng on 2/13/20.
//

#ifndef ATLASSPHERE_SPHEREVALIDITYCHECKER_H
#define ATLASSPHERE_SPHEREVALIDITYCHECKER_H

#include <ompl/base/StateValidityChecker.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/constraint/ConstrainedStateSpace.h>

class SphereValidityChecker : public ompl::base::StateValidityChecker {
public:
    explicit SphereValidityChecker(const ompl::base::SpaceInformationPtr &si) : StateValidityChecker(si) {
    }

    bool isValid(const ompl::base::State *state) const override {
        auto &&x = *state->as<ompl::base::ConstrainedStateSpace::StateType>();

        if (-0.80 < x[2] && x[2] < -0.6)
        {
            if (-0.05 < x[1] && x[1] < 0.05)
                return x[0] > 0;
            return false;
        }
        else if (-0.1 < x[2] && x[2] < 0.1)
        {
            if (-0.05 < x[0] && x[0] < 0.05)
                return x[1] < 0;
            return false;
        }
        else if (0.6 < x[2] && x[2] < 0.80)
        {
            if (-0.05 < x[1] && x[1] < 0.05)
                return x[0] < 0;
            return false;
        }

        return true;
    }
};


#endif //ATLASSPHERE_SPHEREVALIDITYCHECKER_H
