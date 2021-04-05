//
// Created by jiangeng on 3/29/20.
//

#ifndef ATLASMPNET_MPNETSAMPLER_H
#define ATLASMPNET_MPNETSAMPLER_H

#include <ompl/base/ConstrainedSpaceInformation.h>
#include <ompl/base/StateSampler.h>
#include <ompl/base/spaces/constraint/AtlasStateSpace.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/script.h>
#include <utility>

#include "Parameter.h"

namespace AtlasMPNet {
class MPNetSampler : public ompl::base::StateSampler {
public:
    typedef std::shared_ptr<MPNetSampler> Ptr;
    MPNetSampler(const ompl::base::StateSpace *space, const torch::jit::script::Module &pnet, const torch::Tensor &voxel);

    bool sample(const ompl::base::State *start, const ompl::base::State *goal, ompl::base::State *sample);

    bool sampleBatch(const std::vector<const ompl::base::State *> &starts, const std::vector<const ompl::base::State *> &goals, std::vector<ompl::base::State *> &samples);

    void sampleUniform(ompl::base::State *state) override {
    }

    void sampleUniformNear(ompl::base::State *state, const ompl::base::State *near, double distance) override {
    }

    void sampleGaussian(ompl::base::State *state, const ompl::base::State *mean, double stdDev) override {
    }

private:
    torch::jit::script::Module pnet_;
    torch::Tensor voxel_;

    unsigned int dim_; // dimension of config

    torch::Tensor toTensor(const std::vector<double> &src);

    std::vector<double> toVector(const torch::Tensor &tensor);

    static std::vector<float> loadData(const std::string &filename, unsigned int n);
};
} // namespace AtlasMPNet

#endif //ATLASMPNET_MPNETSAMPLER_H
