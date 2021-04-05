//
// Created by jiangeng on 3/29/20.
//

#include "planner/MPNetSampler.h"
#include <ompl/base/spaces/constraint/AtlasStateSpace.h>

AtlasMPNet::MPNetSampler::MPNetSampler(const ompl::base::StateSpace *space, const torch::jit::script::Module &pnet, const torch::Tensor &voxel) : ompl::base::StateSampler(space) {
    dim_ = 3;

    pnet_ = pnet;
    pnet_.to(at::kCUDA);

    voxel_ = voxel;
}

bool AtlasMPNet::MPNetSampler::sample(const ompl::base::State *start, const ompl::base::State *goal, ompl::base::State *sample) {
    std::vector<double> start_config(dim_), goal_config(dim_);
    space_->copyToReals(start_config, start);
    space_->copyToReals(goal_config, goal);

    // sample a config with MPNet
    auto pnet_input = torch::cat({voxel_, toTensor(start_config), toTensor(goal_config)}, 1).to(at::kCUDA);
    auto pnet_output = pnet_.forward({pnet_input}).toTensor().to(at::kCPU);

    auto sample_config = toVector(pnet_output);
    double norm = 0;
    for (auto &x : sample_config) {
        norm += x * x;
    }
    norm = sqrt(norm);
    for (auto &x : sample_config) {
        x /= norm;
    }

    space_->copyFromReals(sample, sample_config);
    return true;
}

bool AtlasMPNet::MPNetSampler::sampleBatch(const std::vector<const ompl::base::State *> &starts, const std::vector<const ompl::base::State *> &goals, std::vector<ompl::base::State *> &samples) {
    std::vector<float> starts_vec, goals_vec;
    long n = starts.size();
    for (const auto &start : starts) {
        auto &start_raw = *start->as<ompl::base::ConstrainedStateSpace::StateType>();
        starts_vec.emplace_back(start_raw[0]);
        starts_vec.emplace_back(start_raw[1]);
        starts_vec.emplace_back(start_raw[2]);
    }
    for (const auto &goal : goals) {
        auto &goal_raw = *goal->as<ompl::base::ConstrainedStateSpace::StateType>();
        goals_vec.emplace_back(goal_raw[0]);
        goals_vec.emplace_back(goal_raw[1]);
        goals_vec.emplace_back(goal_raw[2]);
    }

    auto starts_tensor = torch::from_blob(starts_vec.data(), {n, dim_}).clone();
    auto goals_tensor = torch::from_blob(goals_vec.data(), {n, dim_}).clone();
    auto voxels_tensor = voxel_.repeat_interleave(n, 0);
    auto pnet_input = torch::cat({voxels_tensor, starts_tensor, goals_tensor}, 1).to(at::kCUDA);
    auto pnet_output = pnet_.forward({pnet_input}).toTensor().to(at::kCPU);
    auto samples_data = pnet_output.accessor<float, 2>();
    for (long i = 0; i < n; i++) {
        auto &sample_raw = *samples[i]->as<ompl::base::ConstrainedStateSpace::StateType>();
        sample_raw[0] = samples_data[i][0];
        sample_raw[1] = samples_data[i][1];
        sample_raw[2] = samples_data[i][2];
        auto norm = sample_raw[0] * sample_raw[0] + sample_raw[1] * sample_raw[1] + sample_raw[2] * sample_raw[2];
        sample_raw[0] /= norm;
        sample_raw[1] /= norm;
        sample_raw[2] /= norm;
    }

    return true;
}

std::vector<double> AtlasMPNet::MPNetSampler::toVector(const torch::Tensor &tensor) {
    auto data = tensor.accessor<float, 2>()[0];
    std::vector<double> dest(dim_);
    for (unsigned int i = 0; i < dim_; i++) {
        dest[i] = static_cast<float>(data[i]);
    }
    return dest;
}

torch::Tensor AtlasMPNet::MPNetSampler::toTensor(const std::vector<double> &vec) {
    std::vector<float> scaled_src(dim_);
    for (unsigned int i = 0; i < dim_; i++) {
        scaled_src[i] = vec[i];
    }
    return torch::from_blob(scaled_src.data(), {1, dim_}).clone();
}

std::vector<float> AtlasMPNet::MPNetSampler::loadData(const std::string &filename, unsigned int n) {
    std::ifstream file(filename);
    std::string line;
    std::vector<float> vec(n);
    for (unsigned int i = 0; i < n; i++) {
        if (!getline(file, line))
            break;
        vec[i] = std::stof(line);
    }
    file.close();
    return vec;
}
