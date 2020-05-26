//
// Created by jiangeng on 3/29/20.
//

#include "MPNetSampler.h"
#include <ompl/base/spaces/constraint/AtlasStateSpace.h>


AtlasMPNet::MPNetSampler::MPNetSampler(const ompl::base::StateSpace *space, std::string pnet_path, std::string voxel_path) : ompl::base::StateSampler(space){
    dim_ = 3;

    std::string pnet_filename="./models/pnet_script.pt";    // TODO: change this to a parameter
    pnet_ = torch::jit::load(pnet_path);
    pnet_.to(at::kCUDA);
    OMPL_DEBUG("Load %s successfully.", pnet_path.c_str());

    std::string voxel_filename="./data/encoded_voxels.csv";     // TODO: change this to a parameter
    std::vector<float> voxel_vec = loadData(voxel_path, 128);
    voxel_ = torch::from_blob(voxel_vec.data(), {1, 128}).clone();
    OMPL_DEBUG("Load %s successfully.", voxel_path.c_str());
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
    for (auto &x: sample_config) {
        norm += x*x;
    }
    norm = sqrt(norm);
    for (auto &x: sample_config) {
        x /= norm;
    }

    space_->copyFromReals(sample, sample_config);
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