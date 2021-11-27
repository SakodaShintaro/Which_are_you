#include "agent.h"

#include <random>

constexpr int64_t kPolicyDim = kAllActionNum - 1;

// 行動にはnull_moveも含まれているが、それを選択することはないので-1した値を方策の数とする
Agent::Agent() : network_(kInputDim, kPolicyDim) { network_.to(torch::Device(torch::kCUDA)); }

Action Agent::SelectAction(const State& state, bool first_action) {
  torch::NoGradGuard no_grad_guard;
  std::vector<float> curr_feature = state.GetFeature();
  const torch::Device& device = network_.parameters().front().device();
  torch::Tensor input_tensor = torch::tensor(curr_feature);
  input_tensor = input_tensor.view({1, 1, kInputDim});
  input_tensor = input_tensor.to(device);
  auto [policy, value] = network_.forward(input_tensor);
  policy = policy.flatten();

  std::mt19937_64 engine(std::random_device{}());
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  float prob = dist(engine);

  if (first_action) {
    // 初手で回答することは禁止
    policy = policy.index({torch::indexing::Slice(0, kMoveActionNum)});
    policy = torch::softmax(policy, 0);
    for (int64_t i = 0; i < kMoveActionNum; i++) {
      if ((prob -= policy[i].item<float>()) <= 0) {
        return Action(i);
      }
    }
  } else {
    policy = torch::softmax(policy, 0);
    for (int64_t i = 0; i < kPolicyDim; i++) {
      if ((prob -= policy[i].item<float>()) <= 0) {
        return Action(i);
      }
    }
  }

  //ここに到達することはありえないはず
  std::exit(1);
}

std::tuple<torch::Tensor, torch::Tensor> Agent::Train(const Episode& episode) {
  // LSTMに与える入力を作る
  std::vector<float> input;
  const int64_t episode_length = episode.actions.size();
  assert(episode_length == episode.state_features.size());
  for (uint64_t i = 0; i < episode_length; i++) {
    const std::vector<float>& curr_feature = episode.state_features[i];
    input.insert(input.end(), curr_feature.begin(), curr_feature.end());
  }

  torch::Tensor input_tensor = torch::tensor(input);
  assert(input.size() / episode_length == kInputDim);
  input_tensor = input_tensor.view({episode_length, 1, kInputDim});
  network_.resetState();
  auto [policy, value] = network_.forward(input_tensor);

  torch::Tensor policy_loss = torch::zeros({1}).to(policy.device());
  torch::Tensor value_loss = torch::zeros({1}).to(policy.device());
  for (int64_t i = 0; i < episode_length; i++) {
    // policy lossを計算
    torch::Tensor curr_log_policy = torch::log_softmax(policy[i], 1);
    policy_loss -= curr_log_policy[0][episode.actions[i]] * episode.reward;

    // value lossを計算
    value_loss = torch::pow(value[i] - episode.reward, 2);
  }

  return std::make_tuple(policy_loss, value_loss);
}

std::vector<torch::Tensor> Agent::Parameters() { return network_.parameters(); }