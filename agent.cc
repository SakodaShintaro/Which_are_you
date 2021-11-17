#include "agent.h"

#include <random>

constexpr int64_t kInputSize = (kPlayerNum + 1) * State::kBoardSize + kMoveActionNum;
constexpr int64_t kPolicyDim = kAllActionNum - 1;

// 行動にはnull_moveも含まれているが、それを選択することはないので-1した値を方策の数とする
Agent::Agent() : lstm_(kInputSize, kPolicyDim) { lstm_.to(torch::Device(torch::kCUDA)); }

Action Agent::SelectAction(const State& state, bool first_action) {
  torch::NoGradGuard no_grad_guard;
  std::vector<float> curr_feature = state.GetFeature();
  const torch::Device& device = lstm_.parameters().front().device();
  torch::Tensor input_tensor = torch::tensor(curr_feature);
  input_tensor = input_tensor.view({1, 1, kInputSize});
  input_tensor = input_tensor.to(device);
  torch::Tensor output = lstm_.forward(input_tensor);
  output = output.flatten();

  std::mt19937_64 engine(std::random_device{}());
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  float prob = dist(engine);

  if (first_action) {
    // 初手で回答することは禁止
    output = output.index({torch::indexing::Slice(0, kMoveActionNum)});
    torch::Tensor policy = torch::softmax(output, 0);
    for (int64_t i = 0; i < kMoveActionNum; i++) {
      if ((prob -= policy[i].item<float>()) <= 0) {
        return Action(i);
      }
    }
  } else {
    torch::Tensor policy = torch::softmax(output, 0);
    for (int64_t i = 0; i < kPolicyDim; i++) {
      if ((prob -= policy[i].item<float>()) <= 0) {
        return Action(i);
      }
    }
  }

  //ここに到達することはありえないはず
  std::exit(1);
}

torch::Tensor Agent::Train(const Episode& episode) {
  // LSTMに与える入力を作る
  std::vector<float> input;
  const int64_t episode_length = episode.actions.size();
  assert(episode_length == episode.state_features.size());
  for (uint64_t i = 0; i < episode_length; i++) {
    const std::vector<float>& curr_feature = episode.state_features[i];
    input.insert(input.end(), curr_feature.begin(), curr_feature.end());
  }

  torch::Tensor input_tensor = torch::tensor(input);
  assert(input.size() / episode_length == kInputSize);
  input_tensor = input_tensor.view({episode_length, 1, kInputSize});
  torch::Tensor output = lstm_.forward(input_tensor);

  torch::Tensor loss = torch::zeros({1}).to(output.device());
  for (int64_t i = 0; i < episode_length; i++) {
    torch::Tensor curr_log_policy = torch::log_softmax(output[i], 1);
    loss -= curr_log_policy[0][episode.actions[i]] * episode.reward;
  }

  return loss;
}

std::vector<torch::Tensor> Agent::Parameters() { return lstm_.parameters(); }