#include "agent.h"

#include <random>

constexpr int64_t kInputSize = (kPlayerNum + 1) * State::kBoardSize + kAllActionNum;

// 行動にはnull_moveも含まれているが、それを選択することはないので-1した値を方策の数とする
Agent::Agent() : lstm_(kInputSize, kAllActionNum - 1) {}

Action Agent::SelectAction(const State& state) {
  std::mt19937_64 engine(std::random_device{}());
  std::uniform_int_distribution<int64_t> dist(0, kMoveActionNum + kPlayerNum - 1);
  return Action(dist(engine));
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

  std::cout << "episode_length = " << episode_length << std::endl;

  std::cout << "input.size() = " << input.size() << std::endl;
  torch::Tensor input_tensor = torch::tensor(input);
  const int64_t dim = input.size() / episode_length;
  input_tensor = input_tensor.view({episode_length, 1, dim});
  std::cout << "input_tensor.sizes() = " << input_tensor.sizes() << std::endl;
  torch::Tensor output = lstm_.forwardSequence(input_tensor);
  std::cout << "output.sizes() = " << output.sizes() << std::endl;

  torch::Tensor loss = torch::zeros({1});
  for (int64_t i = 0; i < episode_length; i++) {
    torch::Tensor curr_log_policy = torch::log_softmax(output[i], 1);
    loss += curr_log_policy[0][episode.actions[i]] * episode.reward;
  }

  return loss;
}

std::vector<torch::Tensor> Agent::Parameters() { return lstm_.Parameters(); }