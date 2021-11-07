#include "interface.h"

#include "agent.h"
#include "state.h"

void PrintEpisode(const Episode& episode) {
  for (uint64_t i = 0; i < episode.actions.size(); i++) {
    for (int64_t ch = 0; ch <= kPlayerNum; ch++) {
      std::cout << "ch = " << ch << std::endl;
      for (int64_t j = 0; j < State::kBoardWidth; j++) {
        for (int64_t k = 0; k < State::kBoardWidth; k++) {
          const int64_t index = ch * State::kBoardSize + j * State::kBoardWidth + k;
          std::cout << episode.state_features[i][index] << " ";
        }
        std::cout << std::endl;
      }
    }
    std::cout << "pre_action = ";
    for (int64_t k = 0; k < kAllActionNum; k++) {
      std::cout << episode.state_features[i][(kPlayerNum + 1) * State::kBoardSize + k] << " ";
    }
    std::cout << std::endl;
    std::cout << "action = " << episode.actions[i] << std::endl;
  }
  std::cout << "reward = " << episode.reward << std::endl;
}

void Manual() {
  State state;

  while (true) {
    // 表示
    std::cout << state;

    char op;
    std::cout << "操作: ";
    std::cin >> op;
    if (op == 'q' || op == 'Q') {
      break;
    }

    Action a = kMoveActionNum;
    switch (op) {
      case 'U':
        a = kUp;
        break;
      case 'R':
        a = kRight;
        break;
      case 'D':
        a = kDown;
        break;
      case 'L':
        a = kLeft;
        break;

      default:
        std::exit(1);
        break;
    }
    state.Step(a);
  }
}

void Learn() {
  State state;
  Agent agent;
  std::vector<float> reward_list;
  std::vector<float> loss_list;
  constexpr int64_t kAverageSize = 100;
  for (int64_t i = 0; i < 100000; i++) {
    state.Init();
    agent.ResetLSTM();
    while (true) {
      Action a = agent.SelectAction(state);
      auto [is_finish, reward] = state.Step(a);
      if (is_finish) {
        break;
      }
    }

    constexpr float learn_rate_ = 0.001f;
    torch::optim::SGDOptions sgd_option(learn_rate_);
    sgd_option.momentum(0.9f);
    sgd_option.weight_decay(1e-4f);
    std::vector<torch::Tensor> parameters;
    torch::optim::SGD optimizer_(agent.Parameters(), sgd_option);

    Episode episode = state.GetEpisode();
    torch::Tensor loss = agent.Train(episode);
    reward_list.push_back(episode.reward);
    loss_list.push_back(loss.item<float>());

    if (reward_list.size() == kAverageSize) {
      float reward_average = std::accumulate(reward_list.begin(), reward_list.end(), 0.0f) / kAverageSize;
      float loss_average = std::accumulate(loss_list.begin(), loss_list.end(), 0.0f) / kAverageSize;
      std::cout << reward_average << "\t" << loss_average << std::endl;
      reward_list.clear();
      loss_list.clear();
    }

    optimizer_.zero_grad();
    loss.backward();
    optimizer_.step();
  }
}