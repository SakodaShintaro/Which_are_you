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
    for (int64_t k = 0; k < kMoveActionNum; k++) {
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
      case 'S':
        a = kStay;
        break;

      default:
        std::cout << "Unknown operation: " << op << std::endl;
        std::exit(1);
        break;
    }
    const bool terminal = state.Step(a);
    if (terminal) {
      PrintEpisode(state.GetEpisode());
      break;
    }
  }
}

void Visualize() {
  State state;
  Agent agent;
  std::cout << std::fixed;
  constexpr int64_t kAverageSize = 200;
  state.Init();
  agent.ResetLSTM();
  std::cout << state;
  bool first_select = true;
  while (true) {
    Action a = agent.SelectAction(state, first_select);
    std::cout << "action = " << a << std::endl;
    first_select = false;
    const bool is_finish = state.Step(a);
    std::cout << state;
    if (is_finish) {
      break;
    }
  }

  Episode episode = state.GetEpisode();
  std::cout << "reward = " << episode.reward << std::endl;
  std::cout << "-------------------------------------" << std::endl;

  for (int64_t step = 1; step <= 10000000; step++) {
    constexpr float kBaseLearnRate = 0.01f;
    torch::optim::SGDOptions sgd_option(kBaseLearnRate);
    sgd_option.momentum(0.9f);
    sgd_option.weight_decay(1e-4f);
    std::vector<torch::Tensor> parameters;
    torch::optim::SGD optimizer(agent.Parameters(), sgd_option);

    auto [policy_loss, value_loss] = agent.Train(episode);

    optimizer.zero_grad();
    policy_loss.backward();
    optimizer.step();

    // 学習後の結果を見る
    state.Init();
    agent.ResetLSTM();
    Action a = agent.SelectAction(state, true);

    int64_t wait;
    std::cin >> wait;
  }
}

void Learn(int64_t train_id) {
  State state;
  Agent agent;
  std::vector<float> reward_list;
  std::vector<float> policy_loss_list;
  std::vector<float> value_loss_list;
  std::cout << std::fixed;
  constexpr int64_t kAverageSize = 200;
  constexpr int64_t kMaxStep = 10000;
  constexpr int64_t kPeekStep = kMaxStep / 20;
  constexpr float kBaseLearnRate = 0.01f;
  torch::optim::SGDOptions sgd_option(0.0f);
  sgd_option.momentum(0.9f);
  sgd_option.weight_decay(1e-4f);
  torch::optim::SGD optimizer(agent.Parameters(), sgd_option);

  std::ofstream loss_log("loss_log_" + std::to_string(train_id) + ".txt");

  for (int64_t step = 1; step <= kMaxStep; step++) {
    state.Init();
    agent.ResetLSTM();
    bool first_select = true;
    while (true) {
      Action a = agent.SelectAction(state, first_select);
      first_select = false;
      const bool is_finish = state.Step(a);
      if (is_finish) {
        break;
      }
    }

    Episode episode = state.GetEpisode();
    auto [policy_loss, value_loss] = agent.Train(episode);
    reward_list.push_back(episode.reward);
    policy_loss_list.push_back(policy_loss.item<float>());
    value_loss_list.push_back(value_loss.item<float>());

    if (reward_list.size() == kAverageSize) {
      float reward_average = std::accumulate(reward_list.begin(), reward_list.end(), 0.0f) / kAverageSize;
      float policy_loss_average =
          std::accumulate(policy_loss_list.begin(), policy_loss_list.end(), 0.0f) / kAverageSize;
      float value_loss_average = std::accumulate(value_loss_list.begin(), value_loss_list.end(), 0.0f) / kAverageSize;
      std::cout << step << "\t" << reward_average << "\t" << policy_loss_average << "\t" << value_loss_average
                << std::endl;
      loss_log << step << "\t" << reward_average << "\t" << policy_loss_average << "\t" << value_loss_average
               << std::endl;
      reward_list.clear();
      policy_loss_list.clear();
      value_loss_list.clear();
    }

    if (step <= kPeekStep) {
      (dynamic_cast<torch::optim::SGDOptions&>(optimizer.param_groups().front().options())).lr() =
          kBaseLearnRate * step / kPeekStep;
    } else {
      (dynamic_cast<torch::optim::SGDOptions&>(optimizer.param_groups().front().options())).lr() =
          kBaseLearnRate * (kMaxStep - step) / (kMaxStep - kPeekStep);
    }

    optimizer.zero_grad();
    (policy_loss).backward();
    torch::nn::utils::clip_grad_norm_(agent.Parameters(), 0.1);
    optimizer.step();
  }
}