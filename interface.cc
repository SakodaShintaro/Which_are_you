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
  while (true) {
    Action a = agent.SelectAction(state);
    auto [is_finish, reward] = state.Step(a);
    std::cout << state;
    if (is_finish) {
      std::cout << "reward = " << reward << std::endl;
      break;
    }
  }

  Episode episode = state.GetEpisode();
  PrintEpisode(episode);
}