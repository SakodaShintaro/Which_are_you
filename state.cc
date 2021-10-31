#include "state.h"
#include <random>

State::State() : board_(kBoardSize, std::vector<char>(kBoardSize, '.')), player_positions_(kPlayerNum) {
  for (int64_t i = 0; i < kBoardSize; i++) {
    for (int64_t j = 0; j < kBoardSize; j++) {
      if (i == 0 || i == kBoardSize - 1 || j == 0 || j == kBoardSize - 1) {
        board_[i][j] = '#';
      }
    }
  }
  Init();
}

std::ostream& operator<<(std::ostream& ost, const State& state) {
  ost << "board" << std::endl;
  for (int64_t i = 0; i < State::kBoardSize; i++) {
    for (int64_t j = 0; j < State::kBoardSize; j++) {
      ost << state.board_[i][j];
    }
    ost << std::endl;
  }
  return ost;
}

void State::Init() {
    // 各プレイヤーの位置をランダムに決定する
    std::mt19937_64 engine(std::random_device{}());
    std::uniform_int_distribution<int64_t> dist(1, kBoardSize - 2);
    player_positions_.resize(kPlayerNum);
    for (int64_t i = 0; i < kPlayerNum; i++) {
        while (true) {
            int64_t x = dist(engine);
            int64_t y = dist(engine);
            if (board_[y][x] == '.') {
              player_positions_[i].x = x;
              player_positions_[i].y = y;
              board_[y][x] = 'A' + i;
              break;
            }
        }
    }
}

void State::Step(Action a) {}