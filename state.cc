#include "state.h"

State::State() : board_(kBoardSize, std::vector<int64_t>(kBoardSize, kEmpty)), player_positions_(kPlayerNum) {
  for (int64_t i = 0; i < kBoardSize; i++) {
    for (int64_t j = 0; j < kBoardSize; j++) {
      if (i == 0 || i == kBoardSize - 1 || j == 0 || j == kBoardSize - 1) {
        board_[i][j] = kWall;
      }
    }
  }
}

std::ostream& operator<<(std::ostream& ost, const State& state) {
  ost << "board" << std::endl;
  for (int64_t i = 0; i < State::kBoardSize; i++) {
    for (int64_t j = 0; j < State::kBoardSize; j++) {
      ost << (state.board_[i][j] == kWall ? '#' : '.');
    }
    ost << std::endl;
  }
  return ost;
}

void State::Step(Action a) {}