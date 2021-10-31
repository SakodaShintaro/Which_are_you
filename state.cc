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
  ost << "真のプレイヤー: " << char('A' + state.true_player_) << std::endl;
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
  std::uniform_int_distribution<int64_t> dist_pos(1, kBoardSize - 2);
  player_positions_.resize(kPlayerNum);
  for (int64_t i = 0; i < kPlayerNum; i++) {
    while (true) {
      int64_t x = dist_pos(engine);
      int64_t y = dist_pos(engine);
      if (board_[y][x] == '.') {
        player_positions_[i].x = x;
        player_positions_[i].y = y;
        board_[y][x] = 'A' + i;
        break;
      }
    }
  }

  // どのプレイヤーが真のプレイヤーか決定
  std::uniform_int_distribution<int64_t> dist_player(0, kPlayerNum - 1);
  true_player_ = dist_player(engine);
}

void State::Step(Action a) {
  std::mt19937_64 engine(std::random_device{}());
  std::uniform_int_distribution<int64_t> dist_pos(0, kActionNum - 1);
  for (int64_t i = 0; i < kPlayerNum; i++) {
    const Action action = (i == true_player_ ? a : Action(dist_pos(engine)));
    const int64_t ni = player_positions_[i].y + kDi[action];
    const int64_t nj = player_positions_[i].x + kDj[action];

    // 移動先が空のときだけ更新
    // プレイヤーidが速い方から更新を行うので、同じ空きマスへの移動が同ターンで発生した場合
    // プレイヤーidの速い方のみが移動でき、遅い方はその場に留まる
    if (board_[ni][nj] == '.') {
      board_[player_positions_[i].y][player_positions_[i].x] = '.';
      board_[ni][nj] = 'A' + i;
      player_positions_[i].x = nj;
      player_positions_[i].y = ni;
    }
  }
}