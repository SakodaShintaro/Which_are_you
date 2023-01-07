#include "state.h"

#include <cassert>
#include <random>
#include <tuple>

State::State() : board_(kBoardWidth, std::vector<char>(kBoardWidth, '.')), player_positions_(kPlayerNum) { Init(); }

std::ostream& operator<<(std::ostream& ost, const State& state) {
  ost << "真のプレイヤー: " << char('A' + state.true_player_) << std::endl;
  for (int64_t i = 0; i < State::kBoardWidth; i++) {
    for (int64_t j = 0; j < State::kBoardWidth; j++) {
      ost << state.board_[i][j];
    }
    ost << std::endl;
  }
  return ost;
}

void State::Init() {
  // 壁と床の配置
  for (int64_t i = 0; i < kBoardWidth; i++) {
    for (int64_t j = 0; j < kBoardWidth; j++) {
      if (i == 0 || i == kBoardWidth - 1 || j == 0 || j == kBoardWidth - 1) {
        board_[i][j] = '#';
      } else {
        board_[i][j] = '.';
      }
    }
  }

  // 各プレイヤーの位置をランダムに決定する
  std::mt19937_64 engine(std::random_device{}());
  std::uniform_int_distribution<int64_t> dist_pos(1, kBoardWidth - 2);
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

  // 前の行動を初期化
  pre_action_ = kNullAction;

  // エピソード初期化
  episode_.Init();
}

bool State::Step(Action a) {
  episode_.state_features.push_back(GetFeature());
  episode_.actions.push_back(a);
  pre_action_ = a;
  if (a >= kMoveActionNum) {
    //この場合、移動を止めて正解を答える行動ということ
    const int64_t answer = a - kMoveActionNum;
    const float reward = (answer == true_player_ && episode_.actions.size() != 1);
    const bool correctness = (answer == true_player_);
    episode_.reward = (correctness ? reward / (episode_.actions.size() - 1) : -0.1);
    episode_.correctness = correctness;
    return true;
  }

  if (episode_.actions.size() >= 10) {
    episode_.reward = -0.1;
    episode_.correctness = 0;
    return true;
  }

  std::mt19937_64 engine(std::random_device{}());
  std::uniform_int_distribution<int64_t> dist_pos(0, kMoveActionNum - 1);
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
  return false;
}

std::vector<float> State::GetFeature() const {
  // 盤面に関する情報が(kPlayerNum + 1)×H×Wであり、行動に関する情報がkMoveActionNum
  std::vector<float> feature(kInputDim, 0);

  // board C×H×Wの順番で並べる
  for (int64_t i = 0; i < kBoardWidth; i++) {
    for (int64_t j = 0; j < kBoardWidth; j++) {
      const int64_t index = i * kBoardWidth + j;
      switch (board_[i][j]) {
        case '#':
          feature[kPlayerNum * kBoardSize + index] = 1;
          break;
        case '.':
          break;
        default:
          const int64_t ch = board_[i][j] - 'A';
          feature[ch * kBoardSize + index] = 1;
          break;
      }
    }
  }

  // 行動
  feature[(kPlayerNum + 1) * kBoardSize + pre_action_] = 1;

  return feature;
}

Episode State::GetEpisode() const { return episode_; }