#ifndef WHICH_ARE_YOU_STATE_H_
#define WHICH_ARE_YOU_STATE_H_

#include <cstdint>
#include <iostream>
#include <vector>

static constexpr int64_t kPlayerNum = 2;

enum Action {
  kUp,
  kRight,
  kDown,
  kLeft,
  kStay,
  kMoveActionNum,
};

constexpr int64_t kDi[kMoveActionNum] = {-1, 0, 1, 0};
constexpr int64_t kDj[kMoveActionNum] = {0, 1, 0, -1};

enum SquareKind { kEmpty, kWall };

struct Position {
  int64_t x, y;
  bool operator==(const Position& rhs) const { return (x == rhs.x && y == rhs.y); }
};

struct Episode {
  std::vector<std::vector<float>> state_features;
  std::vector<Action> actions;
  float reward;
  void Init() {
    state_features.clear();
    actions.clear();
    reward = 0.0f;
  }
};

class State {
 public:
  static constexpr int64_t kBoardWidth = 5;
  static constexpr int64_t kBoardSize = kBoardWidth * kBoardWidth;
  State();
  friend std::ostream& operator<<(std::ostream& ost, const State& state);
  void Init();
  bool Step(Action a);
  std::vector<float> GetFeature() const;
  Episode GetEpisode() const;

 private:
  std::vector<std::vector<char>> board_;
  std::vector<Position> player_positions_;
  int64_t true_player_;
  Action pre_action_;
  Episode episode_;
  Position good_position_;
};

static constexpr int64_t kInputDim = State::kBoardSize * (kPlayerNum + 2) + kMoveActionNum;

#endif