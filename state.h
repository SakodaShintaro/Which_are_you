#ifndef WHICH_ARE_YOU_STATE_H_
#define WHICH_ARE_YOU_STATE_H_

#include <cstdint>
#include <iostream>
#include <vector>

enum Action { kUp, kRight, kDown, kLeft, kActionNum };
enum SquareKind { kEmpty, kWall };

struct Position {
    int64_t x, y;
};

class State {
 public:
  static constexpr int64_t kBoardSize = 10;
  static constexpr int64_t kPlayerNum = 4;
  State();
  friend std::ostream& operator<<(std::ostream& ost, const State& state);
  void Init();
  void Step(Action a);

 private:
  std::vector<std::vector<char>> board_;
  std::vector<Position> player_positions_;
};

#endif