#include "interface.h"
#include "state.h"
#include "agent.h"

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

    Action a = kActionNum;
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
    state.Step(a);
    std::cout << state;
  }
}