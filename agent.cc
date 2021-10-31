#include "agent.h"

#include <random>

Agent::Agent() : lstm_(1, 1, 1, 1) {}

Action Agent::SelectAction(const State& state) {
  std::mt19937_64 engine(std::random_device{}());
  std::uniform_int_distribution<int64_t> dist(0, kActionNum - 1);
  return Action(dist(engine));
}
