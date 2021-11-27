#ifndef WHICH_ARE_YOU_ACTION_H_
#define WHICH_ARE_YOU_ACTION_H_

#include "model.h"
#include "state.h"

class Agent {
 public:
  Agent();
  Action SelectAction(const State& state, bool first_action);
  std::tuple<torch::Tensor, torch::Tensor> Train(const Episode& episode);
  std::vector<torch::Tensor> Parameters();
  void ResetLSTM() { network_.resetState(); }

 private:
  Network network_;
};

#endif