#ifndef WHICH_ARE_YOU_ACTION_H_
#define WHICH_ARE_YOU_ACTION_H_

#include "model.h"
#include "state.h"

class Agent {
 public:
  Agent();
  Action SelectAction(const State& state);
  void train(float reward);

 private:
  LSTM lstm_;
};

#endif