#ifndef WHICH_ARE_YOU_ACTION_H_
#define WHICH_ARE_YOU_ACTION_H_

#include "model.h"
#include "state.h"

class Agent {
 public:
  Agent();
  Action SelectAction(const State& state);
  void Train(const Episode& episode);

 private:
  AgentLSTM lstm_;
};

#endif