#include "model.h"

AgentLSTM::AgentLSTM(int64_t input_size, int64_t output_size, int64_t num_layers, int64_t hidden_size)
    : input_size_(input_size), num_layers_(num_layers), hidden_size_(hidden_size) {
  first_layer_ = register_module("first_layer_", torch::nn::Linear(input_size, hidden_size));
  torch::nn::LSTMOptions option(hidden_size, hidden_size);
  option.num_layers(num_layers);
  lstm_ = register_module("lstm_", torch::nn::LSTM(option));
  final_layer_ = register_module("final_layer_", torch::nn::Linear(hidden_size, output_size));
  h_ = register_parameter("h_", torch::zeros({num_layers_, 1, hidden_size_}), false);
  c_ = register_parameter("c_", torch::zeros({num_layers_, 1, hidden_size_}), false);
  resetState();
}

torch::Tensor AgentLSTM::forward(torch::Tensor x) {
  // lstmは入力(input, (h_0, c_0))
  // inputのshapeは(seq_len, batch, input_size)
  // h_0, c_0は任意の引数で、状態を初期化できる
  // h_0, c_0のshapeは(num_layers_ * num_directions, batch, hidden_size_)
  //出力はoutput, (h_n, c_n)

  const torch::Device& device = lstm_->parameters().front().device();
  x = x.to(device);
  x = x.view({-1, 1, input_size_});

  // 1層目
  x = first_layer_(x);

  // outputのshapeは(seq_len, batch, num_directions * hidden_size)
  auto [output, h_and_c] = lstm_->forward(x, std::make_tuple(h_, c_));
  std::tie(h_, c_) = h_and_c;

  output = final_layer_->forward(output);

  return output;
}

void AgentLSTM::resetState() {
  h_.fill_(0.0);
  c_.fill_(0.0);
}

std::vector<torch::Tensor> AgentLSTM::Parameters() {
  std::vector<torch::Tensor> parameters;
  for (auto p : lstm_->parameters()) {
    parameters.push_back(p);
  }
  for (auto p : final_layer_->parameters()) {
    parameters.push_back(p);
  }
  return parameters;
}