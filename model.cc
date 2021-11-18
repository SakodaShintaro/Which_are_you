#include "model.h"

#include "state.h"

AgentLSTM::AgentLSTM(int64_t input_size, int64_t output_size, int64_t num_layers, int64_t hidden_size)
    : input_size_(input_size),
      num_layers_(num_layers),
      hidden_size_(hidden_size),
      conv_layers_(State::kBoardWidth / 2, nullptr) {
  using namespace torch::nn;
  const int64_t input_ch = kPlayerNum + 1;
  const int64_t hidden_ch = hidden_size / 2;
  first_layer_ = register_module("first_layer_", Conv2d(Conv2dOptions(input_ch, hidden_ch, 3).bias(true).padding(1)));
  for (int64_t i = 0; i < State::kBoardWidth / 2; i++) {
    conv_layers_[i] = register_module("conv_layers_" + std::to_string(i),
                                      Conv2d(Conv2dOptions(hidden_ch, hidden_ch, 3).bias(true).padding(0)));
  }
  action_encoder_ = register_module("action_encoder_", Linear(kMoveActionNum, hidden_ch));
  LSTMOptions option(hidden_size, hidden_size);
  option.num_layers(num_layers);
  lstm_ = register_module("lstm_", LSTM(option));
  final_layer_ = register_module("final_layer_", Linear(hidden_size, output_size));
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

  // xの元shape : (seq_len, batch_size, input_size)
  x = x.to(device);

  // 分解
  std::vector<torch::Tensor> x_list = x.split(State::kBoardSize * (kPlayerNum + 1), 2);
  torch::Tensor board_x = x_list[0];
  torch::Tensor action_x = x_list[1];

  board_x = board_x.view({-1, kPlayerNum + 1, State::kBoardWidth, State::kBoardWidth});

  // 盤面のエンコード
  board_x = first_layer_(board_x);
  for (auto& l : conv_layers_) {
    board_x = l(board_x);
  }
  board_x = board_x.view({-1, 1, hidden_size_ / 2});

  // 行動のエンコード
  action_x = action_encoder_(action_x);

  // 盤面と行動を連結してLSTMに入れる
  x = torch::cat({board_x, action_x}, 2);

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