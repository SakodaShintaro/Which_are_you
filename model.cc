#include "model.h"

Network::Network(int64_t input_size, int64_t output_size, int64_t num_layers, int64_t hidden_size)
    : input_size_(input_size), num_layers_(num_layers), hidden_size_(hidden_size) {
  using namespace torch::nn;
  first_layer_ = register_module("first_layer_", Linear(input_size, hidden_size));
  TransformerEncoderLayer layer = TransformerEncoderLayer(hidden_size, 4);
  transformer_ = register_module("transformer_", TransformerEncoder(layer, 1));
  policy_head_ = register_module("policy_head_", Linear(hidden_size, output_size));
  value_head_ = register_module("value_head_", Linear(hidden_size, 1));
  resetState();
}

std::tuple<torch::Tensor, torch::Tensor> Network::forward(torch::Tensor x) {
  // xのshapeは(seq_len, batch, input_size)
  // 出力shape policy:(seq_len, batch, POLICY_DIM) value(seq_len, batch, 1)

  const int64_t seq_len = x.size(0);

  const torch::Device& device = transformer_->parameters().front().device();
  x = x.view({-1, 1, input_size_});

  // 過去の情報と結合
  input_history_.push_back(x);
  x = torch::cat(input_history_, 0);
  x = x.to(device);

  // 1層目
  x = first_layer_(x);

  // maskを作る
  torch::Tensor mask = torch::ones({seq_len, seq_len}).to(device);
  mask = torch::triu(mask);
  mask = mask.transpose(0, 1);

  torch::Tensor output = transformer_->forward(x, mask);
  const int64_t output_len = output.size(0);
  output = output.slice(0, output_len - seq_len, output_len);

  torch::Tensor policy = policy_head_->forward(output);
  torch::Tensor value = value_head_->forward(output);

  return std::make_tuple(policy, value);
}

void Network::resetState() { input_history_.clear(); }