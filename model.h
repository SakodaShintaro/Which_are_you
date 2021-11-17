#ifndef WHICH_ARE_YOU_MODEL_H_
#define WHICH_ARE_YOU_MODEL_H_

#include <torch/torch.h>

class AgentLSTM : public torch::nn::Module {
 public:
  AgentLSTM(int64_t input_size, int64_t output_size, int64_t num_layers = 1, int64_t hidden_size = 256);

  //入力を受けて1ステップ分LSTMの推論を進める関数:これを直に触って使うのはやめた方が良さそう
  torch::Tensor forward(torch::Tensor x);

  //内部状態をリセットする関数:上のforwardを使わないようにできれば自然とこの関数も不要になる
  void resetState();

  // shape(seq_len, batch, input_size)のinputを入力として(seq_len, batch, output_size)をoutputする関数
  torch::Tensor forwardSequence(const torch::Tensor& input);

 private:
  int64_t input_size_;
  int64_t num_layers_;
  int64_t hidden_size_;
  torch::nn::Linear first_layer_{nullptr};
  torch::nn::LSTM lstm_{nullptr};
  torch::nn::Linear final_layer_{nullptr};
  torch::Tensor h_;
  torch::Tensor c_;
};

#endif  // WHICH_ARE_YOU_MODEL_H_