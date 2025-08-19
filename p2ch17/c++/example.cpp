#include <iostream>
#include <vector>

#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/torch.h>

void simple_example() {
  auto model = torch::nn::Linear(10, 5);
  auto input = torch::randn({1, 10});
  auto output = model->forward(input);
  std::cout << output << std::endl;
}

struct Net : torch::nn::Module {
  Net() {
    // Input size 784 (28x28 image), hidden layer 128, output 10 classes
    fc1 = register_module("fc1", torch::nn::Linear(784, 128));
    fc2 = register_module("fc2", torch::nn::Linear(128, 10));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(fc1->forward(x));
    x = fc2->forward(x);
    return torch::log_softmax(x, /*dim=*/1);
  }

  torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

void model_train_example() {
  auto model = std::make_shared<Net>();
  torch::optim::SGD optimizer(model->parameters(), /*lr=*/0.01);

  // Create fake data
  auto x = torch::randn({64, 784});
  auto y = torch::randint(0, 10, {64});

  model->train();
  for (size_t epoch = 0; epoch < 3; ++epoch) {
    auto prediction = model->forward(x);
    auto loss = torch::nll_loss(prediction, y);
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    std::cout << "Epoch: " << epoch << " | Loss: " << loss.item<float>()
              << std::endl;
  }

  // Inference
  model->eval();
  torch::NoGradGuard no_grad;
  auto test_input = torch::randn({1, 784});
  auto output = model->forward(test_input);
  auto predicted = output.argmax(1);

  std::cout << "Prediction: " << predicted.item<int64_t>() << std::endl;
}

void aoti_inductor_example() {
  c10::InferenceMode mode;

  std::vector<torch::Tensor> inputs = {torch::tensor(
      {{11496, 24488, 250, 9457,  19451, 11873, 45992, 24488, 250,   813,
        917,   24488, 250, 30613, 24488, 250,   10067, 2730,  24488, 250,
        6525,  24488, 250, 12571, 24488, 250,   70,    2286}},
      torch::kInt32)};

  torch::inductor::AOTIModelPackageLoader loader("./llm.pt2");
  std::vector<torch::Tensor> outputs = loader.run(inputs);
  std::cout << "Result from the first inference:" << outputs << std::endl;
}

int main() {
  simple_example();
  model_train_example();
  aoti_inductor_example();
  return 0;
}
