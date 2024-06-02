import torch

from inference_mnist_model import Net,SimpleNN


def main():
  input_size = 3
  output_size =2
  pytorch_model = SimpleNN(input_size,output_size)
  # pytorch_model.load_state_dict(torch.load('pytorch_model.pt'))
  pytorch_model.eval()
  dummy_input = torch.randn(1, input_size)
  # torch.onnx.export(pytorch_model, dummy_input, 'onnx_model.onnx', verbose=False)
  torch.onnx.export(pytorch_model, dummy_input, "simple_nn.onnx", input_names=['input'], output_names=['output'], verbose=True)

if __name__ == '__main__':
  main()
