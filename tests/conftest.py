import os

# AutoLiRPA/abcrown can crash when TorchScript JIT is enabled in this environment.
os.environ["PYTORCH_JIT"] = "0"
