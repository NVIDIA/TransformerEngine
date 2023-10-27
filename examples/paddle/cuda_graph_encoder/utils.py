import os
import signal
import paddle
import numpy as np

def print_tensor(t, name=None, print_place=True, print_ptr=True, print_hash=True, hash=None):
    output = []
    if name: output.append(name)
    if hash is None: hash = lambda t: float((t * 10).sum())
    
    if t is None:
        print(f"{name} is None")
    else:
        if print_place: output.append(f"place = {t.place}")
        if print_ptr: output.append(f"ptr = {hex(t.data_ptr())}")
        if print_hash: output.append(f"hash = {hash(t)}")
        print(" | ".join(output))


def print_model_grad(model):
    print("======================== start print weight ========================")
    for idx, (name, param) in enumerate(list(model.named_parameters())[::-1]):
        print_tensor(param, name, print_ptr=False)
        print_tensor(param.grad, f"{name}@GRAD", print_ptr=False)
        if idx > 10:
            break


class gpuTimer:
    def __init__(self, name=""):
        self.start_event = paddle.device.cuda.Event()
        self.end_event = paddle.device.cuda.Event()
        self.elapsed_time = None
        self.name = name

    def __enter__(self):
        self.current_stream = paddle.device.current_stream()
        print("Current Stream: ", self.current_stream)

        self.start_event.record(self.current_stream)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_event.record(self.current_stream)

    def get_elp_time(self):
        self.elapsed_time = self.start_event.elapsed_time(self.end_event)
        if len(self.name):
            print(f"[{self.name}]", end=" ")
        print(f"Elapsed time: {self.elapsed_time:.4f} ms")



def handler(signum, frame):
    print("Signal handler called with signal", signum)

signal.signal(signal.SIGUSR1, handler)
signal.signal(signal.SIGUSR2, handler)

def set_seed():
    paddle.seed(102)
    np.random.seed(102)
    paddle.set_flags(
        {
            "FLAGS_cudnn_deterministic": True,
        }
    )



class Reporter:
    _instance = None
    history = dict()
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Reporter, cls).__new__(cls)
        return cls._instance
    
    def report(self, key, t):
        self.history[key] = t.clone()

    def reset(self):
        self.history = dict()

    def print(self):
        for name, t in self.history.items():
            print_tensor(t, name=name, print_ptr=False)
