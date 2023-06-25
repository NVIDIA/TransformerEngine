# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Example of BF16 training"""

import paddle
from paddle import nn
import paddle.optimizer as optim

import transformer_engine.paddle as te


# Define the network architecture
class SimpleNetwork(nn.Layer):
    """Simple network consisting of two linear layers"""

    def __init__(self):
        super().__init__()
        self.linear1 = te.Linear(10, 20)
        self.linear2 = te.Linear(20, 1)

    def forward(self, x):
        x = paddle.nn.functional.relu(self.linear1(x))
        x = self.linear2(x)
        return x


# Create an instance of the network
model = SimpleNetwork()

# Generate random input and target data
input_data = paddle.randn([100, 10])
target_data = paddle.randn([100, 1])

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(learning_rate=0.01, parameters=model.parameters())

# Train the network
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.clear_grad()    # Clear gradients

    # Use AMP (BF16) training
    with paddle.amp.auto_cast(dtype='bfloat16'):    # pylint: disable=not-context-manager
        with te.fp8_autocast(enabled=False):
            output = model(input_data)

        loss = criterion(output, target_data)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    # Print the loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {float(loss):.4f}")
