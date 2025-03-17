import torch

from torch.utils.tensorboard import SummaryWriter
from discrim_model import ContextEncoder, Discriminator
# Instantiate your model
# model = ContextEncoder(use_channel_fc=True)
discrim = Discriminator()
# Create a dummy input tensor
# dummy_input = torch.randn(256, 3, 227, 227)  # Adjust dimensions as needed
dummy_input_discrim = torch.randn(256, 3, 227, 227)  # 256 batch size

# Initialize TensorBoard writer
writer = SummaryWriter()

# Add the model graph to TensorBoard
# writer.add_graph(model, dummy_input)
writer.add_graph(discrim, dummy_input_discrim)
writer.close()



