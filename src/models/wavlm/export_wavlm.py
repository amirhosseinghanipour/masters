import torch
from transformers import WavLMModel

model = WavLMModel.from_pretrained("microsoft/wavlm-base")
model.eval()
torch.save(model.state_dict(), "wavlm-base.pt")