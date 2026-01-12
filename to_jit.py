import torch
from model import *

def transfer_weights(path, network):
    checkpoint = torch.load(path)
    
    prefix = "0.module."

    state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]
            state_dict[new_key] = v
        else: state_dict[k] = v

    if state_dict:
        network.load_state_dict(state_dict)
        print("Successfully loaded pre-trained weights!")
        
    return network
    
encoder = VisionEncoder()
model = RecurrentActorNetwork(
        3, 
        2, 
        encoder=encoder,
        lstm_hidden_size=256,
        memory_length=20,
        memory_stride=5
    )
model = transfer_weights("actor_BrandsHatch.pt", model)

# Save using torch,jit
model.eval()
model = torch.jit.script(model)
torch.jit.save(model, 'actor_BrandsHatch_jit.pt')