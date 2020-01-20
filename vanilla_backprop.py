
"""
Created on Thu Oct 26 11:19:58 2017
@author: Utku Ozbulak - github.com/utkuozbulak
"""
import torch

from misc_functions import get_example_params, convert_to_grayscale, save_gradient_images


class VanillaBackprop():
    """
        Produces gradients generated with vanilla back propagation from the image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        # Hook the first layer to get the gradient
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
            self.grad_out = grad_out[0]
        
        # Register hook to the first layer
        #print(list(self.model.modules())[0])
        first_layer = self.model.conv1[0]
        first_layer.register_backward_hook(hook_function)

    def generate_gradients(self, input_image, target):
        # Forward
        model_output = self.model(input_image) #segSize here doesn't matter, just use it to indicate inference.
        # Zero grads
        self.model.zero_grad()
        # Target for backprop
        
        tmp = []
        for i in range(2):
            tmp.append((target==i).unsqueeze(0).float())
        one_hot_output = torch.cat(tmp, dim=0).float().unsqueeze(0)
       
        #one_hot_output.requires_grad = True
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.cpu().data.numpy()[0]
        return gradients_as_arr


