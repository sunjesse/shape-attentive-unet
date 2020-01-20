
"""
Created on Thu Oct 26 11:23:47 2017
@author: Utku Ozbulak - github.com/utkuozbulak
"""
import torch
from torch.nn import ReLU

from misc_functions import (get_example_params,
                            convert_to_grayscale,
                            save_gradient_images,
                            get_positive_negative_saliency)

class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.list = ["<class 'torch.nn.modules.container.Sequential'>", "<class 'models.models.DecoderBlock'>", 
                    "<class 'torchvision.models.densenet._DenseLayer'>", "<class 'torchvision.models.densenet._DenseBlock'>"]
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        #for m in self.model.modules():
        #   print(type(m))
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
            self.grad_out = grad_out[0]
            
        # Register hook to the first layer
        first_layer = self.model.conv1[0]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)    
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)
        
        def loop_and_hook_relus(module, layer):
            try:
                if type(layer) == ReLU:
                    #print(type(layer))
                    layer.register_backward_hook(relu_backward_hook_function)
                    layer.register_forward_hook(relu_forward_hook_function)
                #we do string since we cannot import decoderblock from models.models due to concurrent importing b/w both files.
                elif str(type(layer)) in self.list:
                    #print(type(layer))
                    for i in layer:
                        module.loop_and_hook_relus(i)
                return
            except:
                return

        for mod in self.model.modules():
            loop_and_hook_relus(self.model, mod)

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


