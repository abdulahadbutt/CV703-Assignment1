import torch 
from torch import nn
from transformers import AutoImageProcessor, ConvNextV2ForImageClassification

# from convnextv2 import convnextv2_base

class ConvNextV2(nn.Module):
    def __init__(self, num_classes, weights_path='facebook/convnextv2-base-22k-224'):
        super().__init__()
        #NOTE: had to remove use_fast=True from .from_pretrained
        self.preprocessor = AutoImageProcessor.from_pretrained(weights_path)
        self.base_model = ConvNextV2ForImageClassification.from_pretrained(weights_path)

        in_features = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Linear(
            in_features=in_features,
            out_features=num_classes,
            bias=True
        )

    def forward(self, x):
        # print(x.shape)
        x = self.base_model(x)
        # exit()
        return x 
    
if __name__ == '__main__':
    preprocessor = AutoImageProcessor.from_pretrained("facebook/convnextv2-base-22k-224")
    base_model = ConvNextV2ForImageClassification.from_pretrained("facebook/convnextv2-base-22k-224")
    print(base_model)
    pass 
    

