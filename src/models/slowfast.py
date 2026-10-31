import torch
import torch.nn as nn

class SlowFastDriverDistraction(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        """
        Custom SlowFast R50 architecture for Driver Distraction Detection.
        - num_classes: 10 (corresponding to the 10 behavioral classes in State Farm/AUCDD datasets)
        """
        super(SlowFastDriverDistraction, self).__init__()
        
        # Load the official SlowFast R50 model pre-trained on Kinetics-400
        self.backbone = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=pretrained)
        
        # TODO: Modify the classification head for fine-tuning.
        # Replace the final fully connected layer to reduce output from 400 classes to 10 classes.
        # (This will be implemented and trained during the Final Project phase).
        # Example: self.backbone.blocks[6].proj = nn.Linear(in_features=2304, out_features=num_classes)

    def forward(self, x):
        """
        Forward pass for the dual-pathway network.
        - x: A list containing 2 tensors -> [slow_pathway_frames, fast_pathway_frames]
        """
        return self.backbone(x)

if __name__ == "__main__":
    # Dry-run model initialization
    model = SlowFastDriverDistraction(num_classes=10)
    print("Successfully initialized SlowFast architecture for Driver Distraction (10 classes).")
    # print(model) # Uncomment to inspect the network layers