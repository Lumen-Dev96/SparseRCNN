import torch
import numpy as np
import pickle
num_class = 2
device = torch.device('cpu')
pretrained_weights = torch.load('r50_100pro_3x_model.pth', map_location=device)


pretrained_weights["head.head_series.0.class_logits.weight"].resize_(num_class, 256)
pretrained_weights["head.head_series.0.class_logits.bias"].resize_(num_class)
pretrained_weights["head.head_series.1.class_logits.weight"].resize_(num_class, 256)
pretrained_weights["head.head_series.1.class_logits.bias"].resize_(num_class)
pretrained_weights["head.head_series.2.class_logits.weight"].resize_(num_class, 256)
pretrained_weights["head.head_series.2.class_logits.bias"].resize_(num_class)
pretrained_weights["head.head_series.3.class_logits.weight"].resize_(num_class, 256)
pretrained_weights["head.head_series.3.class_logits.bias"].resize_(num_class)
pretrained_weights["head.head_series.4.class_logits.weight"].resize_(num_class, 256)
pretrained_weights["head.head_series.4.class_logits.bias"].resize_(num_class)
pretrained_weights["head.head_series.5.class_logits.weight"].resize_(num_class, 256)
pretrained_weights["head.head_series.5.class_logits.bias"].resize_(num_class)
torch.save(pretrained_weights, "model_%d.pth" % num_class)