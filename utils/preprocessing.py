import torch
import torch.nn as nn
import models.mtcnn as mtc
import utils.conversions as con


class Preprocessor(nn.Module):
    def __init__(self, device=torch.device('cpu'), extension_factor=1.2):
        super().__init__()
        self.mtcnn = mtc.MTCNN(
                                image_size=160, margin=0, min_face_size=20,
                                thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
                                device=device
                              )
        
        self.extension_factor = extension_factor
        self.device = device
        
    def forward(self, x):
        x_aligned, prob, box = self.mtcnn(x, return_prob=True, return_boxes=True)
        box = box[0]
        # box has format left, upper, right, lower
        box_outer = self.get_extended_box(box, x)

        x = x.crop(tuple(box_outer))
        x = con.PIL_to_torch(x)
        x = x.unsqueeze(0).to(self.device)
        return x, box_outer, box

    def get_extended_box(self, box, img):
        factor = self.extension_factor
        W, H = img.size
        w, h = box[2]-box[0], box[3]-box[1]
        xc = box[0] + w/2
        yc = box[1] + h/2

        width_factor = min(factor, 2*xc/w, 2*(W-xc)/w)
        height_factor = min(factor, 2*yc/h, 2*(H-yc)/h)

        w2, h2 = width_factor*w/2, height_factor*h/2
        return (xc-w2, yc-h2, xc+w2, yc+h2)
    
    def to(self, device):
        self.device = device
        return super().to(device)
    