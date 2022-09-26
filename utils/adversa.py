import torch
import torch.nn as nn
import json
import subprocess
from tinydb import TinyDB
import uuid

import shutil
from PIL import Image, ImageDraw, ImageFilter


from torchvision.transforms.functional import adjust_contrast, adjust_saturation, adjust_brightness

import utils.conversions as con
from utils.preprocessing import Preprocessor


class AdversaModel(nn.Module):
    def __init__(self, api_key, save_name=None):
        super().__init__()
        MLSEC_API_KEY = api_key
        self.url = f"https://api.mlsec.io/api/facerecognition/submit_sample/?api_token={MLSEC_API_KEY}"
        self.folder = "/scratch/ameinke03/"
        if save_name is None:
            self.save_name = str(uuid.uuid1())
        else:
            self.save_name = save_name
        self.save_name += '.png'
        
    def forward(self, x):
        raise NotImplemented()
        
    def forward_class(self, x, y_target, y_source=0):
        img = con.torch_to_PIL(x)
        image_path = self.folder + self.save_name
        img.save(image_path)
        url = self.url + f"&source={y_source}&target={y_target}"
        response = subprocess.run(['curl', '-X', 'POST', '--data-binary',  f'@{image_path}', f'{url}'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        model_result = self.parse_response(response)
        return model_result
    
    def parse_response(self, response):
        result = response.stdout.decode("utf-8").split('\n')[0]
        model_result = json.loads(result)
        return model_result
    
    
class SquareAttackWrapper(nn.Module):
    def __init__(self, model, y_source, y_target, weights=torch.ones(2), verbose=False):
        super().__init__()
        self.model = model
        self.weights = weights
        self.y_source = y_source
        self.y_target = y_target
        self.verbose = verbose
        
    def forward(self, x):
        response = self.model.forward_class(x[0], y_target=self.y_target, y_source=self.y_source)
        score = compute_score_from_output(response)
        if self.verbose:
            print(f'Response:\t{response}')
            print(f'Score:\t{score}')
        return torch.tensor([4, score]).unsqueeze(0)
    
    
# random search with attack parameters: position_x, position_y, width, height, lambda, mask_softener
class ImageInterpolator(nn.Module):
    def __init__(self, img_source, img_target):
        super().__init__()
        self.img_s = img_source
        self.img_t = img_target
        self.x_s = con.PIL_to_torch(self.img_s)
        
    def forward(self, v):
        box_s = [int(v[i].item()) for i in range(4)]
        box_t = [int(v[i].item()) for i in range(4,8)]
        border_softening = int(v[8])
        lam = torch.sigmoid(v[9])        
        
        img_t_cropped = self.img_t.crop(tuple(box_t))
        width = box_s[2] - box_s[0]
        height = box_s[3] - box_s[1]
        
        x_t_cropped = con.PIL_to_torch(img_t_cropped.resize((width,height)))
        x_t = torch.zeros_like(self.x_s)
        x_t[:,box_s[1]:box_s[3],box_s[0]:box_s[2]] = x_t_cropped
        
        mask = self.generate_mask(box_s, border_softening)
        
        # mask -= self.create_gaussian_dot(v[16:22])
        mask = torch.clip(mask, 0, 1)
        
        # x_s_transformed = self.transform(v[10:13], self.x_s)
        x_s_transformed = self.x_s
        x_t_transformed = self.transform(v[13:16], x_t)
        sample = mask*(lam*x_t_transformed + (1-lam)*x_s_transformed) + (1-mask)*x_s_transformed
        
        return sample
    
    def transform(self, param_vec, sample):
        contrast_factor = param_vec[0].exp()
        saturation_factor = param_vec[1].exp()
        brightness_factor = param_vec[2].exp()
        
        transformed = adjust_contrast(sample, contrast_factor=contrast_factor)
        transformed = adjust_saturation(transformed, saturation_factor=saturation_factor)
        transformed = adjust_brightness(transformed, brightness_factor=brightness_factor)
        
        return transformed
    
    def create_gaussian_dot(self, param_vec):
        pos_x, pos_y = param_vec[0], param_vec[1]
        var = param_vec[2]**2 + 1e-1
        color = param_vec[3:6]
        
        shape = self.x_s.shape
        xx, yy = torch.arange(shape[1]), torch.arange(shape[2])
        gauss = ( -((xx[:,None]-pos_x)**2 + (yy[None,:]-pos_y)**2) / var ).exp()
        gauss_dot = color[:,None,None] * gauss[None,:,:]
        return gauss_dot
        
        
    def generate_mask(self, box, border_softening):
        mask = torch.zeros_like(self.x_s)
        mask[:, box[1]:box[3], box[0]:box[2]] = 1

        for i in range(border_softening):
            smoothed_value = 1-float(i)/border_softening
            mask[:, box[1]+i, box[0]:box[2]] = 1 - smoothed_value
            mask[:, box[1]:box[3], box[0]+i] = 1 - smoothed_value
            mask[:, box[3]-i, box[0]:box[2]] = 1 - smoothed_value
            mask[:, box[1]:box[3], box[2]-i] = 1 - smoothed_value

        for i in range(border_softening):
            for j in range(border_softening):
                smoothed_value = max([float(i+j)/border_softening-1, 0])
                mask[:, box[1]+i, box[0]+j] = smoothed_value
                mask[:, box[1]+i, box[2]-j] = smoothed_value
                mask[:, box[3]-i, box[0]+j] = smoothed_value
                mask[:, box[3]-i, box[2]-j] = smoothed_value

        return mask
#     def generate_mask(self, box, border_softening):
#         mask = torch.zeros_like(self.x_s)
#         mask[:, box[1]:box[3], box[0]:box[2]] = 1

#         for i in range(border_softening):
#             smoothed_value = 1-float(i)/border_softening
#             mask[:, box[1]+i, box[0]:box[2]] = 1 - smoothed_value
#             mask[:, box[1]:box[3], box[0]+i] = 1 - smoothed_value
#             mask[:, box[3]-i, box[0]:box[2]] = 1 - smoothed_value
#             mask[:, box[1]:box[3], box[2]-i] = 1 - smoothed_value

#             mask[:, box[1]:box[1]+i, box[0]:box[0]+i] = 0.1
#             mask[:, box[1]:box[1]+i, box[2]-i:box[2]] = 0.1
#             mask[:, box[3]-i:box[3], box[0]:box[0]+i] = 0.1
#             mask[:, box[3]-i:box[3], box[2]-i:box[2]] = 0.1
            
#         return mask

def compute_score_from_output(output):
    conf = output['confidence']
    stealth = output['stealthiness']
    if conf<0.01:
        return -1. + stealth
    if conf+stealth < 1.:
        return conf
    elif stealth<0.5:
        return conf + stealth 
    elif conf<1.:
        return 2. + conf
    else:
        return 2. + conf + stealth

    
class RandomSearchWrapper(nn.Module):
    def __init__(self, model, interpolator, source_id, target_id):
        super().__init__()
        self.model = model
        self.interpolator = interpolator
        self.source_id = source_id
        self.target_id = target_id
        
    def forward(self, v):
        sample = self.interpolator(v)
        model_output = self.model.forward_class(sample, self.target_id, y_source=self.source_id)
        return compute_score_from_output(model_output)
        
        
class RandomSearchAttack():
    def __init__(self, wrapper, epochs=20, magnitude=50, verbose=True, step_size=None):
        self.wrapper = wrapper
        self.magnitude = magnitude
        self.epochs = epochs
        self.verbose = verbose
        
        magnitude = self.magnitude
        if step_size is None:
            self.step_size = torch.tensor([20.,20.,20,20, 
                                      20.,20,20,20, 
                                      5, .1] + 6*[0.1]
                                     + [50, 50, 2, .2, .2, .2]) / magnitude
        else:
            self.step_size = step_size
        
    def run(self, v):
        prev_point = v

        prev_value = self.wrapper(prev_point)
        if self.verbose:
            print(f'Starting value: {prev_value}')

        magnitude = self.magnitude
        step_size = self.step_size

        epochs = self.epochs
        for i in range(5*epochs):
            if i==epochs:
                magnitude = 30
            elif i==2*epochs:
                magnitude = 10
            elif i==3*epochs:
                magnitude = 3
            elif i==4*epochs:
                magnitude = 1
            elif i==5*epochs:
                magnitude = .5

            delta = magnitude*step_size*torch.randn(len(v))
            
            try:
                new_point = prev_point + delta
                value = self.wrapper(new_point)
            except:
                continue

            if self.verbose:
                # print('')
                # print(i)
                print(f'{value}')
                # print('{new_point[:4]}\n{new_point[4:8]}\n{new_point[8:]}')

            if value>prev_value:
                prev_value = value
                prev_point = prev_point + delta
                
            if value>3.0:
                break;

        return prev_point

    
class AttackScheduler():
    def __init__(self, api_key, extension_factor=1.):
        self.model = AdversaModel(api_key)
        self.preprocessor = Preprocessor(extension_factor=1.)
        self.db = TinyDB('evals/results.json')
        
    def attack_pair(self, source_id, target_id, verbose=True, epochs=None):
        if epochs is None:
            epochs = 100
        
        assert source_id!=target_id
        img_s = Image.open(f'adversa_data/{source_id}_{source_id}.png')
        img_t = Image.open(f'adversa_data/{target_id}_{target_id}.png')

        interpolator = ImageInterpolator(img_s, img_t)
        
        _, box_s, _ = self.preprocessor(img_s)
        _, box_t, _ = self.preprocessor(img_t)

        initial_point = torch.cat([torch.tensor(box_s), 
                                   torch.tensor(box_t), 
                                   torch.tensor([50, 2.0]), 
                                   torch.zeros(6), 
                                   torch.tensor([100,100,5,.5,.5,.5])
                                  ], 0)
        
        wrapper = RandomSearchWrapper(self.model, interpolator, source_id, target_id)
        attack = RandomSearchAttack(wrapper, verbose=verbose, epochs=epochs)

        final_point = attack.run(initial_point)
        sample = interpolator(final_point)
        
        output = self.model.forward_class(sample, target_id, y_source=source_id)
        
        self.store_output(output, source_id, target_id, final_point)
        
        
    def store_output(self, output, source_id, target_id, final_point):
        doc_id = self.get_doc_id(source_id, target_id)
        entry = self.db.get(doc_id=doc_id)
        assert entry['Name'] == f'{source_id}_{target_id}'
        
        if output['success'] and output['confidence']>entry['confidence']:
            entry['confidence'] = output['confidence']
            entry['success'] = output['success']
            entry['stealthiness'] = output['stealthiness']
            entry['v'] = final_point.tolist()
            
            shutil.copyfile(self.model.folder + self.model.save_name, 
                            f'adversa_results/{source_id}_{target_id}.png')
            self.db.update(entry, doc_ids=[doc_id])
            print('Replaced old entry')
        print(entry)
        
    def get_doc_id(self, source_id, target_id):
        doc_id = 0
        stop = False
        for i in range(10):
            if stop:
                break
            for j in range(10):
                if i==j:
                    continue
                else:
                    doc_id += 1
                if i==source_id and j==target_id:
                    stop = True
                    break
        return doc_id
    
    def summarize_results(self):
        conf = sum([el['confidence'] for el in self.db.all()])
        stealth = sum([el['stealthiness'] for el in self.db.all()])
        print(f'Confidence: {conf}')
        print(f'Stealthiness: {stealth}')
        return conf, stealth

    
class AlternateAttackScheduler(AttackScheduler):       
    def attack_pair(self, source_id, target_id, verbose=True, epochs=None):
        if epochs is None:
            epochs = 100
        
        assert source_id!=target_id
                
        wrapper = AlternateRandomSearchWrapper(self.model, source_id, target_id)
        
        step_size = torch.tensor([3, 3, 3, 3, .2, .2])
        attack = RandomSearchAttack(wrapper, verbose=verbose, epochs=epochs, step_size=step_size)
        
        img_s = Image.open(f'adversa_data/{source_id}_{source_id}.png')
        w, h = img_s.size[0], img_s.size[1]
        x, y = 0, 0
        angle = 0
        blur = 0
        
        initial_point = torch.tensor([w, h, x, y, angle, blur])
        final_point = attack.run(initial_point)
        sample = wrapper.interpolate(final_point)
        
        output = self.model.forward_class(sample, target_id, y_source=source_id)
        
        self.store_output(output, source_id, target_id, final_point)
        
        
class AlternateRandomSearchWrapper(nn.Module):
    def __init__(self, model, source_id, target_id):
        super().__init__()
        self.model = model
        self.source_id = source_id
        self.target_id = target_id
        self.img_s = Image.open(f'adversa_data/{source_id}_{source_id}.png')
        self.img_t = Image.open(f'adversa_data/{target_id}_{target_id}.png')
        self.mask = Image.open(f'segmentation_masks/{target_id}_{target_id}.png').convert("L") 
        self.EPS = 1
        
    def forward(self, v):
        sample = self.interpolate(v)
        model_output = self.model.forward_class(sample, self.target_id, y_source=self.source_id)
        return compute_score_from_output(model_output)
    
    def interpolate(self, v):
        mask_im_blur = self.mask.filter(ImageFilter.GaussianBlur(int(v[5].abs().item()+self.EPS)))
        back_im = self.img_s.copy()
        w, h = int(v[0].abs().item()+self.EPS), int(v[1].abs().item()+self.EPS) #img_t.size
        x, y = int(v[2]), int(v[3]) #0, 0
        angle = v[4]
        new_img_t = self.img_t.resize((w,h)).rotate(angle)
        new_mask_im_blur = mask_im_blur.resize((w,h)).rotate(angle)
        back_im.paste(new_img_t, (x, y), new_mask_im_blur)
        
        sample = con.PIL_to_torch(back_im)
        return sample
