import numpy as np
import pickle

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import imagenet_utils

from keras_applications.imagenet_utils import preprocess_input as _preprocess_input
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu


from tensorflow.keras.utils import get_custom_objects
get_custom_objects().update({'swish': keras.layers.Activation(tf.nn.swish)})





def mean_resize(img, size):
    new_w, new_h = size
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
        
    h, w, c = img.shape   
    w_coef = w // new_w
    h_coef = h // new_h
    result = np.zeros((new_h, new_w, c))

    for y in range(new_h):
        for x in range(new_w):
            result[y, x, :] = img[int(y*h_coef) : int((y+1)*h_coef), 
                                 int(x*w_coef) : int((x+1)*w_coef)].mean(axis=(0,1))
            
    return result

def get_tissue_mask(img_RGB):
    img_HSV = rgb2hsv(img_RGB)
    
    background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])
    background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])
    background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])
    
    tissue_RGB = np.logical_not(background_R & background_G & background_B)
    tissue_S = img_HSV[:, :, 1] > threshold_otsu(img_HSV[:, :, 1])
    
    min_R = img_RGB[:, :, 0] > 50
    min_G = img_RGB[:, :, 1] > 50
    min_B = img_RGB[:, :, 2] > 50

    tissue_mask = tissue_S & tissue_RGB & min_R & min_G & min_B
    return tissue_mask



def preprocess_input(x_batch):
    kwargs = {}
    kwargs['backend'] = keras.backend
    kwargs['layers'] = keras.layers
    kwargs['models'] = keras.models
    kwargs['utils'] = keras.utils
    kwargs = {k: v for k, v in kwargs.items() if k in ['backend', 'layers', 'models', 'utils']}

    return _preprocess_input(x_batch,  mode='torch', **kwargs)


class FixedDropout(tf.keras.layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        symbolic_shape = K.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)
    
    

class InferenceDataset:
    def __init__(self, slide, level=0, crop_size=512):
        if min(slide.level_dimensions[level]) <= crop_size:
            level = 0
        
        self.level=level
        self.crop_size = crop_size 
        self.slide = slide
        
        tissue_img = np.array(self.slide.read_region((0,0), self.slide.level_count - 1, self.slide.level_dimensions[-1]).convert('RGB'))
        self.tissue_img = tissue_img
        self.tissue_mask = get_tissue_mask(tissue_img)
        
        self.x_count = self.slide.level_dimensions[self.level][0] // self.crop_size
        self.y_count = self.slide.level_dimensions[self.level][1] // self.crop_size
         
        self.small_tissue_mask = mean_resize(self.tissue_mask, (self.x_count, self.y_count))[..., 0] > 0.1
        
        
        self.tissue_crops_flatten = np.where(self.small_tissue_mask.flatten())[0]


    def __len__(self):
        return self.tissue_crops_flatten.shape[0]

    def get_coords(self, idx):
        curr_number = self.tissue_crops_flatten[idx]
        y_position = int(curr_number // self.x_count)
        x_position = curr_number % self.x_count
        coef = self.slide.level_dimensions[0][0] // self.slide.level_dimensions[self.level][0]
        coords = x_position * self.crop_size*coef, y_position * self.crop_size*coef    
        
        return coords
        

    def get_exps_mask(self, exps):
        exp_mask = np.zeros((self.y_count, self.x_count))   
        for idx, val in enumerate(exps):
            curr_number = self.tissue_crops_flatten[idx]
            y_position = int(curr_number // self.x_count)
            x_position = curr_number % self.x_count           
            exp_mask[y_position, x_position] = val
        
        return exp_mask
        
        
        
    def __getitem__(self, idx):
        curr_number = self.tissue_crops_flatten[idx]
        y_position = int(curr_number // self.x_count)
        x_position = curr_number % self.x_count
        coef = self.slide.level_dimensions[0][0] // self.slide.level_dimensions[self.level][0]
        img = np.array(self.slide.read_region((x_position * self.crop_size*coef, y_position * self.crop_size*coef),
                   self.level,
                   (self.crop_size, self.crop_size)).convert('RGB'))
        
        return img
    
   
    
    
class ProstateInferencer:
    def __init__(self):
        model_base = load_model("model_data/effb5_lev0_99.h5", custom_objects = {"swish": keras.layers.Activation(tf.nn.swish), 'FixedDropout':FixedDropout})
        feat = model_base.layers[-3].output
        self.model = Model(model_base.input, [feat, model_base.output])

        with open('model_data/effb5_lev0_99_512.pickle', 'rb') as f:
            self.cl = pickle.load(f)
            

    def nn_predict(self, inference_dataset):
        embeddings = []
        probas = []
        for k in range(inference_dataset.__len__()):
            crop_img = inference_dataset.__getitem__(k)
            x_batch = np.expand_dims(crop_img, axis=0)
            x_batch = preprocess_input(x_batch)

            pred = self.model.predict(x_batch)
            embeddings.append(pred[0][0])
            probas.append(pred[1][0])

        embeddings = np.array(embeddings)
        probas = np.array(probas)

        if inference_dataset.__len__() == 0:
            embeddings = np.zeros((1, 2048))
            probas = np.zeros((1, 6))

        return embeddings, probas            
            
            
    def predict(self, slide):
        result = {}
        inference_dataset = InferenceDataset(slide, level=0, crop_size=512)
        embeddings, probas = self.nn_predict(inference_dataset)

        weights = np.repeat(np.expand_dims(probas[:, 1:].sum(axis=1), axis=-1), embeddings.shape[1], axis=-1)

        feats = np.concatenate([
                                embeddings.min(axis=0),
                                embeddings.max(axis=0),
                                embeddings.mean(axis=0),
                                np.median(embeddings, axis=0),
                                (embeddings ** 2).sum(axis=0)**(1/2),
                                (embeddings ** 3).sum(axis=0)**(1/3),
                                embeddings.std(axis=0),
                                (embeddings * weights).sum(axis=0),
                                probas.max(axis=0),
                                probas.mean(axis=0),
                                probas.std(axis=0),
                                probas.sum(axis=0),
                                np.histogram(np.argmax(probas, axis=1), bins=range(probas.shape[1]))[0],
                               ], axis=0)

        X_test = np.array([feats])
        pred_proba = self.cl.predict_proba(X_test)
        exp_pred = (pred_proba * np.array([list(range(6))] * len(pred_proba))).sum(axis=1)

        pred = np.where(np.histogram(exp_pred, bins=[0, 0.5, 1.5, 2.5, 3.5, 4.5, 1e10])[0])[0][0]

        exps = (probas * np.array([list(range(6))] * len(probas))).sum(axis=1)
        coords = []
        for idx in np.argsort(exps)[::-1]:
            if exps[idx] < 0.5:
                continue
            crop_coords = inference_dataset.get_coords(idx)
            coords.append({'x': int(crop_coords[1]), 'y': int(crop_coords[0]), 'expectation': float(exps[idx])})

        result['isup'] = int(pred)
        result['probas'] = [float(x) for x in pred_proba[0].tolist()]
        result['crops_info'] = coords
        
        return result    
    
    
    
    
    
    
    
    
    
