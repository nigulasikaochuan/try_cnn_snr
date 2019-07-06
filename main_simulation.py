import torch
from fastai.vision import cnn_learner, ItemList
# from scipy import misc
import joblib
import numpy as np
import torchvision.transforms as transforms
from fastai.vision import cnn_learner, ItemList
from torchvision.models import resnet18

from try_cnn_snr.mynetwork import CNN


def get_df():
    all_information = joblib.load('fea').real.astype(np.float64)

    fea = all_information[:, :, :-1].real
    fea.shape = -1, 4, 32, 32

    target = []
    for i in all_information:
        target.append(i[0, -1])
    target = np.array(target).real
    target.shape = len(fea), 1

    return all_information


data = ItemList.from_array(get_df()).split_by_rand_pct(0.3)
data = data.label_from_array().databunch()
# print('xixi')
# .label_from_func(lambda data:data[0,-1]).databunch()

learner = cnn_learner(data, CNN, pretrained=False)
learner.model.double()
learner.lr_find()
learner.fit(200)
print('xixi')
