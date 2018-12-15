#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 14:43:42 2018

@author: bramsh
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from dipy.tracking.streamline import Streamlines
from dipy.io.streamline import save_trk, load_trk

import numpy as np
from dipy.tracking.streamline import set_number_of_points

import os

import pickle


os.chdir("data")

subs = os.listdir()
subs.sort()

bundle, bn_hr = load_trk(subs[0]+"/AF_L.trk")
brain, br_hr = load_trk(subs[0]+"/brain.trk")

subs.pop(0)

dis_brain = set_number_of_points(brain, 100)
brain_data = dis_brain.data

af_label = np.zeros(len(brain_data))

af_label[0:(len(bundle)*100)] = 1

for sb in subs:
    
    n = len(brain_data)
    bundle, bn_hr = load_trk(sb+"/AF_L.trk")
    brain, br_hr = load_trk(sb+"/brain.trk")
    
    tem_brain = set_number_of_points(brain, 100)
    tem_data = dis_brain.data
    z = np.zeros(len(tem_data))
    
    af_label = np.concatenate((af_label, z) )
    
    brain_data = np.concatenate((brain_data, tem_data))
    
    
    af_label[n:n+(len(bundle)*100)] = 1
    
    
print(brain_data.shape)
print(len(af_label))

clf = RandomForestClassifier(n_estimators=50, 
                             random_state=0)

clf.fit(brain_data, af_label)

print(clf.feature_importances_)

#try prediction on one for now
brainy, br_hr = load_trk("data/100408/brain.trk")
dis_brainy = set_number_of_points(brain, 100)
brainy_data = dis_brainy.data

pred = (clf.predict(brainy_data))

print(np.sum(pred))

np.save('test1.npy', pred)

filename = 'model1.sav'
pickle.dump(clf, open(filename, 'wb'))