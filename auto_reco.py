#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 14:52:49 2018

@author: bramsh
"""

from dipy.workflows.segment import RecoBundlesFlow

subjects = ['P/3102/3102_first', 'P/3102/3102_second',
 'P/3116/3116_first',
 'P/3105/3105_first', 'P/3105/3105_second',
 'P/3111/3111_first', 'P/3111/3111_second',]

obj = RecoBundlesFlow(mix_names=True, force=True)

for sub in subjects:
    obj.run(streamline_files="/home/bramsh/Desktop/better/"+sub+"/moved.trk",
            model_bundle_files="/home/bramsh/Desktop/better/tuned_bundles/*.trk")