#!/usr/bin/env python
# coding: utf-8

# Bramsh Qamar


import numpy as np
import dipy
import matplotlib.pyplot as plt
from dipy.io.image import load_nifti, save_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel
from dipy.viz import actor, window, ui
from dipy.viz import fvtk
import dipy.data as dpd
import dipy.direction.peaks as dpp
from dipy.segment.mask import median_otsu
from dipy.reconst.csdeconv import auto_response, ConstrainedSphericalDeconvModel
from dipy.direction import peaks_from_model
from dipy.tracking.local import LocalTracking
from dipy.tracking.local import ThresholdTissueClassifier
from dipy.tracking.utils import seeds_from_mask, random_seeds_from_mask
from nibabel.streamlines import save as save_trk
from nibabel.streamlines import load as load_trk
from nibabel.streamlines import Tractogram
from dipy.tracking.streamline import Streamlines
import dipy.reconst.dti as dti
from dipy.tracking import utils
from dipy.viz.colormap import line_colors
from dipy.segment.clustering import QuickBundles
from dipy.tracking.streamline import streamline_near_roi
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from time import time
from dipy.denoise.localpca import localpca
from dipy.denoise.pca_noise_estimate import pca_noise_estimate
from dipy.data import read_isbi2013_2shell
import nibabel as nib

import nibabel as nib
from nibabel.streamlines import Field
from nibabel.orientations import aff2axcodes


def save_trk_n(fname, streamlines, affine, vox_size=None, shape=None, header=None):
    """ function Helper for saving trk files.

    Parameters
    ----------
    fname : str
        output trk filename
    streamlines : list of 2D arrays
        Each 2D array represents a sequence of 3D points (points, 3).
    affine : array_like (4, 4)
        The mapping from voxel coordinates to streamline points.
    vox_size : array_like (3,)
        The sizes of the voxels in the reference image.
    shape : array, shape (dim,)
        The shape of the reference image.
    header : dict
        header from a trk file

    """
    if vox_size and shape:
        if not isinstance(header, dict):
            header = {}
        header[Field.VOXEL_TO_RASMM] = affine.copy()
        header[Field.VOXEL_SIZES] = vox_size
        header[Field.DIMENSIONS] = shape
        header[Field.VOXEL_ORDER] = "".join(aff2axcodes(affine))

    tractogram = nib.streamlines.Tractogram(streamlines)
    tractogram.affine_to_rasmm = affine
    trk_file = nib.streamlines.TrkFile(tractogram, header=header)
    nib.streamlines.save(trk_file, fname)


def load_trk(filename):
    """ function Helper for Loading trk files.

    Parameters
    ----------
    filename : str
        input trk filename

    Returns
    -------
    streamlines : list of 2D arrays
        Each 2D array represents a sequence of 3D points (points, 3).
    hdr : dict
        header from a trk file

    """
    trk_file = nib.streamlines.load(filename)
    return trk_file.streamlines, trk_file.header


def slider( image_actor,  line_actor):

    slicer_opacity = 0.6
    #image_actor.opacity(slicer_opacity)
    ren = window.Renderer()
    ren.add(image_actor)
    ren.add(line_actor)
   
    show_m = window.ShowManager(ren, size=(1200, 900))
    show_m.initialize()


    
    
    line_slider_z = ui.LineSlider2D(min_value=0,
                                    max_value=shape[2] - 1,
                                    initial_value=shape[2] / 2,
                                    text_template="{value:.0f}")
    opacity_slider = ui.LineSlider2D(min_value=0.0,
                                     max_value=1.0,
                                     initial_value=slicer_opacity)


    
    def change_slice_z(i_ren, obj, slider):
        z = int(np.round(slider.value))
        image_actor.display(None, None, z)        

    def change_opacity(i_ren, obj, slider):
        slicer_opacity = slider.value
        image_actor.opacity(slicer_opacity)
   
    
    line_slider_z.add_callback(line_slider_z.slider_disk,
                               "MouseMoveEvent",
                               change_slice_z)
  
    opacity_slider.add_callback(opacity_slider.slider_disk,
                                "MouseMoveEvent",
                                change_opacity)
    
    line_slider_label_z = ui.TextBox2D(text="Slice", width=50, height=20)
  
    opacity_slider_label = ui.TextBox2D(text="Opacity", width=50, height=20)
    
    panel = ui.Panel2D(center=(1030, 120),
                       size=(300, 200),
                       color=(1, 1, 1),
                       opacity=0.1,
                       align="right")
    
    panel.add_element(line_slider_label_z, 'relative', (0.1, 0.4))
    panel.add_element(line_slider_z, 'relative', (0.5, 0.4))
    panel.add_element(opacity_slider_label, 'relative', (0.1, 0.2))
    panel.add_element(opacity_slider, 'relative', (0.5, 0.2))
    
    #show_m.ren.add(panel)
    ren.add(panel)
    global size
    size = ren.GetSize()
    
    
    def win_callback(obj, event):
        global size
        if size != obj.GetSize():
            size_old = size
            size = obj.GetSize()
            size_change = [size[0] - size_old[0], 0]
            panel.re_align(size_change)
    
    show_m.initialize()
    #window.show(ren)
    show_m.add_window_callback(win_callback)
    show_m.render()
    show_m.start()
    

def show_streamlines(streamlines):
    from dipy.viz import window, actor
    ren = window.Renderer()
    ren.add(actor.line(streamlines))
    #window.record(ren, n_frames=10, out_path='masked_after.png', size=(600, 600))
    window.show(ren)


file= "file.nii.gz"
bvalue="file.bval"
bvector="file.bvec"

data, affine = load_nifti(file)
bvals, bvecs = read_bvals_bvecs(bvalue, bvector)
gtab = gradient_table(bvals, bvecs)


img = nib.load(file)
volume= data.shape[:3]
voxel= img.header.get_zooms()[:3]

t = time()
sigma = pca_noise_estimate(data, gtab, correct_bias=True, smooth=3)
print("Sigma estimation time", time() - t)



t = time()

denoised_arr = localpca(data, sigma=sigma, patch_radius=2)

print("Time taken for local PCA (slow)", -t + time())

sli = data.shape[2] // 2
gra = data.shape[3] // 2
orig = data[:, :, sli, gra]
den = denoised_arr[:, :, sli, gra]
rms_diff = np.sqrt((orig - den) ** 2)

fig, ax = plt.subplots(1, 3)
ax[0].imshow(orig, cmap='gray', origin='lower', interpolation='none')
ax[0].set_title('Original')
ax[0].set_axis_off()
ax[1].imshow(den, cmap='gray', origin='lower', interpolation='none')
ax[1].set_title('Denoised Output')
ax[1].set_axis_off()
ax[2].imshow(rms_diff, cmap='gray', origin='lower', interpolation='none')
ax[2].set_title('Residual')
ax[2].set_axis_off()
plt.show()


nib.save(nib.Nifti1Image(denoised_arr,
                         affine), 'denoised_localpca_'+file)

data = denoised_arr
sphere = dpd.get_sphere('repulsion724')

print('Starting median_otsu')


data_masked, mask = median_otsu(data, 2, 1, vol_idx=np.arange(1, 11),
                                dilate=1)
print('Finished masking')

print('Started DTI processing ...')
tenmodel = TensorModel(gtab)
tenfit = tenmodel.fit(data, mask=mask)
print('Finished DTI processing ...')

shape=mask.shape

odf = tenfit.odf(sphere)
ren = window.Renderer()
#odf_slicer 
odf_slicer_actor= actor.odf_slicer(odfs=odf,mask=mask, sphere=sphere, scale=.5)
ren.add(odf_slicer_actor)
slider(odf_slicer_actor, None )


data_small=data[:, :, 38:39]
dti_wls = dti.TensorModel(gtab)
fit_wls = dti_wls.fit(data_small)
    
fa1 = fit_wls.fa
evals1 = fit_wls.evals
evecs1 = fit_wls.evecs
cfa1 = dti.color_fa(fa1, evecs1)
ren = fvtk.ren()
fvtk.add(ren, fvtk.tensor(evals1, evecs1, cfa1, sphere))
#fvtk.record(ren, n_frames=1, out_path='tensor_ellipsoids.png',
#            size=(600, 600))
fvtk.show(ren)


print('Started CSD')
response, ratio = auto_response(gtab, data, roi_radius=10, fa_thr=0.7)
print('Finished response')    
csd_model = ConstrainedSphericalDeconvModel(gtab, response)
csd_peaks = peaks_from_model(model=csd_model,
                                 data=data,
                                 sphere=sphere,
                                 mask=mask,
                                 relative_peak_threshold=.5,
                                 min_separation_angle=25,
                                 parallel=True)
    
# using the peak_slicer
peak_actor=actor.peak_slicer(csd_peaks.peak_dirs,
                          csd_peaks.peak_values,
                          colors=None)
slider(peak_actor, None)


#generating streamlines

tissue_classifier = ThresholdTissueClassifier(tenfit.fa, 0.1)
    
seeds = random_seeds_from_mask(tenfit.fa > 0.3, seeds_count=5)
    
streamline_generator = LocalTracking(csd_peaks, tissue_classifier,
                                         seeds, affine=np.eye(4),
                                         step_size=0.5, return_all=True)
    
streamlines = Streamlines(streamline_generator)

show_streamlines(streamlines)

save_trk_n('streamlines.trk', streamlines, affine=affine, vox_size=voxel, shape=volume)

