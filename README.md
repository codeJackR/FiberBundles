# FiberBundles
White Matter Fiber Bundle Extraction

generate_tractogram.py : Generates tractography from MRI images. it transforms MRI images into streamline space 
                          using loacl tracking algorithm. it performs other steps as well like extracting brain mask from raw 
                          brain MRI containg skull and eyes.
                          
streamline_registration_and_recobundle.py : registers the generated tractograms to common space and extracts bundles 
                                            using RecoBundle for preparing training data set.
                   
tck_to_trk.py : converts streamline files from .tck to .trk file format
                   
auto_reco.py : applies RecoBundle using dipy command line interface

random_forest.py : applies random_forest on training data

probabilistic.ipynb : applies probabilistic approaches on training data (MCMC/Gibbs Sampling)
