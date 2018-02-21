Wavenet
=======

# GUIDE

##### First of all, download whole files.
##### You should download dataset from another way.
##### I used VCTK dataset (Just search at google).
##### This model is made by keras.

### taehyoung_util.py

##### This is util file. This includes preprocessing


### Save_wave_as_array.py

##### This save wave files(dataset) as mu_quantized array( from 0 to 255 ) 
##### This can reduce data loading time and training time.


### wavenet_TH_models.py

##### Wavenet models


### wavenet_TH_main.py

##### Training code. 
#### You must change 'path' to your own local path. (Where your dataset be in)
#### Also, You must adjust number of train set, validation set, and batch size considering CPU & Memory you have. 


### wavenet_TH_generate.py

##### Generation code
