## Mike_IMC_Denoise.ipynb

This is a Jupyter notebook that implements the IMC Denoise approach (https://github.com/PENGLU-WashU/IMC_Denoise/) to marry up with the Bodenmiller IMC pipeline.

It currently uses an older version of the IMC_Denoise package, and so needs updating. It also works specifically with the output of the original (and now less commonly used) Bodenmiller pipeline.

Running it will require  a computer with a good GPU / graphics card, and will also require it to be compatible with the packages that IMCDenoise uses (e.g. specific version of TensorFlow, Keras and Python, all of which are a bit out of date in the IMC Denoise repository on which I've based everything).


## IMC_Denoise
This is the folder with the version of the IMC_Denoise package that worked with the Jupyter notebook.