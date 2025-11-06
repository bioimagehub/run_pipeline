from cellpose import denoise
dn = denoise.DenoiseModel(model_type="denoise_cyto3", gpu=True)
