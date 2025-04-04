import subprocess

# Define your script or code that calls BioImage
def run_bioimage():
    from bioio import BioImage
    import bioio_bioformats

    file_path = r"Z:\Schink\Oyvind\colaboration_user_data\20250124 - Viola\Input\230705_93_mNG-DFCP1_LT_LC3_CMvsLPDS\CM\CM_chol_2h\2023-07-07\230705_mNG-DFCP1_LT_LC3_CM_chol_2h.ims"
    img = BioImage(file_path, reader=bioio_bioformats.Reader)

# Run the code in a subprocess and suppress output
subprocess.run(
    ['python', '-c', 'from bioio import BioImage; import bioio_bioformats; file_path = r"Z:\\Schink\\Oyvind\\colaboration_user_data\\20250124 - Viola\\Input\\230705_93_mNG-DFCP1_LT_LC3_CMvsLPDS\\CM\\CM_chol_2h\\2023-07-07\\230705_mNG-DFCP1_LT_LC3_CM_chol_2h.ims"; img = BioImage(file_path, reader=bioio_bioformats.Reader)'],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
)
