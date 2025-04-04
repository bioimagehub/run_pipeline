import subprocess
import os
# Set multiple logging properties for SLF4J SimpleLogger
os.environ["JAVA_TOOL_OPTIONS"] = (
    "-Dorg.slf4j.simpleLogger.defaultLogLevel=warn "
    "-Dorg.slf4j.simpleLogger.log.loci.formats=error "
    "-Dorg.slf4j.simpleLogger.log.loci.common=error "
    "-Dorg.slf4j.simpleLogger.log.org.scijava.nativelib=error"
)
# Define your script or code that calls BioImage
def run_bioimage():
    from bioio import BioImage
    import bioio_bioformats

    file_path = r"Z:\Schink\Oyvind\colaboration_user_data\20250124 - Viola\Input\230705_93_mNG-DFCP1_LT_LC3_CMvsLPDS\CM\CM_chol_2h\2023-07-07\230705_mNG-DFCP1_LT_LC3_CM_chol_2h.ims"
    

    img = BioImage(file_path, reader=bioio_bioformats.Reader)



run_bioimage()



# # Run the code in a subprocess and suppress output
# subprocess.run(
#     ['python', '-c', 'from bioio import BioImage; import bioio_bioformats; file_path = r"Z:\\Schink\\Oyvind\\colaboration_user_data\\20250124 - Viola\\Input\\230705_93_mNG-DFCP1_LT_LC3_CMvsLPDS\\CM\\CM_chol_2h\\2023-07-07\\230705_mNG-DFCP1_LT_LC3_CM_chol_2h.ims"; img = BioImage(file_path, reader=bioio_bioformats.Reader)'],
#     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
# )
