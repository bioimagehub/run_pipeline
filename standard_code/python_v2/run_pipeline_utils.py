from bioio import BioImage
import bioio_ome_tiff
from numpy import invert
from pathlib import Path
from tifffile import TiffWriter

from typing import Callable, Union

# Functions should be defined in the gloval rp utilities
def load_bioio_file(file: str):
    img = BioImage(file, reader=bioio_ome_tiff.Reader)
    return img

def process_file_by_YX_plane(input_path: str, output_path: str, func: Callable): 

    img = load_bioio_file(input_path)

    T, C, Z, Y, X = img.shape
    print(f"Image shape: T={T}, C={C}, Z={Z}, Y={Y}, X={X}")
    print(f"Writing output to: {output_path}")

    first_in = img.get_image_data("YX", T=0, C=0, Z=0)
    first_out = func(first_in)

    def processed_planes():
        # Yield planes in TCZ order to match OME axes TCZYX.
        for t in range(T):
            for c in range(C):
                for z in range(Z):
                    if t == 0 and c == 0 and z == 0:
                        out = first_out
                    else:
                        plane = img.get_image_data("YX", T=t, C=c, Z=z)
                        out = func(plane)

                    print(f"Processing plane: T={t}, C={c}, Z={z}, shape={out.shape}")
                    yield out

    with TiffWriter(output_path, ome=True, bigtiff=True) as tif:
        tif.write(
            data=processed_planes(),
            shape=(T, C, Z, Y, X),
            dtype=first_out.dtype,
            metadata={"axes": "TCZYX"},
        )

def process_files(file_list: list[str], func: Callable, chunk_type: str = "YX", output_dir: Union[str, None] = None, output_suffix: str = "_processed"):
    # TODO ADD parallel: bool = False

    if chunk_type == "YX":
        print("Hello from process-files!")
        for file in file_list:
            input_path = Path(file)
            if output_dir is not None:
                output_path = str(Path(output_dir) / f"{input_path.stem}{output_suffix}.ome.tif")
            else:
                output_path = str(input_path.with_name(f"{input_path.stem}{output_suffix}.ome.tif"))
            process_file_by_YX_plane(str(input_path), output_path, func)
    else:
        raise NotImplementedError(f"Chunk type {chunk_type} not implemented yet.")




