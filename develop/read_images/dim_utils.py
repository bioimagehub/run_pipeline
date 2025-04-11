from bioio import BioImage
import bioio_bioformats

class DimUtils:
    def __init__(self, bioio_dims):
        """Initialize the DimUtils class with a Dimensions object."""
        # Store the shape and order of dimensions
        self.shape = (
            bioio_dims.T,
            bioio_dims.C,
            bioio_dims.Z,
            bioio_dims.Y,
            bioio_dims.X,
        )
        self.order = bioio_dims.order  # e.g., "TCZYX"
        
        # Creating a mapping of dimension names to their respective indices
        self.dim_map = {dim: idx for idx, dim in enumerate(self.order)}

    def dim_idx(self, dim_name):
        """Return the index for the provided dimension name."""
        return self.dim_map.get(dim_name, None)
    
    def dim_val(self, dim_name):
        """Return the size of the provided dimension name."""
        idx = self.dim_idx(dim_name)
        if idx is not None:
            return self.shape[idx]
        raise ValueError(f"Dimension '{dim_name}' not found in the dimension mapping.")

# Example Usage
# Assuming bioio_dims is the Dimensions object obtained from your BioImage instance
testfile_path = r"C:\Users\oodegard\Desktop\collection_of_different_file_formats\input_nd2\1.nd2"
img = BioImage(testfile_path, reader=bioio_bioformats.Reader)  # TCZYX
bioio_dims = img.dims  # Example: <Dimensions [T: 181, C: 2, Z: 1, Y: 2720, X: 2720]>
dim_utils = DimUtils(bioio_dims)
print(f"Image dimensions: {dim_utils}")  # Should output (181, 2, 1, 2720, 2720)

# Accessing dimensions
print(f"Time index: {dim_utils.dim_idx('T')}")  # Should return 0
# Iterate through all dimensions (T, C, Z, Y, X) and print their indices
print(f"C index: {dim_utils.dim_idx('C')}")  # Should return 1
print(f"Z index: {dim_utils.dim_idx('Z')}")  # Should return 2
print(f"Y index: {dim_utils.dim_idx('Y')}")  # Should return 3
print(f"X index: {dim_utils.dim_idx('X')}")  # Should return 4

print(f"T size: {dim_utils.dim_val('T')}")  # Should return 181
print(f"C size: {dim_utils.dim_val('C')}")  # Should return 2
print(f"Z size: {dim_utils.dim_val('Z')}")  # Should return 1
print(f"Y size: {dim_utils.dim_val('Y')}")  # Should return 2720
print(f"X size: {dim_utils.dim_val('X')}")  # Should return 2720

print(f"Image shape: {dim_utils.shape}")  # Should output (181, 2, 1, 2720, 2720)






# bioio_dims = img.dims  # Assuming this returns a tuple of dimensions (T, C, Z, Y, X)


# print(bioio_dims)
# print(type(bioio_dims))
# print(bioio_dims.T, bioio_dims.C, bioio_dims.Z, bioio_dims.Y, bioio_dims.X)
# print(bioio_dims.order)  # Assuming this returns the order of dimensions as a string (e.g., "TCZYX")


# DimUtils(img.dims)  # Assuming this returns a tuple of dimensions (T, C, Z, Y, X)