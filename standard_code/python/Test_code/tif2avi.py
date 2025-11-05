import sys
import os
import numpy as np

# Use bioio for consistent image loading
try:
    from standard_code.python import bioimage_pipeline_utils as rp
    USE_BIOIO = True
except ImportError:
    print("Warning: bioimage_pipeline_utils not found, falling back to tifffile")
    import tifffile
    USE_BIOIO = False

# Try different video writing approaches
try:
    import imageio
    VIDEO_BACKEND = 'imageio'  # Prefer imageio (more stable)
except ImportError:
    try:
        import cv2
        VIDEO_BACKEND = 'opencv'
    except ImportError:
        print("Error: Neither imageio nor opencv-python is available")
        sys.exit(1)

if len(sys.argv) != 2:
    print("Usage: python tif2avi.py <input_file.tif>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = os.path.splitext(input_file)[0] + ".mp4"

# Load image using bioio (always returns 5D TCZYX)
if USE_BIOIO:
    img = rp.load_tczyx_image(input_file)
    # Get data as numpy array - this is always TCZYX (5D)
    frames_5d = img.data  # Shape: (T, C, Z, Y, X)
    print(f"Loaded image with shape: {frames_5d.shape} (TCZYX), dtype: {frames_5d.dtype}")
    
    # For video, we need to reduce to (T, Y, X) by selecting/projecting C and Z
    # Take max projection across Z, and if multiple channels, max across C too
    T, C, Z, Y, X = frames_5d.shape
    
    if Z > 1:
        print(f"Applying max Z-projection across {Z} slices...")
        frames_4d = frames_5d.max(axis=2)  # Max across Z -> (T, C, Y, X)
    else:
        frames_4d = frames_5d[:, :, 0, :, :]  # Remove Z dimension -> (T, C, Y, X)
    
    if C > 1:
        print(f"Applying max projection across {C} channels...")
        frames = frames_4d.max(axis=1)  # Max across C -> (T, Y, X)
    else:
        frames = frames_4d[:, 0, :, :]  # Remove C dimension -> (T, Y, X)
else:
    # Fallback to tifffile
    with tifffile.TiffFile(input_file) as tif:
        frames = tif.asarray()
    print(f"Loaded frames with shape: {frames.shape}, dtype: {frames.dtype}")
    
    # Handle multi-dimensional arrays (T, C, Y, X) or (T, Y, X)
    if frames.ndim == 4:
        # Multi-channel: Take maximum projection across channels
        print(f"Multi-channel image detected ({frames.shape[1]} channels). Using maximum projection...")
        frames = frames.max(axis=1)  # Max projection across channels
    elif frames.ndim == 2:
        # Single 2D image: add time dimension
        frames = frames[np.newaxis, ...]
    elif frames.ndim != 3:
        print(f"Error: Unexpected frame dimensions: {frames.shape}")
        sys.exit(1)

print(f"Final frame shape for video: {frames.shape} (T, Y, X)")

# Convert uint16 to uint8 for better video compatibility
frames_8bit = (frames / frames.max() * 255).astype('uint8')

# Enhance contrast using histogram equalization


print("Enhancing contrast...")

# Try to use scikit-image for advanced contrast enhancement, fallback to numpy
try:
    from skimage import exposure
    # Apply histogram equalization to enhance contrast
    for i in range(len(frames_8bit)):
        frames_8bit[i] = exposure.equalize_hist(frames_8bit[i]) * 255
        frames_8bit[i] = frames_8bit[i].astype('uint8')
    print("Applied histogram equalization")
except ImportError:
    # Fallback: Simple contrast stretching using numpy
    print("Using simple contrast enhancement...")
    for i in range(len(frames_8bit)):
        frame = frames_8bit[i].astype(np.float32)
        # Stretch contrast by normalizing to full 0-255 range
        min_val = frame.min()
        max_val = frame.max()
        if max_val > min_val:
            frame = (frame - min_val) / (max_val - min_val) * 255
        frames_8bit[i] = frame.astype('uint8')
    print("Applied contrast stretching")

# Pad frames to even dimensions for MP4 compatibility (H.264 requires even dimensions)
height, width = frames_8bit.shape[1], frames_8bit.shape[2]
if height % 2 != 0 or width % 2 != 0:
    new_height = height + (height % 2)
    new_width = width + (width % 2)
    print(f"Padding frames from {height}x{width} to {new_height}x{new_width} for MP4 compatibility")
    
    padded_frames = np.zeros((len(frames_8bit), new_height, new_width), dtype='uint8')
    padded_frames[:, :height, :width] = frames_8bit
    frames_8bit = padded_frames

print(f"Processing {len(frames_8bit)} frames at {frames_8bit.shape[1]}x{frames_8bit.shape[2]}...")

if VIDEO_BACKEND == 'opencv':
    # Use OpenCV for video writing with MP4
    height, width = frames_8bit.shape[1], frames_8bit.shape[2]
    
    # Convert grayscale to 3-channel BGR for better compatibility with OpenCV VideoWriter
    # This avoids stride/step issues with grayscale frames
    frames_bgr = np.stack([frames_8bit, frames_8bit, frames_8bit], axis=-1)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec for MP4
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (width, height), isColor=True)
    
    for i, frame in enumerate(frames_bgr):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Writing frame {i+1}/{len(frames_bgr)}")
        out.write(frame)
    
    out.release()
    
elif VIDEO_BACKEND == 'imageio':
    # Use imageio with ffmpeg for PowerPoint-compatible MP4
    try:
        # Imageio expects frames with shape (H, W) for grayscale or (H, W, C) for color
        # Our frames_8bit has shape (T, H, W), which is correct
        # Use imageio with explicit ffmpeg backend for MP4
        imageio.mimsave(output_file, frames_8bit, fps=20, quality=8, codec='libx264', pixelformat='yuv420p')
    except Exception as e:
        print(f"Error with mimsave: {e}")
        print("Trying alternative MP4 creation...")
        
        # Alternative approach using get_writer with specific MP4 settings
        try:
            writer = imageio.get_writer(output_file, fps=20, codec='libx264', pixelformat='yuv420p', quality=8)
            for i, frame in enumerate(frames_8bit):
                if (i + 1) % 10 == 0 or i == 0:
                    print(f"Writing frame {i+1}/{len(frames_8bit)}")
                writer.append_data(frame)
            writer.close()
        except Exception as e2:
            print(f"Error with MP4 creation: {e2}")
            print("MP4 creation failed. Please check ffmpeg installation.")
print(f"Converted {input_file} to {output_file}")
