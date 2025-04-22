from bioio import BioImage
import bioio_bioformats
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import subprocess

def save_video_with_ffmpeg(frames, output_file_path, fps):
    # Create a folder for the temporary JPG frames
    jpg_folder = os.path.splitext(output_file_path)[0] + "_frames"
    os.makedirs(jpg_folder, exist_ok=True)  # Create the directory if it doesn't exist
    print("Preparing jpg frames for video")

    for i, frame in enumerate(frames):
        frame_filename = f"temp_frame_{i:04d}.png"
        # skip if the file already exists
        if os.path.exists(os.path.join(jpg_folder, frame_filename)):
            print(f"Skipping existing frame: {frame_filename}")
            continue
        # Save each frame as a temporary JPG file
        plt.imsave(os.path.join(jpg_folder, frame_filename), frame, cmap='gray')

    # Use ffmpeg to convert the images to a video file
    ffmpeg_command = [
        'C:/Program_Files/ffmpeg-7.1.1-essentials_build/bin/ffmpeg.exe',
        '-framerate', str(fps),
        '-i', os.path.join(jpg_folder, 'temp_frame_%04d.png'),
        '-c:v', 'libx264',  # Use H.264 codec
        '-pix_fmt', 'yuv420p',  # Ensure compatibility
        '-movflags', 'faststart',  # Allows video to start playing before completely downloaded
        '-metadata', 'title=Converted Video',  # Optional metadata
        '-y',  # Overwrite output files
        output_file_path
    ]

    print(f"Running ffmpeg command: {' '.join(ffmpeg_command)}")
    
    subprocess.call(ffmpeg_command)

    # # Cleanup temporary images if necessary
    # for i in range(len(frames)):
    #     os.remove(os.path.join(jpg_folder, f"temp_frame_{i:04d}.png"))

def process_image(input_file_path, output_file_path=None, fps=24):
    # Set output file path
    if output_file_path is None:
        output_file_path = os.path.splitext(input_file_path)[0] + ".mp4"
    else:
        if not output_file_path.endswith(".mp4"):
            raise ValueError("Output file must have .mp4 extension")

    # Load image data
    print(f"Loading image data from {input_file_path}")

    img = BioImage(input_file_path, reader=bioio_bioformats.Reader)
    
    # Get the image dimensions
    dims = img.dims  # Dimensions object
    print(f"Image dimensions: {dims}")
    
    # Convert to numpy array
    img_array = img.data  

    # Mass project or remove Z dimension if necessary
    print("Maximum projection of Z dimension")
    if dims.Z > 1:
        img_array = np.max(img_array, axis=2)  # Maximum projection of Z
    else:
        img_array = img_array.squeeze(axis=2)


    print(f"Image shape after projection: {img_array.shape}")

    # loop over channels and

    for i in range(dims.C):
        ch_img = img_array[:, i, :, :]
        
        # Normalize to uint8
        ch_img = (ch_img - np.min(ch_img)) / (np.max(ch_img) - np.min(ch_img) + 1e-8) * 255
        ch_img = ch_img.astype(np.uint8)

        ch_name_split = os.path.splitext(output_file_path)
        ch_name = ch_name_split[0] + f"_ch{i+1}" + ch_name_split[1]

        # Create frames for the video
        frames = []
        for frame in range(dims.T):
            frame_img = ch_img[frame, :, :]
            frames.append(frame_img)
        
        # Save video using the new function
        save_video_with_ffmpeg(frames, ch_name, fps)
        
        print(f"Video saved to {ch_name}")

    print("Done saving channels")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", type=str, required=True, help="Path to the file to be processed")
    parser.add_argument("-o", "--output_file", type=str, help="Path for the output MP4 file")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second for output video")

    args = parser.parse_args()

    process_image(args.input_file, args.output_file, args.fps)

if __name__ == "__main__":
    main()
