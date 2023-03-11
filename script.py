import argparse
import os
import multiprocessing as mp
import numpy as np
from PIL import Image

###THIS ONLY BINS BY 4 for now...Will fix later. The multiprocessing works though. ####

######USAGE######
### python script.py input_dir output_dir   #####

def process_file(input_file, output_dir):
    # Construct output filename
    output_file = os.path.join(output_dir, os.path.basename(input_file).replace('.img', '_binned.tif'))

    # Read input file
    with open(input_file, 'rb') as f:
        # Skip header (512 bytes)
        f.read(512)
        # Read image data as 16-bit unsigned integers
        data = np.fromfile(f, dtype=np.uint16)

    # Reshape data into 2D array
    data = data.reshape((4096, 4096))

    # Bin image by 4 by summing instead of averaging
    binned_data = np.zeros((1024, 1024), dtype=np.uint16)
    for i in range(1024):
        for j in range(1024):
            binned_data[i,j] = np.sum(data[4*i:4*(i+1), 4*j:4*(j+1)])

    # Save binned image as TIFF file
    with Image.fromarray(binned_data) as img:
        img.save(output_file)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert SMV images to binned TIFFs')
    parser.add_argument('input_dir', type=str, help='Path to input directory')
    parser.add_argument('output_dir', type=str, help='Path to output directory')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Get list of input files
    input_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith('.img')]

    # Create a pool of worker processes
    num_workers = mp.cpu_count()
    pool = mp.Pool(processes=num_workers)

    # Process input files in parallel
    for input_file in input_files:
        pool.apply_async(process_file, args=(input_file, args.output_dir))

    # Wait for all worker processes to finish
    pool.close()
    pool.join()

if __name__ == '__main__':
    main()
