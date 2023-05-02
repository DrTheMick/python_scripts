#!/mnt/medic/mmartynowycz/miniconda3/bin/python

import argparse
import os
import multiprocessing as mp
import numpy as np
from PIL import Image
from tqdm import tqdm

def process_file(args):
    input_file, output_dir, bin_number = args

    # Construct output filename
    #BACKWARDS output_file = os.path.join(output_dir, os.path.basename(input_file).replace('.img', f'_binned_{bin_number}.tif'))
    #NOT RIGHT output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(input_file))[0] + f'_binned_{bin_number}.tif')
    output_file = os.path.join(output_dir, f'binned_{bin_number}_' + os.path.splitext(os.path.basename(input_file))[0] + '.tif')
    # Read input file
    with open(input_file, 'rb') as f:
        # Skip header (512 bytes)
        f.read(512)
        # Read image data as 16-bit unsigned integers
        data = np.fromfile(f, dtype=np.uint16)

    # Determine dimensions of the input image
    dim = int(np.sqrt(data.size))

    # Reshape data into 2D array
    data = data.reshape((dim, dim))

    # Calculate new dimensions after binning
    new_dim = dim // bin_number

    # Bin image by the specified bin number by summing instead of averaging
    binned_data_rows = np.add.reduceat(data, np.arange(0, dim, bin_number), axis=0)
    binned_data = np.add.reduceat(binned_data_rows, np.arange(0, dim, bin_number), axis=1)

    # Save binned image as TIFF file
    with Image.fromarray(binned_data.astype(np.uint16)) as img:
        img.save(output_file)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert SMV images to binned TIFFs')
    parser.add_argument('input_dir', type=str, help='Path to input directory')
    parser.add_argument('output_dir', type=str, help='Path to output directory')
    parser.add_argument('bin_number', type=int, help='Bin number')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Get list of input files
    input_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith('.img')]

    # Print information about input and output files
    print(f"Found {len(input_files)} input files in {args.input_dir}")
    print(f"Output files will be saved in {args.output_dir}")
    print(f"Images will be binned by {args.bin_number}")

    # Create a pool of worker processes
    num_workers = mp.cpu_count()
    pool = mp.Pool(processes=num_workers)

    # Process input files in parallel
    results = list(tqdm(pool.imap_unordered(process_file, [(input_file, args.output_dir, args.bin_number) for input_file in input_files]), total=len(input_files), desc="Processing files"))

    # Close the pool
    pool.close()
    pool.join()

    # Count the number of output files
    output_files = [f for f in os.listdir(args.output_dir) if f.endswith('.tif')]
    print(f"Generated {len(output_files)} output files in {args.output_dir}")

if __name__ == '__main__':
    main()
