#!/mnt/medic/mmartynowycz/miniconda3/bin/python

import argparse
import os
import multiprocessing as mp
import numpy as np
from PIL import Image
from tqdm import tqdm

def process_file(args):
    input_file, output_dir, bin_number = args

    output_file = os.path.join(output_dir, f'binned_{bin_number}_' + os.path.splitext(os.path.basename(input_file))[0] + '.tif')

    with open(input_file, 'rb') as f:
        f.read(512)
        data = np.fromfile(f, dtype=np.uint16)

    dim = int(np.sqrt(data.size))
    data = data.reshape((dim, dim))

    new_dim = dim // bin_number

    binned_data_rows = np.add.reduceat(data, np.arange(0, dim, bin_number), axis=0)
    binned_data = np.add.reduceat(binned_data_rows, np.arange(0, dim, bin_number), axis=1)

    with Image.fromarray(binned_data.astype(np.uint16)) as img:
        img.save(output_file)

def bin_z_slices(output_dir, bin_number_z):
    tif_files = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.tif')])
    num_files = len(tif_files)
    num_binned_files = num_files // bin_number_z

    print(f"Outputting {num_binned_files} binned images in z and will leave off {num_files % bin_number_z} leftover images.")

    for i in tqdm(range(num_binned_files), desc="Binning in z"):
        summed_data = None
        for j in range(bin_number_z):
            with Image.open(tif_files[i * bin_number_z + j]) as img:
                data = np.array(img)

                if summed_data is None:
                    summed_data = data
                else:
                    summed_data += data

                os.remove(tif_files[i * bin_number_z + j])

        output_file = os.path.join(output_dir, f'binned_{bin_number_z}x{bin_number_z}_{str(i).zfill(3)}.tif')
        with Image.fromarray(summed_data.astype(np.uint16)) as img:
            img.save(output_file)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert SMV images to binned TIFFs')
    parser.add_argument('input_dir', type=str, help='Path to input directory')
    parser.add_argument('output_dir', type=str, help='Path to output directory')
    parser.add_argument('bin_number', type=int, help='Bin number for x-y')
    parser.add_argument('bin_number_z', type=int, help='Bin number for z')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Get list of input files
    input_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith('.img')]

    # Exclude leftover images from processing
    num_images = len(input_files)
    num_leftover_images = num_images % args.bin_number_z
    if num_leftover_images != 0:
        input_files = input_files[:-num_leftover_images]
        num_images -= num_leftover_images
        print(f"Excluding {num_leftover_images} leftover images")

    # Print information about input and output files
    print(f"Found {num_images} input files in {args.input_dir}")
    print(f"Output files will be saved in {args.output_dir}")
    print(f"Images will be binned by {args.bin_number} in x-y and {args.bin_number_z} in z")

    # Create a pool of worker processes
    num_workers = mp.cpu_count()
    pool = mp.Pool(processes=num_workers)

    # Process input files in parallel
    results = list(tqdm(pool.imap_unordered(process_file, [(input_file, args.output_dir, args.bin_number) for input_file in input_files]), total=len(input_files), desc="Binning in x-y"))

    pool.close()
    pool.join()

    bin_z_slices(args.output_dir, args.bin_number_z)

    output_files = [f for f in os.listdir(args.output_dir) if f.endswith('.tif')]
    print(f"Generated {len(output_files)} output files in {args.output_dir}")

if __name__ == '__main__':
    main()
