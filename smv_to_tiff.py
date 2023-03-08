#Right now this just bins 4k x 4k by 4 and throws out the header. 
import argparse
import numpy as np
from PIL import Image

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert SMV image to binned TIFF')
    parser.add_argument('input', type=str, help='Path to input SMV file')
    parser.add_argument('output', type=str, help='Path to output TIFF file')
    args = parser.parse_args()

    # Read input file
    with open(args.input, 'rb') as f:
        # Skip header (512 bytes)
        f.read(512)
        # Read image data as 16-bit unsigned integers
        data = np.fromfile(f, dtype=np.uint16)
    
    # Reshape data into 2D array
    data = data.reshape((4096, 4096))

    # Bin image by 4 by summing instead of averaging
    #This is, in fact, hot garbage and slow as hell. I can do better using reshape in numpy but have not tested it yet. TO DO.
    binned_data = np.zeros((1024, 1024), dtype=np.uint16)
    for i in range(1024):
        for j in range(1024):
            binned_data[i,j] = np.sum(data[4*i:4*(i+1), 4*j:4*(j+1)])
    
    # Save binned image as TIFF file
    with Image.fromarray(binned_data) as img:
        img.save(args.output)
    
if __name__ == '__main__':
    main()
