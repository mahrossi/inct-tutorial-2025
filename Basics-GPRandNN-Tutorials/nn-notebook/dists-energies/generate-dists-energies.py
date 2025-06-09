import argparse
from ase.io import read, write, iread
import numpy as np

def main(inputfile):
    geometries = iread(inputfile, format = 'extxyz')
    with open(outputfile, 'w') as f_out:
        f_out.write('# Distance(Angstrom) Energy(eV)\n')
    
        # Loop through each geometry
        for geom in geometries:
            # Assuming the structure contains exactly two atoms
            distance = np.linalg.norm(geom.positions[0] - geom.positions[1])

            # Extract energy from the comment line
            energy = float(geom.info['energy'])
    
            # Write to the output file
            f_out.write(f'{distance} {energy}\n')

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser( description = """"This script reads extxyz files and does some stuff""")

    parser.add_argument('-i', type = str)
    parser.add_argument('-o', type = str)

    args = parser.parse_args()
    inputfile = args.i
    outputfile = args.o
    print('\n\n************* This script calculates some stuff  *************')
    main(inputfile)
