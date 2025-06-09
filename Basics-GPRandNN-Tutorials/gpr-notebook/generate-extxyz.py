import argparse
from ase.io import read, write, iread

def main(inputfile):
    FILE = iread(inputfile, format = 'aims-output')
    for s in FILE:
        write(outputfile, s, format = 'extxyz', append='True')
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser( description = """"This script is to convert the aims output into extxyz""")

    parser.add_argument('-i', type = str)
    parser.add_argument('-o', type = str)

    args = parser.parse_args()
    inputfile = args.i
    outputfile = args.o
    print('\n\n************* This script is to convert the aims output into extxyz *************')
    main(inputfile)
