#!/bin/python3

"""Calculate the k-10-nearest neighbour for a given set of points
in a 3D space and save the results to a CSV file."""

import os
import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate the k-10-nearest neighbour for a given set of points.")
    parser.add_argument("input_file", type=str,
                        help="Path to the input CSV file containing the points.")
    parser.add_argument("--output_file", type=str,
                        help="Path to the output CSV file to save the k-10-nearest neighbours.")
    parser.add_argument("--save_output", action="store_true",
                        help="Flag to save the output to a file.")
    return parser.parse_args()


def calculate_kn_neighbour(input_file, output_file, save_output=False):

    if output_file is None:
        output_file = input_file.replace('.parquet', '_k10_neighbours.csv')
    if os.path.exists(output_file):
        data = pd.read_csv(output_file)
        print(
            f"Output file {output_file} already exists. Loading existing data.")
        return data
    else:
        data = pd.read_parquet(input_file)
        ra = data.ra
        dec = data.dec

        mask_specz = (data.z > 0.007) & (data.z < 0.037)
        mask_photz = (data.zml > 0.007) & (data.zml < 0.037)

        redshift = data.zml.copy()
        redshift[mask_photz] = data.z[mask_photz]
        redshift[mask_specz] = data.z[mask_specz]

        mask = ~np.isnan(redshift)
        mask &= (redshift > 0.007) & (redshift < 0.037)

        ra = ra[mask]
        dec = dec[mask]
        redshift = redshift[mask]
        newdf = data[mask].reset_index(drop=True)

        coords = SkyCoord(ra=ra, dec=dec, distance=redshift,
                          frame='icrs', unit=(u.deg, u.deg, u.Mpc))
        separation_3d = np.zeros(len(coords))

        for i, coord in enumerate(coords):
            sep2d = coord.separation(coords)
            sep3d = coord.separation_3d(coords)
            separation_10th = sorted(sep2d.value * sep3d.value)[11]
            separation_3d[i] = separation_10th

        newdf['separation_k10'] = separation_3d

        if save_output:
            newdf.to_csv(output_file, index=False)
            print(f"Output saved to {output_file}")
        else:
            print("Output not saved. Use --save_output to save the results.")

        return newdf


def plot_kn_neighbour(data):
    plt.figure(figsize=(10, 6))
    plt.scatter(data['ra'], data['dec'],
                c=data['separation_k10'], cmap='viridis', s=1)
    plt.colorbar(label='Separation to k-10 Nearest Neighbour')
    plt.xlabel('Right Ascension (deg)')
    plt.ylabel('Declination (deg)')
    plt.title('k-10 Nearest Neighbours in 3D Space')
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    data = calculate_kn_neighbour(
        args.input_file, args.output_file, save_output=args.save_output)

    plot_kn_neighbour(data)
