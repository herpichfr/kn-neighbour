#!/bin/python3

"""Calculate the k-10-nearest neighbour for a given set of points
in a 3D space and save the results to a CSV file."""

import os
import pandas as pd
import numpy as np
import math
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
# import mpl_scatter_density
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
    parser.add_argument("--save_plot", action="store_true",
                        help="Flag to save the plot of the k-10-nearest neighbours.")
    parser.add_argument("--map_bins", type=int, default=100,
                        help="Number of bins for the map. Default is 100.")
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


def create_average_map(data):
    """Create a new dataset with average separation for bins of 0.1 degrees."""
    min_ra, max_ra = math.floor(data['ra'].min()), math.ceil(data['ra'].max())
    min_dec, max_dec = math.floor(
        data['dec'].min()), math.ceil(data['dec'].max())
    bin_size = 0.5
    ra_bins = np.arange(min_ra, max_ra, bin_size)
    dec_bins = np.arange(min_dec, max_dec, bin_size)
    avg_map = pd.DataFrame(columns=['ra', 'dec', 'avg_separation'])
    for ra in ra_bins:
        for dec in dec_bins:
            mask = ((data['ra'] >= ra) & (data['ra'] < ra + bin_size)) & \
                (data['dec'] >= dec) & (data['dec'] < dec + bin_size)
            if mask.sum() > 0:
                avg_separation = np.median(data.loc[mask, 'separation_k10'])
                avg_map = pd.concat([avg_map, pd.DataFrame({
                    'ra': [ra + 0.05],
                    'dec': [dec + 0.05],
                    'avg_separation': [avg_separation]
                })], ignore_index=True)
    return avg_map


def load_substructures_data(file_path):
    """Load substructures data from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    data = pd.read_csv(file_path)
    if 'RA' not in data.columns or 'Dec' not in data.columns:
        raise ValueError("Input file must contain 'ra' and 'dec' columns.")
    return data


def plot_kn_neighbour(args, data, substructures_data=None):
    nbins = args.map_bins
    x = data['ra']
    y = data['dec']

    k = gaussian_kde([x, y])
    xi, yi = np.mgrid[x.min():x.max():nbins*1j,
                      y.min():y.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
    fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='scatter_density')
    ax = fig.add_subplot(111)
    # ax.scatter_density(data['ra'], data['dec'],
    #                    c=data['separation_k10'], cmap='autumn', dpi=15, alpha=0.3)
    ax.pcolormesh(xi, yi, zi, cmap='inferno', shading='auto', alpha=0.9)

    cb = ax.scatter(data['ra'], data['dec'],
                    c=data['separation_k10'], cmap='viridis', s=5)

    if substructures_data is not None:
        ax.scatter(substructures_data['RA'], substructures_data['Dec'],
                   color='c', marker='*', label='Substructures', s=25)

    plt.colorbar(cb, label='k-10 Nearest Neighbour Separation')
    plt.xlabel('Right Ascension (deg)')
    plt.ylabel('Declination (deg)')
    # plt.title('k-10 Nearest Neighbours in 3D Space')
    plt.tight_layout()

    if args.save_plot:
        plot_file = args.input_file.replace(
            '.parquet', '_k10_neighbours_plot.png')
        plt.savefig(plot_file, dpi=300)
        print(f"Plot saved to {plot_file}")
    else:
        print("Plot not saved. Use --save_plot to save the plot.")

    plt.show()


if __name__ == "__main__":
    args = parse_args()
    data = calculate_kn_neighbour(
        args.input_file, args.output_file, save_output=args.save_output)

    # datamap = create_average_map(data)
    substructures_file = os.path.join(os.path.dirname(
        args.input_file), 'mkw4_substructures.csv')
    substructures_data = load_substructures_data(substructures_file)

    plot_kn_neighbour(args, data, substructures_data)
