import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.io import fits
from utils import *

from scipy.interpolate import interp1d


def identify_cosmic_rays(imin, nsigma=10, verbose=True, plot_results=True):
    """
        Use the detectors to remove cosmic rays for each file
        This will automatically overwrite the input image, so use the files made in
            your processing directory.

        imin: input filename
        ngisma: the number of sigmas above the median for a pixel to be considered a cosmic ray
        plot_results: print out results to pngs/cleaning/

    """
    if verbose:
        print("  Cleaning", imin)

    with fits.open(imin) as HDUList:
        images = []
        for i in range(1, len(HDUList)):
            data = HDUList[i].data
            images.append(data)
        images = np.asarray(images)
        median_image = np.median(images, axis=0)
        std = sigma_clipped_stats(median_image)[2]

        for i in range(1, len(HDUList)):
            data = HDUList[i].data
            #             print(data.shape)
            mask = (data > median_image + nsigma * std)

            data_masked = np.copy(data)
            data_masked[mask] = np.nan

            # Go through each column, interpolate, and fill nans
            ys = np.arange(0, len(data_masked[:, 0]), 1)
            for j in range(len(data_masked[0])):
                row = data_masked[:, j]
                indices = np.isnan(row)
                indices = np.logical_or(indices, np.roll(indices, -1))
                indices = np.logical_or(indices, np.roll(indices, 1))

                nan_indices = [n[0] for n in np.argwhere(indices)]

                row_ys, row_nonans = ys[~indices], row[~indices]
                interp = interp1d(row_ys, row_nonans, bounds_error=False, fill_value=np.nan)

                for index in nan_indices:
                    row[index] = float(interp(index))

                data_masked[:, j] = row
            HDUList[i].data = data_masked
            HDUList.writeto(imin, overwrite=True)

            if plot_results:
                check_and_make_dirs("pngs/cleaning/")
                combined = np.concatenate((data, data_masked), axis=0)
                quickplot_spectra(combined,
                                  outfile="pngs/cleaning/" + imin.split("/")[-1].split(".")[0] + "_" + str(
                                      i) + "_cleaned.png")


def generate_master(processing_dir, files, outname, show_spectra=False):
    """
        Generate a combined master image for a set of exposures.
        Should only be used for arcs and flats (for now, I think), but for now it seems to work really damn good.
            Also takes care of dead pixels!
    """

    files = obtain_files(processing_dir + files)

    outfile = fits.HDUList()
    # Generate a master flat image (kind of a bad way to do it but whatever it's not horrific - I think)
    for i in range(1, 5):
        det_flat = []
        for filename in files:
            with fits.open(filename) as HDUList:
                det_flat.append(np.copy(HDUList[i].data))
        this_flat = np.median(det_flat, axis=0)

        outfile.append(fits.ImageHDU(data=this_flat))
    outfile.writeto(processing_dir + outname, overwrite=True)

    if show_spectra:
        check_and_make_dirs(["pngs/" + outname + "/"])
        with fits.open(processing_dir + outname) as HDUList:
            for i in range(len(HDUList)):
                quickplot_spectra(HDUList[i].data, show=False)
                plt.tight_layout()
                plt.savefig("pngs/" + outname + "/det_" + str(i + 1) + ".png", dpi=200)


def bin_array(imin, imout, bin_width=2, method=np.sum, axis=0, plot_results=False):
    """
        Bin all frames along the wavelength direction.
        This assumes that the data was transposed earlier in rawprocess, so it will
        retranspose it.

        filename: input filename
        bin_width: the width of the bin to rebin into
        method: which method to use in rebinning, default is numpy.sum
        axis: the axis
    """

    with fits.open(imin) as HDUList:
        for i in range(1, len(HDUList)):
            # At this point the data will need to be transposed back and retransposed at the end
            data = np.transpose(HDUList[i].data)
            # Trim things off and reshape a new array to store the outputs
            newshape_x = int((data.shape[axis] - bin_width) / bin_width)
            newshape_x = int((data.shape[axis]) / bin_width)

            # Create new container array
            new_arr = np.zeros((newshape_x, data.shape[not bool(axis)]))

            # Go through our slices, sum things along the y axis, and add them to the new array
            for j in range(new_arr.shape[axis]):
                new_arr[j] = method(data[bin_width * j: bin_width * j + bin_width], axis=axis)

            # Retranspose at the end
            new_arr = np.transpose(new_arr)
            HDUList[i].data = new_arr

            if plot_results:
                check_and_make_dirs(["pngs/binning/"])
                quickplot_spectra(np.transpose(data),
                                  outfile="pngs/binning/" + imin.split("/")[-1].split(".")[0] + str(
                                      i) + "_unbinned.png")
                quickplot_spectra(new_arr,
                                  outfile="pngs/binning/" + imin.split("/")[-1].split(".")[0] + str(i) + "_binned.png")
        HDUList.writeto(imout, overwrite=True)



