import numpy as np
from tqdm import tqdm
from astropy.io import fits
from matplotlib import pyplot as plt
from utils import *
from scipy.interpolate import interp1d
from astropy.stats import sigma_clipped_stats


def calc_ydist(processing_dir, imin, edges, trace_eval, dsub, im_index=1, fit_degree=2,
               x_width=251, verbose=True, plot_results=False, diagnostic_plots=False, plot_limits=[-3, 3]):
    #     if verbose:
    #         print("Fitting line tilts in arc spectrum")
    slit_edge = edges[im_index]

    # Get wavelength image
    with fits.open(processing_dir + imin) as HDUList:
        hdr = HDUList[im_index].header
        img = HDUList[im_index].data

    trace_eval = trace_eval[im_index][0]  # Just take one of the lines and assume the rest is fine
    trace_eval = np.poly1d(trace_eval)

    # quickplot_spectra(img)

    nx = len(img[0])  # This is the wavelength axis
    ny = slit_edge[1] - slit_edge[0] + 1  # This is the physical axis

    x, y = np.arange(0, nx, dtype=np.float), np.arange(0, ny, dtype=np.float)  # define x and y axes

    line_loc = np.arange(0, nx, x_width)

    nlines = len(line_loc)
    # Create container arrays
    dist_xyval = np.zeros((nlines, ny))
    im_dist = np.zeros((ny, nx))

    dx = 0
    for i in tqdm(range(nlines), desc="Fitting line tilts"):
        #         if verbose:
        #             print("  Working around: " + str(line_loc[i]) + " -- "
        #                   + str(line_loc[i] + x_width))

        # Get the image slice
        img_slice = img[:, line_loc[i]:line_loc[i] + x_width]

        # Get the regions in x
        x1 = line_loc[i]
        x2 = line_loc[i] + len(img_slice[0])

        # Create container arrays
        xl = np.arange(x1, x2, dtype=np.float)
        xls = np.arange(x1, x2, dsub)
        xlc = np.arange(x1, x2, 0.05)

        top = int(trace_eval(line_loc[i])) + 10
        bottom = int(trace_eval(line_loc[i]) + ny) - 10
        centre_y = int(trace_eval(line_loc[i]) + ny / 2)

        # spectrum_centre = np.median(img_slice[centre_y - 2:centre_y + 3], axis=0)
        spectrum_centre = img_slice[centre_y]
        spectrum_centre -= np.nanmedian(spectrum_centre)

        ys = np.arange(top, top + ny, dtype=int)
        shifts = np.arange(0, ny, dtype=np.float)

        # Then for each y value get spectrum and correllate
        for j in range(len(ys[::])):
            row = ys[j]
            this_line = img_slice[row]
            this_line -= np.nanmedian(this_line)

            cc = np.correlate(spectrum_centre, this_line, "same")
            # do spline fit to get subpixel maximum
            f = interp1d(xl, cc, kind='cubic', bounds_error=False,
                         fill_value=0.)
            ccs = f(xlc)

            max_pos = xlc[np.argmax(ccs)] - (x2 - x1) / 2 - x1
            shifts[j] = max_pos

        # Reject extremely bad values using sigma clipping
        stats = sigma_clipped_stats(shifts)
        good_indices = np.abs(shifts) < 3 * stats[2]
        # print(np.sum(good_indices))

        ys_good, shifts_good = ys[good_indices], shifts[good_indices]

        # Generate a polynomial fit
        shifts_fit = np.poly1d(np.polyfit(ys_good, shifts_good, deg=1))
        shifts_interpolated = shifts_fit(ys)

        dist_xyval[i] = shifts_interpolated

        if plot_results and diagnostic_plots:
            _ydist_diagnostic_plot(img_slice, spectrum_centre, x1, x2, xl,
                                  ys, ys_good, shifts_good, shifts_interpolated,
                                  imin, top, bottom, centre_y)

    # populate ydist image
    ydist_image = np.ndarray((len(ys), img.shape[1]))

    xs = np.arange(0, img.shape[1])
    x_centres = line_loc + (x_width / 2)
    dist_xyval = np.transpose(dist_xyval)

    # Now go through each y value, construct a interp1d object, and populate the final array
    # We will use linear extrapolation for simplicity
    for i in tqdm(range(len(ys)), desc="Interpolating to get final distortion map."):
        y_slice = dist_xyval[i]

        fit = np.polyfit(x_centres, y_slice, deg=fit_degree)
        fit = np.poly1d(fit)

        ydist_image[i] = fit(xs)
    #         #         fit = np.polyfit()
    # #         ydist_image[i] =

    #         ydist_image[i] = interp(xs)

    if plot_results:
        check_and_make_dirs(["pngs/ydists/"])
        plt.figure(figsize=(10, 4), facecolor="white")
        plt.imshow(ydist_image, vmin=plot_limits[0], vmax=plot_limits[1])
        plt.xlabel("Wavelength")
        plt.ylabel("Spatial")
        plt.colorbar()
        plt.savefig("pngs/ydists/" + imin.split("/")[-1].split(".")[0] + "_dist_xyval.png", dpi=200)

    return ydist_image


def rectify_array(a, distortion_map, dsub=0.5):
    """ Applies a distortion mapping onto an image """

    ny = a.shape[0]
    nx = int(a.shape[1] / dsub)

    rectified = np.zeros(a.shape)
    subsampled = np.zeros((ny, nx))

    print(a.shape, subsampled.shape)

    xs = np.arange(0, a.shape[1])

    for i in range(0, ny):
        row_i = a[i]
        dist_i = distortion_map[i]

        these_xs = xs + dist_i
        interp = interp1d(these_xs, row_i, bounds_error=False, fill_value="extrapolate")

        rectified[i] = interp(xs)

    plt.figure(figsize=(10, 6))
    plt.imshow(rectified + a / 20, vmin=0, vmax=0.5)
    plt.savefig("post_rectification.png", dpi=100)

    return rectified


def _ydist_diagnostic_plot(img_slice, spectrum_centre, x1, x2, xl,
                          ys, ys_good, shifts_good, shifts_interpolated,
                          imin, top, bottom, centre_y):
    """
        Create a diagnostic plot for the ydist calculation.
        NOTE: This is a purely internal plotting routing and should NEVER
            be used outside of the calc_ydist() method.
    """

    spectrum_top = img_slice[top]
    spectrum_top -= np.nanmedian(spectrum_top)

    spectrum_bottom = img_slice[bottom]
    spectrum_bottom -= np.nanmedian(spectrum_bottom)

    lims = spectra_lims(img_slice, contrast=1)

    check_and_make_dirs(["pngs/ydists/"])

    fig, ax = plt.subplots(1, 3, facecolor="white")

    fig.set_figheight(8)
    fig.set_figwidth(13)
    ax[0].imshow(img_slice, cmap="Greys_r", vmin=lims[0], vmax=lims[1])
    ax[0].set_title("Slice [" + str(x1) + "," + str(x2) + "]")

    ax[1].plot(xl, spectrum_centre, color="black", lw=3, label="centre (template)")
    ax[1].plot(xl, spectrum_top, color="red", lw=1, alpha=0.7, label="top")
    ax[1].plot(xl, spectrum_bottom, color="blue", lw=1, alpha=0.7, label="bottom")
    ax[1].legend()

    ax[1].set_title("1D Line Spectrum")

    ax[2].plot(ys_good, shifts_good)
    ax[2].plot(ys, shifts_interpolated)
    ax[2].axhline(0, color="black")
    ax[2].set_ylim(-3, 3)

    ax[2].set_title("Shifts")
    plt.tight_layout()
    plt.savefig("pngs/ydists/" + imin.split("/")[-1].split(".")[0] + "_" + str(x1) + "_" + str(x2) + ".png", dpi=200)

