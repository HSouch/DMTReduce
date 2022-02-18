from tqdm import tqdm

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import numpy as np
from scipy.interpolate import interp1d

from matplotlib import pyplot as plt
from utils import *


def model_sky(processing_dir, imin, distmap, slit_edges, trace_evals, inner_edges,
              dsub, interptype, sampling, im_index=1, padding=20, verbose=True, plot_results=False):
    # Load in image to process
    with fits.open(processing_dir + imin) as HDUList:
        hdr, img = HDUList[im_index].header, HDUList[im_index].data

    # quickplot_spectra(img)

    slit_edge = slit_edges[im_index]
    trace_eval = trace_evals[im_index][1]
    trace_eval = np.poly1d(trace_eval)

    inner_edge = inner_edges[im_index]
    inner_low, inner_high = np.poly1d(inner_edge[0]), np.poly1d(inner_edge[1])

    temp_bottom = np.poly1d([inner_edge[0][0], slit_edge[0]])
    temp_top = np.poly1d([inner_edge[0][0], slit_edge[1]])

    # ny is the spatial direction, nx the wavelength direction
    nx, ny = len(img[0]), slit_edge[1] - slit_edge[0] + 1
    print("nx", nx, "ny", ny)

    x, y = np.arange(0, nx, dtype=np.float), np.arange(0, ny, dtype=np.float)

    # This is the subsampled x array
    xs = np.arange(0, nx, dsub)

    line = np.arange(0, nx, dtype=np.float)

    ysamples = int(ny / sampling)
    subarray = np.zeros((ysamples, len(xs)))
    line = np.arange(0, nx, dtype=np.float)

    masked_img = np.zeros(np.copy(img).shape)

    # Generate a masked array
    # I am comfortable working with nan pixels so this shouldn't be an issue
    for i in range(img.shape[1]):
        strip = np.copy(img[:, i])
        strip[0:int(temp_bottom(i)) + 20] = np.nan
        strip[int(temp_top(i) - 20):] = np.nan
        strip[int(inner_low(i)) - padding:int(inner_high(i)) + padding] = np.nan

        masked_img[:, i] = strip

    # Now go through xs
    line = np.arange(0, nx, dtype=np.float)

    forbidden_regions = []
    bottom = temp_bottom(xs)
    forbidden_regions.append((0, padding))
    forbidden_regions.append((int(np.min(inner_low(xs) - bottom) - padding),
                              int(np.min(inner_high(xs) - bottom) + padding)))
    forbidden_regions.append((ny - padding, ny))
    print(forbidden_regions)

    for i in tqdm(range(ysamples), desc="Filling subarray with sky samples"):
        y_i = i * sampling

        # Check and make sure that we can use this line by checking forbidden regions
        try:
            for n in forbidden_regions:
                if n[0] < y_i < n[1]:
                    raise IndexError
        except IndexError:
            subarray[i, :] = np.nan
            continue

        for j in range(nx):
            bottom = temp_bottom(j)
            line[j] = img[np.int(y_i + bottom), j]

        # Find the new x-coord for this line
        try:
            x_thisline = x + distmap[int(y_i), :]
        except IndexError:
            # print("Issue with index:", y_i)
            continue
        # Now do a spline fit for the line
        spline = interp1d(x_thisline, line, kind=interptype, bounds_error=False, fill_value=0.)
        subarray[i, :] = spline(xs)

    # Now that we have a full subarray, we interpolate over and deproject
    ys = np.arange(0, subarray.shape[0], 1)

    skymodel_1D = np.nanmedian(subarray, axis=0)
    skymodel_1D_interp = interp1d(xs, skymodel_1D, bounds_error=False, fill_value=0)
    plt.plot(skymodel_1D)
    plt.show()

    # Now deproject back onto the original grid using the collapsed skymodel
    skymodel = np.copy(img) * 0
    for i in tqdm(range(ny), desc="Deprojecting onto original grid"):
        try:
            x_thisline = x + distmap[i, :]
        except IndexError:
            # print("Index issue")
            continue
        model_thisline = skymodel_1D_interp(x_thisline)
        for j in range(len(x)):
            skymodel[int(i + temp_bottom(j)), j] = model_thisline[j]

    if plot_results:
        fig, ax = plt.subplots(1, 3, facecolor="white")
        fig.set_figwidth(12)
        fig.set_figheight(4)

        ax[0].imshow(img)
        ax[1].imshow(skymodel)
        ax[2].imshow(img - skymodel)

        for n in range(len(ax)):
            ax[n].set_xticks([])
            ax[n].set_yticks([])

        plt.tight_layout()
        plt.savefig("pngs/skymodel.png", dpi=150)
    return skymodel
