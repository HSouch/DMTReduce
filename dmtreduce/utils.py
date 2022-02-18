import os

import pathlib as path

from matplotlib import pyplot as plt
from astropy.visualization import ZScaleInterval

from scipy.signal import savgol_filter


def quickplot_spectra(data, contrast=1, make_figure=True, show=True, outfile=None, colorbar=False):
    """ Quickly plot an easily viewable spectra for debugging and testing purposes. """
    zscale = ZScaleInterval(contrast=contrast)
    scale = zscale.get_limits(data)
    if make_figure:
        fig = plt.figure(facecolor="white")
    plt.imshow(data, vmin=scale[0], vmax=scale[1], cmap="Greys_r")
    if colorbar:
        plt.colorbar()
    plt.tight_layout()
    if outfile is not None:
        plt.savefig(outfile, dpi=200)
    elif show:
        plt.show()


def spectra_lims(data, contrast):
    zscale = ZScaleInterval(contrast=contrast)
    scale = zscale.get_limits(data)

    return scale


def get_line_val(linregress_obj, x):
    return linregress_obj.slope * x + linregress_obj.intercept


def get_polyfit_line(polyfit_obj, x):
    # Assumes a 1 degree polynomial
    return polyfit_obj[1] + (x * polyfit_obj[0])


def check_and_make_dirs(directories):
    """ Make directories (one or multiple) """
    if type(directories) == str:
        directories = [directories]

    for directory in directories:
        if not os.path.isdir(directory):
            os.makedirs(directory, exist_ok=True)


def obtain_files(instring):
    """ Gather a list of files for processing. """
    p = path.Path(instring).expanduser()
    parts = p.parts[p.is_absolute():]
    generator = path.Path(p.root).glob(str(path.Path(*parts)))
    files = [str(n) for n in generator]
    return files


def smooth(spectrum, window_length, polyorder=2):
    return savgol_filter(spectrum, window_length=window_length, polyorder=polyorder)