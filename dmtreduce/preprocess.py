import os
import numpy as np
from astropy.io import fits

from utils import *


def gen_front_matter(objfile, params=None, verbose=True):
    images = np.genfromtxt(objfile, dtype='str', comments="#")
    image_files = images[:, 0]
    image_type = images[:, 1]
    nimages = len(image_files)

    global science_files, arc_files, flat_files

    science_files = image_files[image_type == 'sci']
    arc_files = image_files[image_type == 'arc']
    flat_files = image_files[image_type == 'flat']

    if verbose:
        print(" Loaded", len(science_files), "science files.")
        print(" Loaded", len(arc_files), "arc files.")
        print(" Loaded", len(flat_files), "flat files.")

    global x_overscan, x_good, gapsize, mill2det, instr, dx_pixels, xref

    x_overscan = [1092, 1150]
    x_good = [52, 1076]
    x_good = [52, 3000]
    gapsize = 100
    mill2det = (10.26, 3990)
    instr = "LRISBLUE"
    dx_pixels = 5.74
    xref = 3679

    tempdir = "test_dir/" if params is None else params["TEMP_DIR"]
    if not os.path.isdir(tempdir):
        os.makedirs(tempdir, exist_ok=True)


def fitbadcol(processing_dir, imin, imout, instr):
    """
        Fit the bad columns using simple interpolation

        imin: input image filename
        imout: output image filename
        instr: instrument name (either LRIS or LRISBLUE)
    """

    badcols = np.genfromtxt(instr + "_badcol/bad.dat", dtype='str', comments="#")
    bc_y1 = badcols[:, 0]
    bc_y2 = badcols[:, 1]
    bc_y1 = bc_y1.astype(np.int) - 1
    bc_y2 = bc_y2.astype(np.int) + 1

    outfile = fits.HDUList()
    with fits.open(imin) as HDUList:
        for i in range(0, len(HDUList)):
            try:
                data = HDUList[i].data
                for col in badcols:
                    cent = int(col[0])
                    data_slice = data[cent]

                    interp = np.mean((data[cent - 1], data[cent + 1]), axis=0)
                    data[cent] = interp
                HDUList[i].data = data
                outfile.append(HDUList[i])
            except:
                outfile.append(HDUList[i])
        outfile.writeto(processing_dir + imout, overwrite=True)


def rawprocess(processing_directory, image, outname,
               x_overscan, x_good, gapsize, instr, chipflip=True,
               show_spectra=False, verbose=True):
    """
        Process raw LRIS files

        processing_directory: The directory to place completed files
        image: Filename of FITS LRIS input image
        tmpdir: temporary directory for temp files
        outname: output filename
        x_overscan: overscan region
        x_good: good region for LRIS spectrum
        gapsize: I dunno (yet)
        instr: instrument (either LRIS or LRISBLUE)
    """

    if verbose:
        print("Processing Raw:  ", image, " -> ", outname)

    if instr == "LRISBLUE":
        gain = 1.61  # gain from web page - average of 4 chips
        order = [1, 2, 3, 4]  # order of chips, from bottom to top

    outfile = fits.HDUList()
    with fits.open(image) as HDUList:
        for i in range(1, len(HDUList)):
            data = HDUList[i].data
            bias_val = np.median(data[:, x_overscan[0]:x_overscan[1]])

            data = data[x_good[0]:x_good[1], :] - bias_val  # Trim and subtract bias value
            data = np.transpose(data)
            HDUList[i].data = data

        # Handle a chip flip (if necessary?)
        if instr == "LRISBLUE" and chipflip:
            HDUList[1].data = np.flip(HDUList[1].data, axis=0)
            HDUList[3].data = np.flip(HDUList[3].data, axis=0)

        if show_spectra:
            for i in range(1, len(HDUList)):
                quickplot_spectra(HDUList[i].data)

        for i in range(len(HDUList)):
            outfile.append(HDUList[i])
        outfile.writeto(processing_directory + outname, overwrite=True)

