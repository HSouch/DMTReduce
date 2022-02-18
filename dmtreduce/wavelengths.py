import numpy as np
from astropy.io import fits

def wavelength_soln(arc, slit_edges, im_index=1 ,):
    """ Get wavelength solution from arc file"""

    with fits.open(arc) as HDUList:
        img, hdr = HDUList[im_index].data, HDUList[im_index].header

    for n in hdr:
        print(n)

    slit_edge = slit_edges[im_index]
    print(slit_edge)

    # Trim off excess
    img = img[slit_edge[0]: slit_edge[1], :]

    quickplot_spectra(img)

    # Collapse to 1D spectra

    spectra_1D = np.mean(img, axis=0)

    plt.plot(spectra_1D)
    plt.axhline(np.median(spectra_1D), color="black")
    plt.ylim(0, 2000)
    plt.show()

    return