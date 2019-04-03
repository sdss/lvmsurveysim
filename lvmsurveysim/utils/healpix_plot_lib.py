def healpix_shader(data,
                    masks, 
                    outfile="tmp.png",
                    nest=True,
                    save=True,
                    gui=False,
                    graticule=True,
                    healpixMask=False, 
                    vmin=False, vmax=False, 
                    norm=1.0, pad=0.1,
                    scale=False, 
                    cmaps=["Blues", "Greens", "Reds"],
                    background='w',
                    title=""):
    """Healpix shader.

    Parameters
    ----------
    data : numpy.ndarray
        data array of representing a full healpix
    masks : numpy.ma
        Masks of teh data array, which define which pixel gets which color
    nest : bol
        Is the numpy array nested, or ringed
    cmaps : matplotlib.cmaps
        Any maptplotlib cmap, called by name. The order must be the same order as the masks.
    Returns
    -------
    matplotlib.plt plot : 
        It returns a plot....
    """

    import healpy
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.colors as mcolors

    color_list = []
    for mask_i in range(len(masks)):
        if mask_i <= len(cmaps) -1:
            cmap = cmaps[mask_i]
        else:
            #Use the last cmap
            cmap = cmaps[-1]
        color_list.append(plt.get_cmap(cmap)(np.linspace(0.,1,128)))

    colors = np.vstack(color_list)
    cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
    cmap.set_under(background)

    # Copy the data to a new array to be cropped and scaled
    I = data.copy()

    # Crop data at max an min values if provided.
    # Otherwise set to range of data
    if vmax==False:
        vmax = np.max(I)
    if vmin==False:
        vmin = np.min(I)

    # Crop data
    I[I > vmax] = vmax
    I[I < vmin] = vmin

    # Rescale all data from 0+pad to (normalization-pad)
    I = (I - vmin) * ( (norm - pad) - pad) / (vmax - vmin) + pad

    normalized_I = np.full(len(I), -1.6375e+30)


    # add the offset to the data to push it into to each color range
    if scale is not False:
        for i in range(len(masks)):
            normalized_I[masks[i]] = I[masks[i]].copy()*scale[i] + min(i, len(cmaps)-1) * norm

    else:
        for i in range(len(masks)):
            normalized_I[masks[i]] = I[masks[i]].copy() + min(i, len(cmaps)-1) * norm

    # I could add all the masks here and plot the un-masked values in grey scale.

    # If there is a healpix mask apply it.
    normalized_I_masked = healpy.ma(normalized_I)

    if healpixMask != False:
        normalized_I_masked.mask = np.logical_not(healpixMask)

    healpy.mollview(normalized_I_masked, nest=nest, cbar=True, cmap=cmap, rot=(0,0,0), min=0, max=norm*len(masks), xsize=4000, title=title)

    if graticule == True:
        healpy.graticule()
    if save == True:
        plt.savefig(outfile)
    if gui==True:
        plt.show()
    plt.close()

if __name__ == "__main__":
    from astropy.io import fits
    import astropy.units as u
    from astropy_healpix import HEALPix
    from astropy.coordinates import Galactic
    import numpy as np

    hdu_list = fits.open("Halpha_fwhm06_1024.fits")
    nside = hdu_list[0].header['NSIDE']
    order = hdu_list[0].header['ORDERING']
    data = np.array(hdu_list[1].data.tolist())[:,0]



    log_I = np.log10(data)
    log_I_max = 2.0
    log_I_min = -1.0

    # Lets create a simple mask using the galactic longitude
    hp = HEALPix(nside=nside, order='nested', frame=Galactic())    
    npix=12*nside**2
    long_lat = hp.healpix_to_lonlat(range(npix))

    masks = []
    # I should add a grey scale in for non-masked values

    masks.append(np.full(npix, True))
    masks.append(long_lat[1] < -30*u.deg)
    masks.append(long_lat[1] > 30*u.deg)

    healpix_shader(log_I, masks, title=r"MW H$\alpha$", vmin=log_I_min, vmax=log_I_max, outfile="shaded_MW.png", gui=True)
