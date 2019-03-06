#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2019-03-04
# @Filename: healpix.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2019-03-06 14:59:25

import healpy
import numpy
import shapely.geometry
import shapely.prepared
import shapely.vectorized


__all__ = ['nested_regrade', 'get_minimum_nside_pixarea', 'tile_geometry']


def nested_regrade(pixels, nside_in, nside_out):
    """Returns the parent/children pixels from a given HealPix nested pixel.

    The HealPix nested mode follows a quadrilateral tree pixel scheme (see
    Figure 1 in Górski et al. 2005) When the resolution (nside) increases,
    each pixel is divided in four pixels. The numbering of such pixels follows
    a simple binary representation achieved by appending binary digits to the
    new pixels. So, for example, pixel 22 (``b10110``) becomes four new pixels
    with binary numbers ``b1011000, b1011001, b1011010, and b1011011``,
    corresponding to decimal ``91, 92, 93, and 94``.

    This function returns the child pixels from a given pixel when going from
    resolution ``nside_in`` to ``nside_out``, if ``nside_out > nside_in``, or
    the parent pixel if ``nside_out < nside_in``.

    Note that this function works only for pixels using the nested indexing and
    should not be used with the ring indexing.

    Parameters
    ----------
    pixels : `int` or `~numpy.ndarray`
        The pixels for which we want to get the parents/children. Can be a
        single integer or an array of indices.
    nside_in : int
        The ``nside`` of the input pixels. Must be one of :math:`2^k` where
        :math:`k\in[0,1,2,\dots]`.
    nside_out : int
        The destination ``nside``.

    Returns
    -------
    output : `int` or `~numpy.ndarray`
        If ``nside_out < nside_in`` and ``pixels`` is a single value, this
        will be an integer with the parent of the input pixel. If ``pixels``
        is an array the output will be an array of the same size in which each
        element is the parent of the corresponding pixel in the input array.
        If ``nside_out > nside_in`` and ``pixels`` is an integer, the output
        will be an array with the child pixels. The size of the array will be
        :math:`2^{2l}` where ``l=k_out-k_in`` (:math:`n_{side}=2^k`).
        If the input in an array of pixels, the output will be a 2D array in
        which each row contains the child pixels of the corresponding pixel
        in the input array.

    """

    assert nside_in != nside_out, 'nside_in cannot be equal to nside_out.'

    k_in = numpy.log2(nside_in)
    k_out = numpy.log2(nside_out)

    assert k_in.is_integer() and k_out.is_integer(), \
        'nside_in or nside_out are not power of 2.'

    pixels = numpy.atleast_1d(pixels).astype(int)
    assert pixels.ndim == 1, 'dimension of input pixels is invalid.'

    # Npix = 12 * nside**2
    assert numpy.all(pixels <= 12 * nside_in**2), \
        'some pixel indices are greater than the maximum allowed for nside_in.'

    if k_in > k_out:

        degraded = numpy.right_shift(pixels, 2 * int(k_in - k_out))

        return degraded[0] if len(degraded) == 1 else degraded

    else:

        prograded = numpy.zeros((len(pixels), int(2**(2 * (k_out - k_in)))), dtype=int)
        prograded += pixels[numpy.newaxis].T
        for ii in range(int(k_out - k_in))[::-1]:

            prograded = numpy.left_shift(prograded, 2)

            repeats = numpy.repeat([0, 1, 2, 3], (4 ** ii) or 1)
            tiles = numpy.tile(repeats, prograded.shape[1] // len(repeats))

            prograded += tiles

        return prograded


def get_minimum_nside_pixarea(pixarea):
    """Returns the minimum nside needed to achieve a certain pixel resolution.

    Parameters
    ----------
    pixarea : float
        The pixel area in square degrees.

    Returns
    -------
    nside : int
        The minimum nside that guarantees a pixel size of at least the input
        pixel area.

    """

    nside = numpy.sqrt(3. / numpy.pi / pixarea) * 60.
    kk = int(numpy.ceil(numpy.log2(nside)))

    return int(2**kk)


def tile_geometry(polygon, nside, ring=False, return_coords=False, inclusive=True):
    """Returns a list of pixels that tile a Shapely polygon.

    While `healpy.query_polygon` provides an efficient method to tile a
    polygon, that function assumes that the lines joining the vertices of the
    polygon are the segments of the great circles that intersect the vertices.
    This is relatively irrelevant for polygons that cover a small area on the
    sky but for large polygons the result is significantly different from what
    one would expect.

    This function tiles a `Polygon <shapely:Polygon>` assuming an Euclidian
    distance between the vertices. The polygon must have longitude
    limits defined between -360 and 720 degrees.

    If ``inclusive=True`` the function will return all pixels that overlap with
    the region. The area of the pixel is approximated as a circle with the
    same area as a pixel. This approximation is good for low latitudes but
    becomes worse for higher latitudes, where the shape of the pixel cannot
    be approximated by a circle. If ``inclusive=False`` only pixels whose
    centre is inside the region will be returned.

    Parameters
    ----------
    polygon : `Polygon <shapely:Polygon>` or `~numpy.ndarray`
        The polygon to tile. If an array, it must be a collection of ``Nx2``
        points defining the position of the vertices. The polygon must be
        convex.
    nside : int
        The nside of the pixels used to tile the polygon.
    ring : bool
        By default the function returns the values in the nested pixel
        ordering. If ``ring=True`` the equivalent ring pixels will be returned.
    return_coords : bool
        If `True`, returns an array with the longitude and latitude of the
        pixels in degrees.
    inclusive : bool
        Whether to return pixels that only partially overlap with the region.

    Returns
    -------
    tiling : `~numpy.ndarray`
        An array with the list of pixels that tile the geometry or (if
        ``return_coords=True``) the longitude and latitude of the pixels.

    """

    # nside = 2^k

    assert numpy.log2(nside).is_integer(), 'nside is not a power of 2.'
    k_end = int(numpy.log2(nside))

    if not isinstance(polygon, shapely.geometry.Polygon):
        polygon = shapely.geometry.Polygon(polygon.tolist())

    # Create a prepared polygon. This allows only contained and intersect
    # operations but that's all we need and it's more efficient.
    prep_polygon = shapely.prepared.PreparedGeometry(polygon)

    pixels = []
    intersect = []

    for kk in range(0, k_end + 1):

        nside_k = 2**kk

        # Approximates the pixel as a circle of radius r.
        rr = numpy.sqrt(healpy.nside2pixarea(nside_k, degrees=True) / numpy.pi)

        # If k=0 (first HealPix level) we test all the 12 pixels. Otherwise we
        # take the pixels that overlapped in the previous level and test each
        # one of their children.
        if kk == 0 and len(intersect) == 0:
            pix_to_test = list(range(0, 12))
        else:
            pix_to_test = nested_regrade(intersect, 2**(kk - 1), nside_k).flatten().tolist()
            intersect = []

        for pix in pix_to_test:

            lon, lat = healpy.pix2ang(nside_k, pix, nest=True, lonlat=True)

            # We offset the pixel to check so that if the polygon wraps around
            # [0, 360] we still overlap with it.
            for offset in [0, -360, 360]:

                # Create a Point object with a radius dd centred at the
                # position of the pixel +/- the offset.
                point = shapely.geometry.Point(lon + offset, lat).buffer(rr)

                # If a pixel is completely contained by the polygon, adds
                # all the nested pixels at the nside resolution and we are done.
                if prep_polygon.contains(point):
                    if nside_k < nside:
                        pixels += nested_regrade(pix, nside_k, nside).flatten().tolist()
                    else:
                        pixels.append(pix)
                    break

                # If we are at the final nside level and the pixel intersects
                # with the polygon, we include it.
                if nside_k == nside and inclusive and prep_polygon.intersects(point):
                    pixels.append(pix)
                    break

                # If we are not yet at the final nside level we want to be greedy.
                # We create a pixel with twice the radius and check if it intersects.
                # If it does we add it to the list of pixels to test in the next
                # nside level. We do this to compensate for the fact that pixels
                # at high latitudes are significantly non-circular and if we
                # just use point we may be missing some pixels.
                if nside_k < nside:
                    point_2r = shapely.geometry.Point(lon + offset, lat).buffer(2. * rr)
                    if prep_polygon.intersects(point_2r):
                        intersect.append(pix)
                        break

        intersect = numpy.unique(intersect).tolist()

    if len(pixels) == 0:
        raise ValueError('the list of tiling pixels is empty.')

    pixels = numpy.unique(pixels)

    if return_coords:
        lon, lat = healpy.pix2ang(nside, pixels, nest=True, lonlat=True)
        return numpy.array([lon, lat]).T

    if ring:
        return healpy.nest2ring(nside, pixels)

    return pixels
