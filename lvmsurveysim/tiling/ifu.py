#!/usr/bin/env python
# encoding: utf-8
#
# ifu.py
#
# Created by José Sánchez-Gallego on 5 Sep 2017.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import shapely.geometry

import matplotlib.pyplot as plt
import matplotlib.collections
import matplotlib.patches
import seaborn as sns

from .. import config


sns.set_style('white')
current_palette = sns.color_palette()


__all__ = ('fibres_to_rows', 'SubIFU', 'IFU', 'MonolithicIFU', 'AbuttableTriangleIFU')


def fibres_to_rows(fibres):
    """Calculates the number of rows for a set of fibres.

    Given a total number of ``fibres``, returns the number rows. Given that the
    IFU is an hexagon, the central row has as many fibres as rows in the IFU.
    It returns ``None`` if the number of fibres can not be arranged in a
    full IFU.

    """

    tmp_fibres = 0
    n_central = 1

    while tmp_fibres <= fibres:

        n_central += 2  # Central row needs to be odd

        n_rows = int(1 + 2 * np.floor(n_central / 2.))
        tmp_fibres = n_central + 2 * np.sum(np.arange(n_central - 1, n_central / 2., -1))

        if tmp_fibres == fibres:
            return n_rows

    return None


class Fibre(object):

    def __init__(self, xx, yy, radius):

        self.x = xx
        self.y = yy
        self.radius = radius

        self.patch = matplotlib.patches.Circle((self.x, self.y), radius=self.radius,
                                               edgecolor='k', facecolor='None', lw=1)


class SubIFU(object):
    """Represents a sub-IFU (group of contiguous fibres) within a larger IFU.

    A sub-IFU is an hexagonal group of contiguous fibres, separated from other
    groups of fibres. LVM IFUs are formed by a series of sub-IFUs (a
    monolithic IFU can be considered as an IFU with a single sub-IFU). It is
    assumed that all sub-IFUs in an IFU have the same size.

    Parameters:
        n_subifu (int):
            An integer used to identify the sub-IFU.
        parent (:class:`IFU` object)
            The parent :class:`IFU` object.
        centre (tuple):
            A 2D tuple describing the centre of the sub-IFUs. It asumes the
            diameter of the sub-IFU hexagon is 1.
        n_fibres (int):
            Number of fibres in the sub-IFUs.
        fibre_size (float):
            The real size, in microns, of each fibre including buffer.

    """

    def __init__(self, id_subifu, parent, centre, n_fibres, fibre_size=None):

        assert isinstance(centre, (list, tuple, np.ndarray)), 'centre is not a list'
        assert isinstance(parent, IFU), 'parent must be an IFU object.'
        assert len(centre) == 2, 'centre must be a 2D tuple.'

        assert n_fibres is not None, 'incorrect inputs.'

        self.id_subifu = id_subifu
        self.parent = parent

        self.n_fibres = n_fibres
        self.n_rows = fibres_to_rows(self.n_fibres)

        self.centre = np.array(centre)

        self.fibre_size = fibre_size or config['fibre_size']

        self.polygon = self._create_polygon()
        self.fibres = self._create_fibres()

    def _create_polygon(self):
        """Creates a Shapely Polygon collection representing the sub-IFU."""

        RR = 0.5                   # Assumes unitary diameter
        rr = np.sqrt(3) / 2. * RR  # Inner radius
        cos60 = 0.5

        xx, yy = self.centre
        vertices = [(xx - RR, yy),
                    (xx - RR * cos60, yy + rr),
                    (xx + RR * cos60, yy + rr),
                    (xx + RR, yy),
                    (xx + RR * cos60, yy - rr),
                    (xx - RR * cos60, yy - rr)]
        polygon = shapely.geometry.Polygon(vertices)

        return polygon

    def _create_fibres(self):
        """Creates Fibre objects for each of the fibres in the sub-IFU."""

        n_centre = self.n_rows
        if self.n_rows is None:
            raise ValueError('incorrect number if fibres. Cannot create a sub-IFU.')

        fibres = []
        fibre_width = 1 / n_centre

        for row in range(int(-self.n_rows / 2), int(self.n_rows / 2) + 1):
            n_fibres_row = n_centre - np.abs(row)
            y_row = row * np.sqrt(3) / 2 / self.n_rows
            row_length = 1 - 2 * np.abs(y_row) * np.sqrt(3) / 3.

            fibres_x = np.arange(n_fibres_row) * fibre_width - row_length / 2. + fibre_width / 2.

            fibres += [Fibre(self.centre[0] + fibre_x,
                             self.centre[1] + y_row,
                             fibre_width / 2.) for fibre_x in fibres_x]

        return fibres

    def get_patch(self, filled=False):
        """Returns a matplotlib patch for the sub-IFU."""

        colour = current_palette[self.id_subifu - 1]
        colour_alpha = tuple(list(colour) + [0.3])

        return matplotlib.patches.Polygon(self.polygon.exterior.coords,
                                          edgecolor=colour,
                                          facecolor=colour_alpha if filled else 'None',
                                          lw=2)

    def get_patch_collection(self, ax):
        """Returns a collection of fibre patches."""

        n_centre = self.n_rows
        fibre_width = 1 / n_centre * 0.8  # 0.8 to make plotting look nicer.
        fibres = self.fibres
        fibres_patch = matplotlib.collections.EllipseCollection(
            [fibre_width] * len(fibres), [fibre_width] * len(fibres), [0] * len(fibres),
            offsets=[[fibre.x, fibre.y] for fibre in fibres],
            transOffset=ax.transData,
            lw=1, edgecolor='k', facecolor='None',
            units='x')

        return fibres_patch


class IFU(object):
    """A generic class representing a LVM hexagonal IFU.

    This class is intended to be subclassed into real examples of IFU
    designs that LVM will use. An ``IFU`` is defined by a series of sub-IFU
    centres (or a single one for a monolithic IFU) and a fibre size, so
    that the real size of the IFU on the sky can be calculated.

    Parameters:
        n_fibres (int):
            Number of fibres in each of the sub-IFUs.
        n_ifus (int):
            Number of sub-IFUs that compose the IFU.
        centres (list):
            A list of 2D tuples describing the centres of each of the
            sub-IFUs. It asumes the diameter of the sub-IFU hexagon is 1.
        padding (int):
            Number of fibres the IFU should overlap when tiling. This will
            be used to slightly modify the distance between sub-IFUs.
        fibre_size (float):
            The real size, in microns, of each fibre including buffer.
        gaps (list):
            If the IFU is composed of multiple sub-IFUs and gaps exist
            between them, a list with the same format of ``centres`` must
            be provided, with the list of the hexagonal gap centres.

    """

    def __init__(self, n_fibres=None, n_ifus=None, centres=None, padding=0,
                 fibre_size=None, gaps=None):

        assert isinstance(centres, (list, tuple, np.ndarray)), 'centres is not a list'
        assert len(centres) > 0, 'centres must be a non-zero length list.'
        for ll in centres:
            assert isinstance(ll, (list, tuple, np.ndarray)), 'each centre must be a 2D tuple.'
            assert len(ll) == 2, 'each centre must be a 2D tuple.'

        assert None not in [n_fibres, n_ifus], 'incorrect inputs.'

        assert n_fibres % n_ifus == 0, 'number of fibres is not a multiple of number of sub-IFUs.'

        self.n_fibres = n_fibres
        self.n_ifus = n_ifus
        self.centres = np.array(centres)

        self.fibre_size = fibre_size or config['fibre_size']

        self.padding = padding

        self.subifus = self._create_subifus()
        self.polygon = shapely.geometry.MultiPolygon([subifu.polygon for subifu in self.subifus])
        self.gaps = self._create_gaps(gaps)

    def _create_subifus(self):
        """Creates each one of the individual sub-IFUs in this IFU."""

        subifus = []
        n_subifu = 1
        for centre in self.centres:
            subifus.append(SubIFU(n_subifu, self, centre, self.n_fibres / self.n_ifus,
                                  fibre_size=self.fibre_size))
            n_subifu += 1

        return subifus

    def _create_gaps(self, gaps):
        """Creates polygons for the gaps."""

        if len(self.subifus) == 1:
            return []

        areas = [subifu.polygon.area for subifu in self.subifus]
        if (self.polygon.envelope.area - np.sum(areas)) == 0:
            return []

        assert gaps is not None, 'gaps are not defined for this IFU.'
        assert len(gaps) > 0, 'gapss must be a non-zero length list.'
        for ll in gaps:
            assert isinstance(ll, (list, tuple, np.ndarray)), 'each gap must be a 2D tuple.'
            assert len(ll) == 2, 'each gap must be a 2D tuple.'

        gap_polygons = []

        RR = 0.5                   # Assumes unitary diameter
        rr = np.sqrt(3) / 2. * RR  # Inner radius
        cos60 = 0.5

        for gap in gaps:
            xx, yy = gap
            vertices = [(xx - RR, yy),
                        (xx - RR * cos60, yy + rr),
                        (xx + RR * cos60, yy + rr),
                        (xx + RR, yy),
                        (xx + RR * cos60, yy - rr),
                        (xx - RR * cos60, yy - rr)]
            gap_polygons.append(shapely.geometry.Polygon(vertices))

        return gap_polygons

    def plot(self, show_fibres=False, show_gaps=False, filled=True):
        """Plots the IFU."""

        fig, ax = plt.subplots()

        if show_gaps:
            for gap in self.gaps:
                patch = matplotlib.patches.Polygon(gap.exterior.coords,
                                                   edgecolor='0.75',
                                                   facecolor='None', lw=0.5,
                                                   ls='dashed')

                ax.add_patch(patch)

        for subifu in self.subifus:
            ax.add_patch(subifu.get_patch(filled=filled))

            if show_fibres:
                ax.add_collection(subifu.get_patch_collection(ax))

        ax.autoscale_view()

        # Pads the xy limits

        if not show_gaps:
            bounds = self.polygon.bounds
        else:
            bounds = shapely.geometry.MultiPolygon(
                [subifu.polygon for subifu in self.subifus] + self.gaps).bounds

        xx_pad = 0.1 * (bounds[2] - bounds[0])
        yy_pad = 0.1 * (bounds[3] - bounds[1])

        ax.set_xlim(bounds[0] - xx_pad, bounds[2] + xx_pad)
        ax.set_ylim(bounds[1] - yy_pad, bounds[3] + yy_pad)

        return fig


class MonolithicIFU(IFU):

    def __init__(self, *args, **kwargs):

        super(MonolithicIFU, self).__init__(**config['ifus']['monolithic'])


class AbuttableTriangleIFU(IFU):

    def __init__(self, *args, **kwargs):

        super(AbuttableTriangleIFU, self).__init__(**config['ifus']['abuttable_triangle'])
