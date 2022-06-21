# -*- coding: utf-8 -*-
"""Code based on the pyntcloud library.

The MIT License

Copyright (c) 2017-2019 The pyntcloud Developers

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from typing import Iterable, List

import numpy as np
from scipy.spatial import cKDTree


def cartesian(arrays: List[Iterable], out: np.ndarray = None) -> np.ndarray:
    """Generate a Cartesian product of input arrays.

    Args:
        arrays (List[Iterable]): list of array-like 1-D arrays to form the
            Cartesian product of.
        out (np.ndarray): Array to  place the cartesian product in.
            Defaults to None.

    Returns:
        np.ndarray: 2-D array of shape (M, len(arrays)) containing cartesian
        products formed of input arrays.

    Examples:
        >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
        array([[1, 4, 6],
            [1, 4, 7],
            [1, 5, 6],
            [1, 5, 7],
            [2, 4, 6],
            [2, 4, 7],
            [2, 5, 6],
            [2, 5, 7],
            [3, 4, 6],
            [3, 4, 7],
            [3, 5, 6],
            [3, 5, 7]])
    """
    arrays = [np.asarray(x) for x in arrays]
    shape = (len(x) for x in arrays)
    dtype = arrays[0].dtype

    ix = np.indices(shape)
    ix = ix.reshape(len(arrays), -1).T

    if out is None:
        out = np.empty_like(ix, dtype=dtype)

    for n, _ in enumerate(arrays):
        out[:, n] = arrays[n][ix[:, n]]

    return out


class VoxelGrid:
    def __init__(
        self,
        points,
        properties=None,
        n_x=1,
        n_y=1,
        n_z=1,
        size_x=None,
        size_y=None,
        size_z=None,
        regular_bounding_box=True,
    ):
        """Initialize a VoxelGrid.

        Args:
            points (np.ndarray): shape (N,  3)
            properties (numpy.array): Shape (N, 3). Defaults to None.
            n_x (int): The number of segments in which each axis will be divided.
                Ignored if corresponding size_x, size_y or size_z is not None. Defaults to 1.
            n_y (int): The number of segments in which each axis will be divided.
                Ignored if corresponding size_x, size_y or size_z is not None. Defaults to 1.
            n_z (int):  The number of segments in which each axis will be divided.
                Ignored if corresponding size_x, size_y or size_z is not None. Defaults to 1.
            size_x (float): The desired voxel size along each axis.
                If not None, the corresponding n_x, n_y or n_z will be ignored.
                Defaults to None.
            size_y (float): The desired voxel size along each axis.
                If not None, the corresponding n_x, n_y or n_z will be ignored.
                Defaults to None.
            size_z (float): The desired voxel size along each axis.
                If not None, the corresponding n_x, n_y or n_z will be ignored.
                Defaults to None.
            regular_bounding_box (bool): If True, the bounding box
                of the point cloud will be adjusted in order to have all
                the dimensions of equal length. Defaults to True.
        """
        self._points = points
        self.properties = properties
        self.x_y_z = np.asarray([n_x, n_y, n_z])
        self.sizes = np.asarray([size_x, size_y, size_z])
        self.regular_bounding_box = regular_bounding_box

        self.id = None
        self.xyzmin, self.xyzmax = None, None
        self.segments = None
        self.shape = None
        self.n_voxels = None
        self.voxel_x, self.voxel_y, self.voxel_z = None, None, None
        self.voxel_n = None
        self.voxel_centers = None
        self.voxel_colors = None

    def compute(self):
        """Compute the voxel grid."""
        xyzmin = self._points.min(0)
        xyzmax = self._points.max(0)
        xyz_range = self._points.ptp(0)

        if self.regular_bounding_box:
            #: adjust to obtain a minimum bounding box with all sides of equal length
            margin = max(xyz_range) - xyz_range
            xyzmin = xyzmin - margin / 2
            xyzmax = xyzmax + margin / 2

        for n, size in enumerate(self.sizes):
            if size is None:
                continue
            margin = (((self._points.ptp(0)[n] // size) + 1) * size) - self._points.ptp(0)[n]
            xyzmin[n] -= margin / 2
            xyzmax[n] += margin / 2
            self.x_y_z[n] = ((xyzmax[n] - xyzmin[n]) / size).astype(int)

        self.xyzmin = xyzmin
        self.xyzmax = xyzmax

        segments = []
        shape = []
        for i in range(3):
            # note the +1 in num
            s, step = np.linspace(xyzmin[i], xyzmax[i], num=(self.x_y_z[i] + 1), retstep=True)
            segments.append(s)
            shape.append(step)

        self.segments = segments
        self.shape = shape

        self.n_voxels = np.prod(self.x_y_z)

        self.id = "V({},{},{})".format(self.x_y_z, self.sizes, self.regular_bounding_box)

        # find where each point lies in corresponding segmented axis
        # -1 so index are 0-based; clip for edge cases
        self.voxel_x = np.clip(
            np.searchsorted(self.segments[0], self._points[:, 0]) - 1, 0, self.x_y_z[0]
        )
        self.voxel_y = np.clip(
            np.searchsorted(self.segments[1], self._points[:, 1]) - 1, 0, self.x_y_z[1]
        )
        self.voxel_z = np.clip(
            np.searchsorted(self.segments[2], self._points[:, 2]) - 1, 0, self.x_y_z[2]
        )
        # for each point get the index of the voxel it belongs to
        self.voxel_n = np.ravel_multi_index([self.voxel_x, self.voxel_y, self.voxel_z], self.x_y_z)

        # compute center of each voxel
        midsegments = [(self.segments[i][1:] + self.segments[i][:-1]) / 2 for i in range(3)]
        self.voxel_centers = cartesian(midsegments).astype(np.float32)

        if self.properties is not None:
            order = np.argsort(self.voxel_n)

            _, breaks, counts = np.unique(
                self.voxel_n[order], return_index=True, return_counts=True
            )
            repeated_counts = np.repeat(counts[:, None], self.properties.shape[1], axis=1)
            summed_colors = np.add.reduceat(self.properties, breaks, axis=0)
            averaged_colors = summed_colors / repeated_counts
            self.averaged_properties = averaged_colors

    def query(self, points):
        """Query structure."""
        voxel_x = np.clip(np.searchsorted(self.segments[0], points[:, 0]) - 1, 0, self.x_y_z[0])
        voxel_y = np.clip(np.searchsorted(self.segments[1], points[:, 1]) - 1, 0, self.x_y_z[1])
        voxel_z = np.clip(np.searchsorted(self.segments[2], points[:, 2]) - 1, 0, self.x_y_z[2])
        voxel_n = np.ravel_multi_index([voxel_x, voxel_y, voxel_z], self.x_y_z)

        return voxel_n

    def get_feature_vector(self, mode="binary", flatten: bool = False):
        """Get feature vector.

        Args:
            mode (str): Available modes are:
                * binary: 0 for empty voxels, 1 for occupied.
                * density: number of points inside voxel / total number of points.
                * TDF: Truncated Distance Function. Value between 0 and 1 indicating
                the distance between the voxel's center and the closest point. 1 on the surface,
                0 on voxels further than 2 * voxel side.
                Defaults to "binary".
            flatten (bool): Returns a flattened vector.
                Defaults to False.

        Raises:
            NotImplementedError: If the mode is not implemented.

        Returns:
            np.ndarray: _description_
        """
        vector = np.zeros(self.n_voxels)

        if mode == "binary":
            vector[np.unique(self.voxel_n)] = 1

        elif mode == "density":
            count = np.bincount(self.voxel_n)
            vector[: len(count)] = count
            vector /= len(self.voxel_n)

        elif mode == "TDF":
            kdt = cKDTree(self._points)
            vector, i = kdt.query(self.voxel_centers, n_jobs=-1)

        elif isinstance(mode, int):
            unique_voxels = np.unique(self.voxel_n)
            vector[unique_voxels] = self.averaged_properties[:, mode]

        else:
            raise NotImplementedError("{} is not a supported feature vector mode".format(mode))

        if flatten:
            return vector
        return vector.reshape(self.x_y_z)

    def get_voxel_neighbors(self, voxel):
        """Get all voxel neighbors.

        Args:
            voxel (int): Voxel index.

        Returns:
            List[int]: Indices of the valid, non-empty 26 neighborhood around voxel.
        """
        x, y, z = np.unravel_index(voxel, self.x_y_z)

        valid_x = []
        valid_y = []
        valid_z = []
        if x - 1 >= 0:
            valid_x.append(x - 1)
        if y - 1 >= 0:
            valid_y.append(y - 1)
        if z - 1 >= 0:
            valid_z.append(z - 1)

        valid_x.append(x)
        valid_y.append(y)
        valid_z.append(z)

        if x + 1 < self.x_y_z[0]:
            valid_x.append(x + 1)
        if y + 1 < self.x_y_z[1]:
            valid_y.append(y + 1)
        if z + 1 < self.x_y_z[2]:
            valid_z.append(z + 1)

        valid_neighbor_indices = cartesian((valid_x, valid_y, valid_z))

        ravel_indices = np.ravel_multi_index(
            (
                valid_neighbor_indices[:, 0],
                valid_neighbor_indices[:, 1],
                valid_neighbor_indices[:, 2],
            ),
            self.x_y_z,
        )

        return [x for x in ravel_indices if x in np.unique(self.voxel_n)]
