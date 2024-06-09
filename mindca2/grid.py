import math
import functools

import numpy as np
import numpy.typing as npt
from dtypes import Cell

type GridArr = npt.NDArray[np.bool_]


class ReuseConstraintViolation(Exception):
    def __init__(self, gridarr: GridArr, row: int, col: int):
        super().__init__(
            f"Channel Reuse Constraint violated in Cell {row=} {col=} on grid\n{gridarr}"
        )


class Grid:
    """
    Rhombus grid with an axial coordinate system.
    """

    def __init__(
        self, rows: int, cols: int, n_channels: int, *args, state=None, **kwargs
    ):
        self.rows = rows
        self.cols = cols
        self.n_channels = n_channels

        if state is None:
            # For each cell, for each channel, marks whether the channel is in use or not
            self.state = np.zeros((self.rows, self.cols, self.n_channels), dtype=bool)
        else:
            self.state = state

    @classmethod
    def from_arr[T: Grid](cls: type[T], gridarr: GridArr) -> T:
        """Construct from an existing numpy array"""
        return cls(*gridarr.shape, state=gridarr)

    def validate_reuse_constr(self) -> None:
        """
        Verify that the channel reuse constraint of 3 is not violated.
        The reuse constraint is violated if a channel that is in use at a cell
        is also in use in any of its neighbors with a distance of 2 or less.

        Raises a ReuseConstraintViolation if the channel reuse constraint
        is violated
        """
        for r in range(self.rows):
            for c in range(self.cols):
                # Neighbor indecies
                neighs = self.neighbors2(r, c, True)
                # Channels in use at any neighbor
                inuse_neigh = np.bitwise_or.reduce(self.state[neighs])
                # Channels in use at neighbor AND focal cell
                inuse = np.bitwise_and(self.state[r][c], inuse_neigh)
                if np.any(inuse):
                    raise ReuseConstraintViolation(self.state, r, c)

    @staticmethod
    def move_n(row: int, col: int) -> tuple[int, int]:
        """Move north"""
        return (row - 1, col)

    @staticmethod
    def move_ne(row: int, col: int) -> tuple[int, int]:
        """Move north-east"""
        if col % 2 == 0:
            return (row, col + 1)
        else:
            return (row - 1, col + 1)

    @staticmethod
    def move_se(row: int, col: int) -> tuple[int, int]:
        """Move south-east"""
        if col % 2 == 0:
            return (row + 1, col + 1)
        else:
            return (row, col + 1)

    @staticmethod
    def move_s(row: int, col: int) -> tuple[int, int]:
        """Move south"""
        return (row + 1, col)

    @staticmethod
    def move_sw(row: int, col: int) -> tuple[int, int]:
        """Move south-west"""
        if col % 2 == 0:
            return (row + 1, col - 1)
        else:
            return (row, col - 1)

    @staticmethod
    def move_nw(row: int, col: int) -> tuple[int, int]:
        """Move north-west"""
        if col % 2 == 0:
            return (row, col - 1)
        else:
            return (row - 1, col - 1)

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def neighbors1sparse(row, col) -> list[tuple[int, int]]:
        """
        Returns a list with indexes of neighbors within a radius of 1,
        not including self. The indexes may not be within grid.
        In clockwise order starting from up-right.
        """
        idxs = [
            Grid.move_n(row, col),
            Grid.move_ne(row, col),
            Grid.move_se(row, col),
            Grid.move_s(row, col),
            Grid.move_sw(row, col),
            Grid.move_nw(row, col),
        ]
        return idxs

    @functools.lru_cache(maxsize=None)
    def neighbors2(
        self, row: int, col: int, separate: bool = False, include_self: bool = False
    ) -> list[tuple[int, int]] | tuple[list[int], list[int]]:
        """
        If 'separate' is True, return ([r1, r2, ...], [c1, c2, ...])
        else return [(r1, c1), (r2, c2), ...]

        Returns a list with indexes of neighbors within a radius of 2,
        not including the cell self.
        """
        if separate:
            rs = []
            cs = []
        else:
            idxs = []

        r_low = max(0, row - 2)
        r_hi = min(self.rows - 1, row + 2)
        c_low = max(0, col - 2)
        c_hi = min(self.cols - 1, col + 2)
        if col % 2 == 0:
            cross1 = row - 2
            cross2 = row + 2
        else:
            cross1 = row + 2
            cross2 = row - 2
        for r in range(r_low, r_hi + 1):
            for c in range(c_low, c_hi + 1):
                if not (
                    (not include_self and (r, c) == (row, col))
                    or (r, c) == (cross1, col - 2)
                    or (r, c) == (cross1, col - 1)
                    or (r, c) == (cross1, col + 1)
                    or (r, c) == (cross1, col + 2)
                    or (r, c) == (cross2, col - 2)
                    or (r, c) == (cross2, col + 2)
                ):
                    if separate:
                        rs.append(r)
                        cs.append(c)
                    else:
                        idxs.append((r, c))
        if separate:
            return (rs, cs)
        else:
            return idxs


    def _get_eligible_chs_bitmap(self, cell: Cell) -> npt.NDArray[np.bool_]:
        """
        Get a mask/bitmap of eligible channels at the given cell

        A channel is True is it is use
        """
        r, c = cell
        # Find eligible chs by bitwise ORing the allocation maps of neighbors
        neighs = self.neighbors2(r, c, separate=True, include_self=True)
        alloc_map = np.bitwise_or.reduce(self.state[neighs])
        eligible_chs = np.invert(alloc_map)
        return eligible_chs


    def get_eligible_chs(self, cell: Cell) -> npt.NDArray[np.int_]:
        """
        Find the channels that are free in 'cell' and all of
        its neighbors with a distance of 2 or less.

        These are the eligible channels, i.e. those that can be assigned
        without violating the reuse constraint.
        """
        eligible_map = self._get_eligible_chs_bitmap(cell)
        eligible = np.nonzero(eligible_map)[0]
        return eligible


    def get_n_eligible_chs(self, cell: Cell) -> int:
        """Return the number of eligible channels"""
        eligible_map = self._get_eligible_chs_bitmap(cell)
        n_eligible = eligible_map.sum()
        return n_eligible


class FixedGrid(Grid):
    """
    A grid where each cell has a fixed set of channels that it can allocate
    calls on.

    Assigns the channels to cells such that they will not interfere with each
    other within a channel reuse constraint of 3.
    The channels assigned to a cell are its nominal channels.
    """

    def __init__(self, n_nom_channels: int = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # A label for each cell. Each cell with the same label will share
        # the same set of nominal channels.
        self.labels = np.zeros((self.rows, self.cols), dtype=int)
        self._partition_cells()
        # Nominal channels for each cell. The index (r, c, ch) is True
        # if the given channel is a nominal channel for that cell; False if not.
        self.nom_chs = np.zeros((self.rows, self.cols, self.n_channels), dtype=bool)
        self._assign_chs(n_nom_channels)

    def _partition_cells(self) -> None:
        """
        Partition cells into 7 lots such that the minimum distance
        between cells with the same label ([0..6]) is at least 2
        (which corresponds to a minimum reuse distance of 3).
        """

        def right_up(x, y):
            x_new = x + 3
            y_new = y
            if x % 2 != 0:
                # Odd column
                y_new = y - 1
            return (x_new, y_new)

        def down_left(x, y):
            x_new = x - 1
            if x % 2 == 0:
                # Even column
                y_new = y + 3
            else:
                # Odd Column
                y_new = y + 2
            return (x_new, y_new)

        def label(l, x, y):
            # A center and some part of its subgrid may be out of bounds.
            if x >= 0 and x < self.cols and y >= 0 and y < self.rows:
                self.labels[y][x] = l

        # Center of a 'circular' 7-cell subgrid where
        # each cell has a unique label
        center = (0, 0)
        # First center in current row which has neighbors inside grid
        first_row_center = (0, 0)
        # Move center down-left until subgrid goes out of bounds
        while (center[0] >= -1) and (center[1] <= self.rows):
            # Move center right-up until subgrid goes out of bounds
            while (center[0] <= self.cols) and (center[1] >= -1):
                # Label cells 0..6 with given center as 0
                label(0, *center)
                for i, neigh in enumerate(self.neighbors1sparse(center[1], center[0])):
                    label(i + 1, neigh[1], neigh[0])
                center = right_up(*center)
            center = down_left(*first_row_center)
            # Move right until x >= -1
            while center[0] < -1:
                center = right_up(*center)
            first_row_center = center

    def _assign_chs(self, n_nom_channels: int = 0) -> None:
        """
        Partition the cells and channels up to and including 'n_nom_channels'
        into 7 lots, and assign
        the channels to cells such that they will not interfere with each
        other within a channel reuse constraint of 3.
        The channels assigned to a cell are its nominal channels.

        Sets a (rows*cols*n_channels) array
        where a channel for a cell has value 1 if nominal, 0 otherwise.
        """
        if n_nom_channels == 0:
            n_channels = self.n_channels
        channels_per_subgrid_cell = []
        channels_per_subgrid_cell_accu = [0]
        channels_per_cell = n_channels / 7
        ceil = math.ceil(channels_per_cell)
        floor = math.floor(channels_per_cell)
        tot = 0
        for i in range(7):
            if tot + ceil + (6 - i) * floor > n_channels:
                tot += ceil
                cell_channels = ceil
            else:
                tot += floor
                cell_channels = floor
            channels_per_subgrid_cell.append(cell_channels)
            channels_per_subgrid_cell_accu.append(tot)
        for r in range(self.rows):
            for c in range(self.cols):
                label = self.labels[r][c]
                lo = channels_per_subgrid_cell_accu[label]
                hi = channels_per_subgrid_cell_accu[label + 1]
                self.nom_chs[r][c][lo:hi] = 1
