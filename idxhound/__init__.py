import bidict
from collections import abc
import numpy as np


class Selection(bidict.FrozenOrderedBidict):
    """
    Selection that tracks indexes across applications.

    Parameters
    ----------
    obj : np.ndarray or iterable
        Selection object supported by numpy advanced indexing.
    mapping : bool
        Whether the first argument is a mapping (primarily for internal use).
    """
    def __init__(self, obj, *, mapping=False, **kwargs):
        if not mapping:
            obj = np.asarray(obj)
            self._array = obj
            if obj.ndim != 1:
                raise ValueError("selection object must be one-dimensional")
            if obj.dtype == bool:
                obj, = np.nonzero(obj)
            obj = [(i, j) for j, i in enumerate(obj)]

        super(Selection, self).__init__(obj, **kwargs)

    def __array__(self):
        """
        Allows the index to act as a numpy array [1], e.g. for indexing.

        References
        ----------
        .. [1] Writing custom array containers.
           https://numpy.org/doc/stable/user/basics.dispatch.html
        """
        if not hasattr(self, '_array') or self._array is None:
            self._array = np.fromiter(self, int)
        return self._array

    def __matmul__(self, other):
        return self.compose(other)

    def __getitem__(self, key):
        if isinstance(key, abc.Collection) and not isinstance(key, (str, bytes, bytearray)):
            return [self[x] for x in key]
        return super(Selection, self).__getitem__(key)

    def compose(self, other):
        """
        Evaluate the composite selection equivalent to applying this selection followed by another.

        .. note::

           Selection composition is not commutative.

        Parameters
        ----------
        other : Selection
            Selection to apply subsequently.

        Returns
        -------
        composite : Selection
            Composite selection equivalent to applying this selection followed by another.
        """
        if not isinstance(other, Selection):
            other = Selection(other)
        return self.__class__(
            [(self.inverse[key], value) for key, value in other.items()],
            mapping=True,
        )

    @classmethod
    def from_iterable(cls, keys):
        """
        Create a selection object from an iterable sequence of keys.

        .. note::

           Calling the constructor directly is suitable in most cases, but this method may be useful
           if the keys are provided in the form of an iterator or another type not easily castable
           to a numpy array, such as a :py:class:`set`.

        Parameters
        ----------
        keys : iterable
            Sequence of keys.

        Returns
        -------
        obj : Selection
            Selection with the given sequence of keys.

        Examples
        --------
        Create a mapping from a sequence of characters.

        >>> idxhound.Selection.from_iterable('abc')
        Selection([('a', 0), ('b', 1), ('c', 2)])
        """
        return cls([(x, i) for i, x in enumerate(keys)], mapping=True)
