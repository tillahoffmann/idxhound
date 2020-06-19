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
        >>> idxhound.Selection.from_iterable('abc')
        Selection([('a', 0), ('b', 1), ('c', 2)])
        """
        return cls([(x, i) for i, x in enumerate(keys)], mapping=True)


def array_to_dict(x, *objects, squeeze=True):
    """
    Convert an array to a dictionary of key-value pairs.

    Parameters
    ----------
    x : np.ndarray
        Array to convert.
    *objects : iterable[Selection]
        Selection objects corresponding to the axes of the array.
    squeeze : bool
        If ``True``, return a simple key when the array is one-dimensional. If ``False``, a
        :py:class:`tuple` with one element is returned (cf. :py:func:`dict_to_array`).

    Returns
    -------
    d : dict
        Dictionary of key-value pairs encoding the array.

    Example
    -------
    >>> cities = ['Rome', 'Berlin', 'Paris', 'London']
    >>> population = [2.873, 3.769, 2.148, 8.982]
    >>> idxhound.array_to_dict(population, idxhound.Selection(cities))
    {'Rome': 2.873, 'Berlin': 3.769, 'Paris': 2.148, 'London': 8.982}
    """
    x = np.asarray(x)
    if x.ndim != len(objects):
        raise ValueError('dimension of `x` and `objects` must match')
    # Get indices with shape `(ndim, prod(shape))`
    idx = np.reshape(np.indices(x.shape), (x.ndim, -1))
    # Map to the original space and transpose if necessary
    idx = [obj.inverse[i] for i, obj in zip(idx, objects)]
    if x.ndim == 1 and squeeze:
        idx, = idx
    else:
        idx = zip(*idx)
    return dict(zip(idx, x.ravel()))


def dict_to_array(d, *objects, fill_value=np.nan, dtype=None, squeezed=True):
    """
    Convert a dictionary of key-value pairs to an array.

    Parameters
    ----------
    d : dict
        Dictionary to convert.
    *objects : iterable[Selection]
        Selection objects corresponding to the axes of the array.
    fill_value : numbers.Number
        Value used for missing data.
    dtype :
        Data type of the resultant array.
    squeezed : bool
        If ``True`` and one selection object is provided, the keys of the dictionary are assumed to
        be elements of the selection. If ``False``, the keys of the dictionary are assumed to be
        tuples comprising elements of the selection ``objects`` (cf. :py:func:`array_to_dict`).

    Returns
    -------
    x : np.ndarray
        Array encoded by the dictionary of key-value pairs.

    Example
    -------
    >>> cities = ['Rome', 'Berlin', 'Paris', 'London']
    >>> population = {'Rome': 2.873, 'Berlin': 3.769, 'London': 8.982}
    >>> idxhound.dict_to_array(population, idxhound.Selection(cities))
    array([2.873, 3.769,   nan, 8.982])
    """
    # Create an array
    shape = tuple(max(obj.values()) + 1 for obj in objects)
    x = np.empty(shape, dtype)
    x[...] = fill_value
    # Transpose and map into the integer space
    if squeezed and x.ndim == 1:
        obj, = objects
        idx = [obj[i] for i in d]
    else:
        idx = tuple(obj[i] for i, obj in zip(zip(*d), objects))
    x[idx] = list(d.values())
    return x
