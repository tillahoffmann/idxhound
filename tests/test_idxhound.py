import idxhound
import numpy as np
import pytest


def test_from_iterable():
    sequence = 'abcde'
    index = idxhound.Selection.from_iterable(sequence)
    assert index['c'] == 2
    assert index.inverse[3] == 'd'


def test_inverse_type():
    index = idxhound.Selection.from_iterable('ab')
    assert isinstance(index.inverse, index.__class__)


def test_composition():
    a = 'abc'
    b = 'def'
    index1 = idxhound.Selection.from_iterable(a)
    index2 = idxhound.Selection.from_iterable(b).inverse
    index = index1 @ index2

    for x, y in zip(a, b):
        assert index[x] == y


def test_boolean():
    x = np.random.normal(0, 1, 100)
    fltr = x > np.median(x)
    index = idxhound.Selection(fltr)
    fltrd = x[fltr]
    for i, j in index.items():
        assert x[i] == fltrd[j]


def test_integer():
    x = np.random.normal(0, 1, 100)
    fltr = np.random.permutation(100)[:50]
    index = idxhound.Selection(fltr)
    fltrd = x[fltr]
    for i, j in index.items():
        assert x[i] == fltrd[j]


def test_indexing():
    x = np.random.normal(0, 1, 100)
    fltr = x > np.median(x)
    index = idxhound.Selection(fltr)
    np.testing.assert_array_equal(x[fltr], x[index])


def test_two_filters():
    x = np.random.normal(0, 1, 10)

    # First filtration
    fltr1 = x > np.median(x)
    idx1 = idxhound.Selection(fltr1)
    y = x[fltr1]

    # Second filtration
    fltr2 = y > np.median(y)
    z = y[fltr2]

    # Evaluate the composite index
    index = idx1 @ fltr2
    np.testing.assert_array_equal(z, x[index])
    np.testing.assert_array_equal(z[index.inverse], x[index])


def test_multiple_filters():
    num_filters = 4
    x = y = np.random.normal(0, 1, 100)
    index = None
    for _ in range(num_filters):
        fltr = y > np.median(y)
        y = y[fltr]
        if index is None:
            index = idxhound.Selection(fltr)
        else:
            index = index @ idxhound.Selection(fltr)

    np.testing.assert_array_equal(y, x[index])


def test_multiindex():
    index = idxhound.Selection.from_iterable('abc')
    assert index[['a', 'c']] == [0, 2]


def test_wrong_ndim():
    with pytest.raises(ValueError):
        idxhound.Selection(np.random.normal(size=(2, 2)))


def test_array_to_dict():
    x = np.random.normal(size=(2, 3))
    obj1 = idxhound.Selection.from_iterable('ab')
    obj2 = idxhound.Selection.from_iterable('xyz')
    d = idxhound.array_to_dict(x, obj1, obj2)
    assert d[('b', 'z')] == x[1, 2]


def test_array_to_dict_wrong_ndim():
    x = np.random.normal(size=(3, 4))
    with pytest.raises(ValueError):
        idxhound.array_to_dict(x)


@pytest.mark.parametrize('squeeze', [True, False])
def test_vector_to_dict(squeeze):
    x = np.random.normal(size=3)
    obj = idxhound.Selection.from_iterable('abc')
    d = idxhound.array_to_dict(x, obj, squeeze=squeeze)

    if squeeze:
        assert x[2] == d['c']
        for key in d:
            assert isinstance(key, str)
    else:
        assert x[2] == d[('c',)]
        for key in d:
            assert isinstance(key, tuple)


def test_dict_to_array():
    obj1 = idxhound.Selection.from_iterable('ab')
    obj2 = idxhound.Selection.from_iterable('xyz')
    d = {
        ('a', 'y'): 3,
        ('b', 'z'): 4,
    }
    x = idxhound.dict_to_array(d, obj1, obj2)
    np.testing.assert_array_equal(x, [
        [np.nan, 3, np.nan],
        [np.nan, np.nan, 4],
    ])


@pytest.mark.parametrize('shape', [
    (4,),
    (3, 8),
    (9, 7, 13),
])
def test_dict_array_roundtrip(shape):
    prefixes = 'abcdefg'
    objects = [idxhound.Selection([f'{prefixes[i]}{j}' for j in range(size)])
               for i, size in enumerate(shape)]
    x = np.random.normal(size=shape)
    d = idxhound.array_to_dict(x, *objects)
    y = idxhound.dict_to_array(d, *objects)
    np.testing.assert_array_equal(x, y)
