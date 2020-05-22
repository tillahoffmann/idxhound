🐶 idxhound
===========

.. image:: https://github.com/tillahoffmann/idxhound/workflows/Python%20package/badge.svg
  :target: https://github.com/tillahoffmann/idxhound/actions?query=workflow%3A%22Python+package%22

.. image:: https://img.shields.io/pypi/v/idxhound.svg?style=flat-square
   :target: https://pypi.python.org/pypi/idxhound

.. image:: https://readthedocs.org/projects/idxhound/badge/?version=latest
  :target: https://idxhound.readthedocs.io/en/latest/?badge=latest

``numpy`` provides outstanding indexing through its advanced indexing capabilities [1]_. ``idxhound`` tracks indices across one or more selections to make sure you always know where your data (in the form of array elements) came from.

Alternatives include :py:class:`pandas.Index` and :py:class:`xarray.DataArray` which allow for indices other than monotonic integers. But sometimes one just wants to deal with the raw data arrays, e.g. to avoid any impact on performance or integrate with third-party libraries that expect raw numpy arrays. That's where ``idxhound`` can help.

Getting started
---------------

Obtaining a :py:class:`idxhound.Selection` object is straightforward: simply pass a selection object (such as a boolean filter or array of integer indices) as an argument to the constructor. For example, let's create an array and filter it using a boolean selection.

>>> x = np.asarray(list('abcdef'))
>>> obj = idxhound.Selection(x > 'c')
>>> y = x[obj]
>>> y
array(['d', 'e', 'f'], dtype='<U1')

The indexing behaviour is exactly the same as if we'd used ``y = x[x > 'c']``. But ``obj`` allows us to track where the elements in ``x`` ended up in ``y``. The example below illustrates how to find the index of ``x[3]`` in ``y``.

>>> i = obj[3]
>>> i, y[i]
(0, 'd')

But indexing by an element that has been eliminated by the selection raises an error as one might expect.

>>> obj[2]
Traceback (most recent call last):
    ...
KeyError: 2

Using the inverse of ``i`` allows us to retrieve the index of an element in ``x`` given its index in ``y``.

>>> j = obj.inverse[1]
>>> j, x[j], y[1]
(4, 'e', 'e')

Advanced use
------------

While the above examples illustrate that ``idxhound`` can deliver what was promised, more advanced use cases is where it shines.

Composition
^^^^^^^^^^^

Suppose we want to reorder and further filter the character sequence ``y`` but still keep track of indices across multiple selections. Easy!

>>> obj2 = idxhound.Selection([2, 0])
>>> y[obj2]
array(['f', 'd'], dtype='<U1')

Let's construct a composite index that has the same effect as the sequential application of selections.

>>> composite = obj @ obj2  # use the `compose` method for python < 3.5
>>> z = x[composite]
>>> z
array(['f', 'd'], dtype='<U1')

So where did the first element of ``z`` occur in ``x`` and ``y``, respectively?

>>> composite.inverse[0], obj2.inverse[0]
(5, 2)

Non-integer indices
^^^^^^^^^^^^^^^^^^^

Real data often use labels rather than integer indices (they might even be readable by humans if we're lucky). Suppose we have a simple dataset of populations of some European cities and we intend to order them.

>>> cities = ['Rome', 'Berlin', 'Paris', 'London']
>>> population = [2.873, 3.769, 2.148, 8.982]
>>> mapping = idxhound.Selection(cities)
>>> obj = (mapping @ np.argsort(population))
>>> obj[['London', 'Berlin']]
[3, 2]

London and Berlin would end up in last and second to last position in the ordered array, respectively. Indeed, they are the two largest cities. We can also easily retrieve the smallest city.

>>> obj.inverse[0]
'Paris'

Named columns
^^^^^^^^^^^^^

Because :py:class:`idxhound.Selection` is agnostic to the dimensions of the tensor being indexed, it can also be used to select named columns.

>>> latitude = [41.9028, 52.5200, 48.8566, 51.5074]
>>> longitude = [12.4964, 13.4050, 2.3522, 0.1278]
>>> data = np.transpose([population, latitude, longitude])
>>> columns = idxhound.Selection(['population', 'latitude', 'longitude'])
>>> data[mapping['Berlin'], columns[['latitude', 'longitude']]]
array([52.52 , 13.405])

Properties
----------

More formally, an :py:class:`idxhound.Selection` satisfies the following properties. Let ``x`` be a one-dimensional array, ``idx`` be a selection that can be applied to ``x``, ``y = x[idx]``, and ``obj = idxhound.Selection(idx)``. Then

1. indexing by ``obj`` is equivalent to indexing by ``idx``, i.e. all elements of ``y`` and ``x[obj]`` are equal,
2. ``obj[i]`` retrieves the index of the element in ``y`` given its index ``i`` in ``x``, i.e. ``x[i] == y[obj[i]]``,
3. and, conversely, ``obj.inverse[j]`` retrieves the index of the element in ``x`` given its index ``j`` in ``y``, i.e. ``x[obj.inverse[j]] == y[j]``.

.. [1] Indexing.
   https://numpy.org/doc/stable/reference/arrays.indexing.html#advanced-indexing
