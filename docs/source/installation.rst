Installation
================

Do to the external dependencies, we recommend installation via conda

.. code-block:: shell

    $ conda install -c conda-forge mofdscribe

The most recent release can be installed from
`PyPI <https://pypi.org/project/mofdscribe>`_ with:

.. code-block:: shell

    $ pip install mofdscribe

However, in this case, the following dependencies need to be manually installed
(e.g. via conda):

.. code-block:: shell

    conda install -c conda-forge cgal zeopp-lsmo raspa2

The most recent code and data can be installed directly from GitHub with:

.. code-block:: shell

    $ pip install git+https://github.com/kjappelbaum/mofdscribe.git

To install in development mode, use the following:

.. code-block:: shell

    $ git clone git+https://github.com/kjappelbaum/mofdscribe.git
    $ cd mofdscribe
    $ pip install -e .