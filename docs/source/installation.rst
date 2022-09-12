Installation
================

.. Do to the external dependencies, we recommend installation via conda

.. .. code-block:: shell

..     $ conda install -c conda-forge mofdscribe

.. The most recent release can be installed from
.. `PyPI <https://pypi.org/project/mofdscribe>`_ with:

.. .. code-block:: shell

..     $ pip install mofdscribe

.. However, in this case, the following dependencies need to be manually installed
.. (e.g. via conda):

.. .. code-block:: shell

..     conda install -c conda-forge cgal zeopp-lsmo raspa2

The most recent code and data can be installed directly from GitHub with:

.. code-block:: shell

    git clone git+https://github.com/kjappelbaum/mofdscribe.git
    cd mofdscribe
    pip install -e .

If you want to use all utilities, you can use the :code:`all` extra: :code:`pip install -e ".[all]"`

We depend on many other external tools. Currently, you need to manually install these dependencies (due to pending merges for conda-recipies):

.. code-block:: shell
    
    # RASPA and Zeo++ (if you want to use energy grid/Henry coefficient and pore descriptors)
    conda install -c conda-forge raspa2 zeopp-lsmo

    # cgal dependency for moltda (if you want to use persistent-homology based features)
    # on some systems, you might need to replace this with sudo apt-get install libcgal-dev or brew install cgal 
    conda install -c conda-forge cgal dionysus

    # openbabel dependency for moffragmentor (if you want to use SBU-centered features)
    conda install -c conda-forge openbabel
