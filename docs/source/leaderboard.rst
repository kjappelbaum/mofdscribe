Leaderboards
===================

Use the table of contents to navigate to the leaderboards for the various tasks.


.. toctree::
   :maxdepth: 2
   :caption: Leaderboards per task


   leaderboards/logkH_CO2_id
   leaderboards/logkH_CO2_ood
   leaderboards/pbe_bandgap_id
   leaderboards/ch4dc_id

Contributing
------------------

You want your model to appear here? Then

1. Run the :code:`bench` method for the task you want to contribute to.
2. Create a pull request with the :code:`json` file produced by the :code:`bench` method and fill the pull request template.

If you have a general-purpose model you might be interested in also submitting it to the `matbench <https://matbench.materialsproject.org>`_ leaderboard. We're currently investigating how to integrate the leaderboard into the matbench project.

If you have a modeling approach for a task that is not yet covered in the leaderboards, please open a new issue, and we'll make sure to add it.



.. admonition::  No benchmark for your problem?
    :class: note 

    Working on an ML problem for MOFs for which we do not have a benchmark? 
    Please let us know and work together with us to create one!
    This will make it much easier for the community to build on top of your work!