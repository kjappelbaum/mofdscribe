Maintaining mofdscribe
========================

Updating datasets
------------------

Following tasks need to performed prior to updating the datasets:

- Recompute all hashes with the latest version of ``structuregraph_helpers``.
- Recompute all features with the latest version of ``mofdscribe``.


Updating the leaderboard
----------------------------

Typically, the leaderboard is updated automatically (after approving the PR).
However, if there is an unexpected issue, the leaderboard can be updated manually by running ``dev_scripts/update_bench.py``.