# NAS for declarative_privacy_preserving

You may find examples for the search process with different accuracy and speed tradeoff in [NAS_Demo.ipynb](NAS_Demo.ipynb).

The search function can be found in [search.py](search.py) with detailed instruction. You can tune the argument `flops_balance_factor` to balance performance (CE Loss or MSE Loss) and speed (FLOPs). Larger `flops_balance_factor` leads to a faster network with worse performance. You can find examples with different `flops_balance_factor` in [NAS_Demo.ipynb](NAS_Demo.ipynb).
