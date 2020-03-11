# DGI

## Overview
Here I provide an implementation of my R250 report on experiments with DGI in PyTorch. The repository is organised as follows:
- `models/` contains the implementation of the DGI pipeline (`dgi.py`) and our logistic regressor (`logreg.py`);
- `layers/` contains the implementation of all encoder layers GCN layer that were not provided by default from PyTorch and the averaging readout (`readout.py`), and the bilinear discriminator (`discriminator.py`);
- `utils/` contains the utils subroutines such as standardisation.

Finally, `execute.py` puts all of the above together and may be used to
execute a full training run on Cora. The experiment can be selected by
changing the experimental parameters in the `__main__` subroutine of `execute.py`
