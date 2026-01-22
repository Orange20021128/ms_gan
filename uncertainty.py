# uncertainty.py
import numpy as np

def dispersion(realizations):
    """
    realizations: (N, nx, ny, nz)
    """
    mpp = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(),
        axis=0,
        arr=realizations
    )

    disp = np.mean(realizations != mpp, axis=0)
    return mpp, disp
