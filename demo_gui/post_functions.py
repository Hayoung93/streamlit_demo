from typing import Union, List
nonetype = type(None)
import torch
import numpy as np
from collections import Iterable


def post_vote(
    preds: Union[np.ndarray,
                 torch.Tensor,
                 List[torch.Tensor],
                 List[np.ndarray],
                 list, tuple],
    th: float = 0.5
    ) -> Union[bool, nonetype]:
    """
    param preds: Sequence of probabilities for determining whether the video was made of deepfake or not
    param th: Threshold for determining whether the frame was made of deepfake or not. Frame with greater
              than or equal to this value will be considered as fake.
    
    return isfake: True if the number of fake frames is greater than of real frames, False if less than,
                   None if equal.
    """
    preds = torch.as_tensor(preds)
    assert len(preds.size()) == 1, "This function only supports 1D vector"
    preds[preds >= th] = 1
    preds[preds < th] = 0

    fakes = (preds == 1).sum()
    reals = (preds == 0).sum()

    isfake =  True if (fakes > reals) else False if (fakes < reals) else None
    return isfake