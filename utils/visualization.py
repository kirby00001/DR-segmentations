import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def image_show(image, masks):
    plt.imshow(image)
    cmaps = ["Greys_r", "Reds_r", "Greens_r", "Blues_r", "Purples_r"]
    masks = np.ma.masked_equal(masks, 0)
    for i in range(1, 5):
        plt.imshow(
            masks[:, :, i],
            alpha=0 if i == 0 else 0.5,
            cmap=cmaps[i],
            interpolation="none",
        )
    handles = [
        Rectangle((0, 0), 1, 1, color=_c)
        for _c in [
            (0.0, 0.0, 0.0),
            (0.667, 0.0, 0.0),
            (0.0, 0.667, 0.0),
            (0.0, 0.0, 0.667),
            (0.5, 0, 0.5),
        ]
    ]
    labels = ["Background", "MA", "HE", "EX", "SE"]
    plt.legend(handles, labels)
