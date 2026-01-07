import numpy as np
import matplotlib.pyplot as plt

import numpy as np

x = np.load(r"C:\Users\FS-GAMING PRO\Desktop\NN project\patch\synthetic_npy\adhd_40.npy")
print(x.shape, x.dtype)

i = 30  # pick a slice/frame index 0..39
plt.imshow(x[i], cmap="gray")
plt.title(f"Slice/Frame {i}  |  min={x[i].min():.1f} max={x[i].max():.1f}")
plt.axis("off")
plt.show()
