import matplotlib.pyplot as plt

feats = [
    ['top_1', 0.4825940755],
    ['top_2', 0.4860865399],
    ['top_3', 0.498893917],
    ['top_4', 0.3657454759],
    ['top_5', 0.3657454759],
    ['top_6', 0.3808515722],
    ['top_7', 0.3808515722],
    ['top_8', 0.3847330075],
]

fig, ax = plt.subplots()
ax.set_ylim(0.2, 0.6)
ax.axhline(0.3030, ls='--')
plt.plot([x[0] for x in feats], [x[1] for x in feats], '-o')
plt.show()

