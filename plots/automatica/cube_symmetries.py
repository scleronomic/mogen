import numpy as np
from wzk import new_fig

n = 360
angle = np.zeros((4, n))
angle[:, :] = np.arange(n)[np.newaxis, :]
angle[1] = np.roll(angle[1], +90)
angle[2] = np.roll(angle[2], 180)
angle[3] = np.roll(angle[3], 270)

fig, ax = new_fig(aspect=1)
ax.hlines([0, 90, 180, 270, 360], xmin=0, xmax=360, color='black', ls=':', lw=1)
ax.plot(angle[0], color='blue', alpha=0.8)
ax.plot(angle[1], color='blue', alpha=0.6)
ax.plot(angle[2], color='blue', alpha=0.4)
ax.plot(angle[3], color='blue', alpha=0.2)

# The mapping for the network to learn is
#  Input:
#   position: xyz [table size]
#   angle: [0-90] (is periodic so maybe choose 2D unit vector instead)
#   height: [table height + 30 cm]
#   environment: bps
#
#  Output:
#   A) q, hopefully this is close enough that I can infer the correct options out of the 4 possibilities
#   B) q + one hot
#
# rows to save in SQL: f, q
#
# Alternative:
#   just learn f -> q in general but restrict the input space to the described
#   advantages: easier to disturb the frames, cleaner interface
#   disadvantages:  need to check the 4 cases manually, not the smallest encoding, the planner afterwards should be able to handle the cases anyway
#
