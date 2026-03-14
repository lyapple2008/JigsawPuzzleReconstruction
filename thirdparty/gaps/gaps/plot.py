import warnings

import matplotlib
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)


class Plot(object):
    MAX_WIDTH = 16
    MAX_HEIGHT = 12

    def __init__(self, image, title="Initial problem"):
        aspect_ratio = image.shape[0] / float(image.shape[1])

        # Calculate figure size while keeping within screen limits
        width = 8
        height = width * aspect_ratio

        # Scale down if too large
        if width > self.MAX_WIDTH:
            width = self.MAX_WIDTH
            height = width * aspect_ratio
        if height > self.MAX_HEIGHT:
            height = self.MAX_HEIGHT
            width = height / aspect_ratio

        fig = plt.figure(figsize=(width, height), frameon=False)

        # Let image fill the figure
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 0.9])
        ax.set_axis_off()
        fig.add_axes(ax)

        self._current_image = ax.imshow(image, aspect="auto", animated=True)
        self.show_fittest(image, title)

    def show_fittest(self, image, title):
        plt.suptitle(title, fontsize=20)
        self._current_image.set_data(image)
        plt.draw()

        # Give pyplot 0.05s to draw image
        plt.pause(0.05)
