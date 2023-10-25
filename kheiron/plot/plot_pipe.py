import py3Dmol
import numpy as np


class PlotPipe:

    def __init__(self, plot_list):
        self.plot_list = plot_list

    def __call__(self, run, output, batch):
        [plot(run, output, batch) for plot in self.plot_list]

