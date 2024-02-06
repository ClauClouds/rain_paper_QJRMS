"""
Color constants and matplotlib style definitions
"""

import matplotlib as mpl
import cmcrameri.cm as cmc

COLOR_SHALLOW = "#ff9500"
COLOR_CONGESTUS = "#008080"
COLOR_N = cmc.grayC(0.75)
COLOR_T = cmc.grayC(0.5)
COLOR_S = cmc.grayC(0.25)
CMAP = cmc.batlow
CMAP_an = cmc.vik
CMAP_discr = cmc.batlowW


mpl.rcParams["legend.frameon"] = False

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = [
    "Tahoma",
    "DejaVu Sans",
    "Lucida Grande",
    "Verdana",
]

mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.grid"] = True

mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["savefig.transparent"] = True
