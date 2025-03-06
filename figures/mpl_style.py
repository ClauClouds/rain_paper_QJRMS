"""
Color constants and matplotlib style definitions
"""

import matplotlib as mpl
import cmcrameri.cm as cmc

COLOR_SHALLOW = "#ff9500"
COLOR_CONGESTUS = "#008080"
COLOR_N = 'royalblue'#(237/255, 174/255, 73/255)#cmc.grayC(0.75)
COLOR_T = 'lightsteelblue'#(209/255, 73/255, 91/255)#cmc.grayC(0.5)
COLOR_S = 'lavender' #(0/255, 121/255, 140/255) #cmc.grayC(0.25)
CMAP_HF_ALL = cmc.grayC
CMAP = cmc.batlow
CMAP_an = cmc.vik
CMAP_discr = cmc.batlowW
cmap_rs_shallow = cmc.roma
cmap_rs_congestus = cmc.managua
cmap_gif = cmc.vik
cmap_gif2 = cmc.vanimo

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
