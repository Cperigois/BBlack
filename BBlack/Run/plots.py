import pandas as pd
import matplotlib as plt

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import json

params = json.load(open('Run/Params.json', 'r'))

def plots():

    #extract parameters
    variation = params['observable_variation']
    channelLabels = params['astro_model_list']
    runs = params['observing_runs']

    # Basic markers and colors
    markers = ['X', 'P', 'o', '*', 'p', 'd', '^', 'v', '<', '>', 's']
    cmap = plt.get_cmap('plasma')

    #generate colormaps
    colors = cmap(np.linspace(0, 1, len(channelLabels)))
    cmdict = dict.fromkeys(channelLabels)
    for i in range(len(channelLabels)):
        dict[channelLabels[i]] = {'marker': markers[i], 'color' : colors[i]}
    observables = params['observable_range']


for var in variation.keys():
# Set the different colors for each channel
    channelFileName = ['AGN_Dyn', 'AGN_Dyn_NoGasHardening', 'Field', 'GC_Dyn', 'NSC_Dyn', 'YSC_Dyn']
    colorByChannel = ['darkblue', 'crimson', 'forestgreen', 'darkorange', 'gold', 'pink']
    markerByChannel = ['X', 'P', 'o', '*', 'p', 'd']

runs = ['O1', 'O2', 'O3a_CAT', 'O3b_CAT']

catObservationsName = 'LVKEvents.dat'
cat = pd.read_csv(catObservationsName, sep='\t', index_col=None)
catSelection = cat[(cat['m2'] > 3) & (cat[
                                          'pastro'] > 0.90)]  # select events with m2>2(remove BNS ans NSBH), and very confident events p_astro > 0.9

path2Catalogs = 'Catalogs/'

McRange = {'min': 5,
           'max': 90,
           'bins': np.linspace(5, 90, 50)}
qRange = {'min': 0,
          'max': 1,
          'bins': np.linspace(0, 1, 50)}
chieffRange = {'min': -0.75,
               'max': 0.75,
               'bins': np.linspace(-0.75, 0.75, 50)}
m1Range = {'min': 3,
           'max': 130,
           'bins': np.linspace(3, 130, 50)}

legend_elements = []
events = np.array([])

plt.figure(figsize=(15, 8))
with PdfPages('Plots/Hist2D_log.pdf') as pdf:
    for channelIdx in range(len(channelLabels)):
        PaolaCatalogName = 'Catalog_co_BBH_formation_channel_' + channelFileName[channelIdx] + '.dat'
        # PaolaCatalogName = 'Princess_'+channelFileName[channelIdx]+'.dat'
        Catalog = pd.read_csv(path2Catalogs + PaolaCatalogName, sep='\t', index_col=None)
        print(Catalog.describe())
        Catalog = Catalog[Catalog['snr_HLV_opt'] > 8]  # from the catalog select detected events (SNR > 8)

        fig = plt.figure(constrained_layout=False)
        gs = fig.add_gridspec(3, 2, left=0.15, right=0.95, top=0.9, bottom=0.15,
                              wspace=0.5, hspace=0.5)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist2d(Catalog.m1, Catalog.q, bins=[m1Range['bins'], qRange['bins']], norm=mpl.colors.LogNorm())
        ax1.errorbar(catSelection['m1'], catSelection['q'],
                     xerr=[catSelection['m1_low'] + catSelection['m1'], catSelection['m1_up'] + catSelection['m1']],
                     yerr=[catSelection['q_low'] + catSelection['q'], catSelection['q_up'] + catSelection['q']],
                     fmt='o', mfc='crimson',
                     mec='crimson', ms=1, mew=1, elinewidth=0, color='crimson')
        ax1.set_xlabel(r'$m_1$ [M$_\odot$]', fontsize=8)
        ax1.set_ylabel(r'$q$', fontsize=8)
        ax1.set_xlim(m1Range['min'], m1Range['max'])
        ax1.set_ylim(qRange['min'], qRange['max'])

        ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
        ax2.hist2d(Catalog.Mc, Catalog.q, bins=[McRange['bins'], qRange['bins']], norm=mpl.colors.LogNorm())
        ax2.errorbar(catSelection['Mc'], catSelection['q'],
                     xerr=[catSelection['Mc_low'] + catSelection['Mc'], catSelection['Mc_up'] + catSelection['Mc']],
                     yerr=[catSelection['q_low'] + catSelection['q'], catSelection['q_up'] + catSelection['q']],
                     fmt='o', mfc='crimson',
                     mec='crimson', ms=1, mew=1, elinewidth=0, color='crimson')
        ax2.set_xlabel(r'$\mathcal{M}_{\rm chirp}$ [M$_\odot$]', fontsize=8)
        ax2.set_ylabel(r'$q$', fontsize=8)
        ax2.set_xlim(McRange['min'], McRange['max'])
        ax2.set_ylim(qRange['min'], qRange['max'])

        ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)
        ax3.hist2d(Catalog.m1, Catalog.chieff, bins=[m1Range['bins'], chieffRange['bins']], norm=mpl.colors.LogNorm())
        ax3.errorbar(catSelection['m1'], catSelection['chieff'],
                     xerr=[catSelection['m1_low'] + catSelection['m1'], catSelection['m1_up'] + catSelection['m1']],
                     yerr=[catSelection['chieff_low'] + catSelection['chieff'],
                           catSelection['chieff_up'] + catSelection['chieff']],
                     fmt='o', mfc='crimson',
                     mec='crimson', ms=1, mew=1, elinewidth=0, color='crimson')
        ax3.set_xlabel(r'$m_1$ [M$_\odot$]', fontsize=8)
        ax3.set_ylabel(r'$\chi_{eff}$', fontsize=8)
        ax3.set_xlim(m1Range['min'], m1Range['max'])
        ax3.set_ylim(chieffRange['min'], chieffRange['max'])

        ax4 = fig.add_subplot(gs[1, 1], sharey=ax3, sharex=ax2)
        ax4.hist2d(Catalog.Mc, Catalog.chieff, bins=[McRange['bins'], chieffRange['bins']], norm=mpl.colors.LogNorm())
        ax4.errorbar(catSelection['Mc'], catSelection['chieff'],
                     xerr=[catSelection['Mc_low'] + catSelection['Mc'], catSelection['Mc_up'] + catSelection['Mc']],
                     yerr=[catSelection['chieff_low'] + catSelection['chieff'],
                           catSelection['chieff_up'] + catSelection['chieff']],
                     fmt='o', mfc='crimson',
                     mec='crimson', ms=1, mew=1, elinewidth=0, color='crimson')
        ax4.set_xlabel(r'$\mathcal{M}_{\rm chirp}$ [M$_\odot$]', fontsize=8)
        ax4.set_ylabel(r'$\chi_{eff}$', fontsize=8)
        ax4.set_ylim(chieffRange['min'], chieffRange['max'])
        ax4.set_xlim(McRange['min'], McRange['max'])

        ax5 = fig.add_subplot(gs[2, 0])
        ax5.hist2d(Catalog.q, Catalog.chieff, bins=[qRange['bins'], chieffRange['bins']], norm=mpl.colors.LogNorm())
        ax5.errorbar(catSelection['q'], catSelection['chieff'],
                     yerr=[catSelection['chieff_low'] + catSelection['chieff'],
                           catSelection['chieff_up'] + catSelection['chieff']],
                     xerr=[catSelection['q_low'] + catSelection['q'], catSelection['q_up'] + catSelection['q']],
                     fmt='o', mfc='crimson',
                     mec='crimson', ms=1, mew=1, elinewidth=0, color='crimson')
        ax5.set_xlabel(r'$q$', fontsize=8)
        ax5.set_ylim(chieffRange['min'], chieffRange['max'])
        ax5.set_xlim(qRange['min'], qRange['max'])
        ax5.set_ylabel(r'$\chi_{eff}$', fontsize=8)

        fig.suptitle(channelFileName[channelIdx], fontsize=12)

        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()
