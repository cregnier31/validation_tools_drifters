#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
import os
import time
import sys




def init_fig(title=None, xlabel=None, ylabel=None, y2label=None, xlim=None, ylim=None, y2lim=None, zero=None, c1='k', c2='grey', figsize=(10, 6), fontsize=14):
    """
    Initialize a figure with 1 ax or 2 axes (if y2label), title and labels.
    zero: float (horizontal line) or 'diag' (diagonal line). Need xlim
    c1/c2: ax and ax2 colors
    xlim/ylim: xlim=(x0,x1)
    """
    # General
    plt.rc('ytick',labelsize=fontsize); plt.rc('xtick',labelsize=fontsize)
    fig, ax = plt.subplots()

    fig.set_size_inches(figsize)
    if title is not None:      
        plt.title(title, fontsize=fontsize+2, fontweight='bold')
    ax.grid(linestyle='--', zorder = 0)

    # Left axis
    set_axis(ax, ylabel, c1, fontsize)
    if ylim: ax.set_ylim(ylim)
    
    # Right axis
    if y2label is not None:
        ax2  = ax.twinx()
        set_axis(ax2, y2label, c2, fontsize)
        if y2lim: ax2.set_ylim(y2lim)

    # x axis
    if xlim: plt.xlim(xlim)
    if xlabel is not None: 
        ax.set_xlabel(xlabel, fontsize=fontsize, fontweight='bold')
    
    # zero
    plot_zero(ax, zero, xlim)
    
    # Return
    if y2label is not None:
        return (fig, ax, ax2)
    else:
        return (fig, ax)


def plot_map(ax, lonmin, lonmax, latmin, latmax, lon_space=20, lat_space=10, res='l', proj='cyl', lines=True, fontsize=10):
    """ 
    Plot continents with basemap.
        - lonmin/lonmax: in [0 - 360]
        - lon_space/lat_space: space between 2 meridians/parallels in degrees
        - res: map resolution ('c' for crude; 'l' for low'; 'i' for intermediate; 'h' for high; 'f' for full) 
        - proj: projection ('cyl'; 'merc',...)
        - lines: to plot meridians and parallels
    """
    # Keep lonmax > lonmin
    if lonmax < lonmin:
        lonmax += 360.
    
    # Map
    map = Basemap(ax=ax, llcrnrlon=lonmin, llcrnrlat=latmin, urcrnrlon=lonmax, urcrnrlat=latmax, projection=proj, resolution=res)
           
    # Draw parallels & meridians
    if lines:
        map.drawparallels(np.arange(latmin+lat_space, latmax, lat_space), labels=[1,0,0,0], fontsize=fontsize, zorder= 1, linewidth=0.1)
        map.drawmeridians(np.arange(lonmin+lon_space, lonmax, lon_space), labels=[0,0,0,1], fontsize=fontsize, zorder= 1, linewidth=0.1)
    map.drawcountries(color='k',linewidth=0.05, zorder=1.2)
    map.drawcoastlines(color='grey',linewidth=0.2, zorder=1.2)
    map.fillcontinents(color='0.35', zorder = 1.1)
    
    return ax
    
    
def save_fig(fname, dpi=200, show=False):
    """
    Save figure fname as fname.png
    """
    if fname:
        fname += '.png'
        plt.savefig(fname, bbox_inches='tight', dpi=dpi)
        print ('Saved ', fname)
        if show:
            plt.show()
        plt.close()
        
        
def set_axis(ax, label=None, col='k', fontsize=12):
    """
    set ax with color col and label 
    """
    if label is not None:
        ax.set_ylabel(label, fontsize=fontsize, fontweight='bold')
    ax.tick_params(axis='y', colors = col)
    ax.yaxis.label.set_color(col)
    
    
def plot_zero(ax, zero, xlim, c='k', vertical=False):
    """
    Plot an horizontal line at y=zero or vertical line if vertical==True or diagonal line.
    xlim: xlim=(0,10) for example.
    """
    if zero != None and xlim:
        if vertical:
            y = np.linspace(xlim[0], xlim[1])
            ax.plot(zero * np.ones(len(y)), y,'--k', zorder=3, c=c)
        else:
            x = np.linspace(xlim[0], xlim[1])
            if zero == 'diag':
                ax.plot(x, x,'--k', zorder=3, c=c)   
            else:
                ax.plot(x, zero * np.ones(len(x)),'--k', zorder=3, c=c)

def init_polar(title=None, xlabel=None, ylabel=None, figsize=(8, 5), fontsize=14):

    plt.rc('ytick',labelsize=fontsize); plt.rc('xtick',labelsize=fontsize)
    fig = plt.figure(figsize=figsize) 
    ax = plt.subplot(111, projection='polar')
    plt.title(title, fontsize=fontsize+2, fontweight='bold')
    
    return fig, ax 



def compute_directions(U, V, nb_directions, bins_classes):
    """
    Compute directions histogram for current rose.
    Return:
        rose: yaxis of histogram, size nb_directions * nb_classes
              (nb of occurences per speed class and per direction).
        directions: x axis of histogram (-pi -> pi)
    """
    nb_classes = len(bins_classes) - 1
    
    # Norm and direction
    speed = np.array(np.sqrt(U**2 + V**2))
    theta = np.array(np.arctan2(V, U))
    
    # Flatten and remove nan
    speed = speed[~np.isnan(speed)]
    theta = theta[~np.isnan(theta)]

    # Compute polar histogram
    hist = np.histogram(theta, bins=nb_directions, range=(-np.pi, np.pi))   

    # Compute histogram on norm for each direction
    rose = np.zeros((nb_directions, nb_classes))
    for k in range(nb_directions):
        idx = np.where((theta >= hist[1][k]) & (theta < hist[1][k+1]))[0]
        sp = speed[idx]
        h = np.histogram(sp, bins=bins_classes)
        rose[k, :] = h[0]
    print(rose)
    directions = (hist[1][:-1] + hist[1][1:]) / 2
    return rose, directions


def current_rose(U, V, bins_classes=[0,0.1,0.2,0.3,0.5], title=None, fname=False, nb_directions=32, pmax='auto', legend=True, fontsize=20,diff = False):
    """
    Plot circular histogram of current velocity with different colors depending on current norm.
    
    Parameters:
        -U/V: current zonal/meridional velocity
        -bins_classes: classes of current velocity. Last class is V > bins_classes[-1]
        -title: figure title
        -save: False or figure name
        -nb_directions: divide the circle in nb_directions categories
        -pmax: maximum percentage of data for yaxis (auto or int)
        -legend: boolean, to plot legend
    """
    
    # Calculate circular histogram
    rose, directions = compute_directions(U, V, nb_directions, bins_classes)


    # Useful variables for plot
    delta = 2 * np.pi / nb_directions - 0.05 # Angle du segment de cercle
    width = np.ones(nb_directions) * 2 * np.pi /nb_directions - 0.05
    colors = ['blue', 'deepskyblue', 'greenyellow', 'yellow', 'orange', 'red', 'purple', 'k'] # max 8 classes
    if legend:
        labels = ['%.2f - %.2f m/s'%(bins_classes[k], bins_classes[k+1]) for k in range(len(bins_classes)-1)]
        labels[-1] = '> %.2f m/s'%bins_classes[-1]
    else:
        labels = [None for i in range(len(bins_classes))]

    # Decoration
    fig, ax = init_polar(figsize=(10,12), fontsize=fontsize)

    # Plot
    bottom=0
    nb_data = np.sum(~np.isnan(U))
    for k in range(len(bins_classes) - 1):
        height = rose[:, k] * 100 / nb_data
        ax.bar(directions, height, width=width, bottom=bottom, edgecolor='k', color = colors[k], label=labels[k])
        bottom += height

    # Maximum % for yticks
    if pmax == 'auto':
        pmax = np.max([np.sum(rose[k, :]) for k in range(nb_directions)]) * 100 / nb_data
    
    # Plot mean direction       
    mean_dir = np.arctan2(np.nanmean(V), np.nanmean(U))
    ax.plot([mean_dir, mean_dir],[0,pmax], '-.',color='k',linewidth =5)

    # Legend, title and ticks
    if legend:
        plt.legend(fontsize=fontsize, loc=(1,0), title='Current velocity (m/s)', title_fontsize=fontsize+2)
    plt.title(title, fontsize=fontsize+8, fontweight='bold')
    ax.set_xticks(np.linspace(0,2*np.pi-np.pi/8,16))
    if diff==True:
        ax.set_xticklabels(np.linspace(0,360 - 360 /16,16), fontsize=fontsize, fontweight='bold')    

    else :
        ax.set_xticklabels(('E','','NE','','N','','NW','','W','','SW','','S','','SE',''), fontsize=fontsize, fontweight='bold')    
    ax.set_ylabel('Percentage of total data', fontsize=fontsize, position=(0,0.75), rotation=60)
    ax.set_rlabel_position(150)      
    nticks = 5
    yticks = np.linspace(pmax/nticks, pmax, nticks, dtype='int')
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(i) + '%' for i in yticks], fontsize=fontsize)

    # Save figure
    save_fig(fname)
