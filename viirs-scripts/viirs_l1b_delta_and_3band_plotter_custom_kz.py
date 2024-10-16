# This script produces a false colour composite (FCC) plot of the VIIRS VNP02IMG L1B
# I01 visible, I02 near-infrared and I03 shortwave-infrared (VIS, NIR, and SWIR,
# respectively) bands, with their histograms normalized.
#
# This script also produces an I04-I05 (MWIR-LWIR) delta plot, an I04 (MWIR)
# plot, and an I05 (LWIR) plot.

# =================================================================================
# Imports
# =================================================================================

# Built-ins.

# Used for file and folder path manipulation.
from pathlib import Path

# Universe.

# Used for reading of NetCDF4/HDF5 format VIIRS imagery, and WFS synthetic imagery.
import netCDF4 as nC

# Used for numerical processing and stacking of grids.
import numpy as np

# Used for image histogram equalization.
import cv2

# Used to perform copying of colormaps.
import copy

# Base plotting library.
import matplotlib as mpl

# Plotting.
import matplotlib.pyplot as plt

# =================================================================================
# Input and output files/folders
# =================================================================================

# Inputs.
input_viirs_data_path = Path(f'VNP02IMG.A2021227.2036.002.2022267024210.nc')

# Outputs.
output_data_folder = Path('')

# =================================================================================
# Constants
# =================================================================================

# Used to crop the input layers to a specific sample range.
# The values below correspond to a view zenith angle of < 30 degrees.
# They perform a "center swath" crop to remove all bowtie artifacts (a total crop
# of approx. two-thirds of the entire swath).
SAMPLE_CROP_INDICES_MIN = 2221  # inclusive
SAMPLE_CROP_INDICES_MAX = 4179  # exclusive

# FCC grid index references in 3D stack (RGB).
SWIR_GRID = 0  # red
NIR_GRID = 1   # green
VIS_GRID = 2   # blue

# Maximum possible RGB index (max uint8).
RGB_MAX = 255

# Plot-specific constants.
AXIS_LABELS_FONT_SIZE = 12
AXIS_MAJOR_TICKS_FONT_SIZE = 12
AXIS_MINOR_TICKS_FONT_SIZE = 12
COLORBAR_MAJOR_TICKS_FONT_SIZE = 12
DPI = 3000
FIGSIZE = (6, 10)
TITLE_FONT_SIZE = 12
TITLE_PADDING = 20

# Constants for debug / status console message verbosity levels.
DEBUG_VERBOSITY_LEVEL_0 = 0
DEBUG_VERBOSITY_LEVEL_1 = 1
DEBUG_VERBOSITY_LEVEL_2 = 2


# =================================================================================
# Plotting methods
# =================================================================================

def colorbar_v4(fig,
                ax,
                mesh,
                fontsize=8,
                ticks=None,
                ticklab=None,
                cblab=None,
                cbax_options=None,
                cb_options=None
                ):
    """
    A nice way to deal with colorbar sizing on mpl plots

    v2: + works on cartopy 'geoaxes' objects as well as regular matplotlib objects,
          so probably superceeds 'colorbar' func
    v3: + uses 'set_clim' to place ticks at patch mid-points on discrete CBs.
          see: https://jakevdp.github.io/PythonDataScienceHandbook/04.07-customizing-colorbars.html
          Doesn't seem to negatively impact continuous CBs
        + added arg to control scale of cbar, as best option varies depending on specific fig/subplot setup
    v4: + revised to make better use of kwargs:
          cbax_options: kwargs for make_axis.
          NOTE: if 'location': 'right', probably want 'orientation': 'vertical'

    ** ticks could possibly be in kwargs??

    NOTE: take care when making a 'banded' discrete map of continuous values (e.g. 0.1, ... 1 with 0.79 etc.).
    extent of each class will be abs(ticks[0] - ticks[1]) / 2 CENTRED on the tick values!

    :param fig:  mpl fig containing ax, mesh
    :param ax:   mpl axes containing mesh
    :param mesh:  matplotlib.cm.ScalarMappable (i.e., Image, ContourSet, etc.)
    :param fontsize: int
    :param ticks: list of floats, e.g. [0.5, 0.8, 0.16]
    :param ticklab: list of str, used as labels on cb - only used if 'ticks' not None. must be same length as 'ticks'
    :param cbax_options: dict, **kwargs for mpl.colorbar.make_axes(). 'shrink' scales cbar relative to ax, aspect
      changes width/height ratio
    :param cb_options: dict, **kwargs for fig.colorbar()
    :return: colorbar object
    """
    # cax, kw = mpl.colorbar.make_axes(ax, location='right', pad=0.05, shrink=cbar_scale)
    cax, kw = mpl.colorbar.make_axes(ax, **cbax_options)
    if ticks:
        out = fig.colorbar(mesh, cax=cax, ticks=ticks, **cb_options)
        # ~~~~~ new for v3 ~~~~~
        # need 'set_clim' as default behaviour doesn't allow segments
        # and ticks to line up properly.
        offset = abs(ticks[0] - ticks[1]) / 2
        out.mappable.set_clim(np.min(ticks) - offset, np.max(ticks) + offset)
        """
        print(f'NOTE: discrete colorbar specified. ensure that ticks match ticklabs! '
              f'tickrange={np.min(ticks)}-{np.max(ticks)} '
              f'clim={np.min(ticks) - offset}-{np.max(ticks) + offset}, '
              f'offset={offset}'
              )
        """
        # ~~~~~~~~~~~~~~~~~~~~~~~
        if cb_options.get('orientation') == 'vertical':
            out.ax.set_yticklabels(ticklab, fontsize=fontsize)
        elif cb_options.get('orientation') == 'horizontal':
            out.ax.set_xticklabels(ticklab, fontsize=fontsize)
        else:
            raise ValueError('colorbar_v4(): Error with cb_options - orientation.')
    else:
        out = fig.colorbar(mesh, cax=cax, **cb_options)
        out.ax.tick_params(labelsize=fontsize)
    
    if cblab:
        if cbax_options.get('location') == 'right':
            label = out.set_label(cblab, size=fontsize, rotation=-90, verticalalignment='bottom')
        elif cbax_options.get('location') == 'bottom':
            label = out.set_label(cblab, size=fontsize, rotation=0, verticalalignment='top')
        else:
            pass
    return out


def continuous_plotter(axis_labels_font_size: int,
                       axis_major_ticks_font_size: int,
                       axis_minor_ticks_font_size: int,
                       clim: list | None,
                       colorbar_ticks_font_size: int,
                       cpalette: str,
                       debug_verbosity: int,
                       dpi: int,
                       figsize: tuple[float, float],
                       grid: str | None,
                       img_array: np.ndarray,
                       label_xaxis: str | None,
                       label_yaxis: str | None,
                       output_data_folder: Path,
                       output_file_name: Path,
                       title: str | None,
                       title_font_size: int,
                       title_padding: float
                       ):
    """
    Plot array with imshow with specified cmap, sets NaN = grey.

    Optional contrast stretching.

    :param axis_minor_ticks_font_size: An int which specifies the axis minor ticks font size.
    :param axis_major_ticks_font_size: An int which specifies the axis major ticks font size.
    :param axis_labels_font_size: An int which specifies the axis labels font size.
    :param clim: Optional two element list, float. If set, creates a contrast stretch
      between these two values. Default = None.
    :param colorbar_ticks_font_size: An int which specifies the colorbar ticks font size.
    :param cpalette: A string to specify the name to use for the continuous colour palette (i.e. 'jet').
    :param debug_verbosity: An integer that defines the verbosity of debugging and status console messages.
    :param dpi: An integer that specifies the dots per inch (resolution) to use for the plot.
    :param figsize: A tuple of two integers to specify the (width, height) of the figure, in inches.
    :param grid: An optional string; if present, add grid lines of this colour.
    :param img_array: A 2D numpy array of raw data to plot.
    :param label_xaxis: A string label to be used for the x-axis of the plot.
    :param label_yaxis: A string label to be used for the y-axis of the plot.
    :param output_data_folder: A Path object representing the folder location to place the output plot image.
    :param output_file_name: A Path object representing the file name to use for the output plot image.
    :param title: A string to use for the title of the plot.
    :param title_font_size: An int which specifies the plot title font size.
    :param title_padding: A float which specifies the amount of padding to apply between the plot axis title
      and the plot itself, in units of "points".
    """
    
    if debug_verbosity >= DEBUG_VERBOSITY_LEVEL_1:
        print('continuous_plotter(): Plotting {} . . .'.format(title.split("\n")[0]))
    
    plt.close('all')
    
    # change default colour maps to plot NaN as grey
    cmap = copy.copy(eval('mpl.cm.' + cpalette))
    cmap.set_bad(color='black',
                 alpha=0.9
                 )
    
    # generate ticks for colour bar at evenly spaced points between min - max
    # if not specified, use array min/max
    if not clim:
        minval = np.nanmin(img_array)
        maxval = np.nanmax(img_array)
        lims = [minval, maxval]
    else:
        minval = clim[0]
        maxval = clim[1]
        lims = clim
    
    interval = (maxval - minval) / 4.0
    t = np.arange(minval, maxval, interval)
    t = np.round(np.append(t, maxval), decimals=1)
    
    # make figure and axes object
    fig1 = plt.figure(figsize=figsize,
                      num=1
                      )
    
    ax1 = fig1.add_axes((0.07, 0.1, 0.8, 0.8))  # [left, bottom, width, height]
    # axcb1 = fig1.add_axes([0.88, 0.4, 0.015, 0.2])
    
    if grid:
        ax1.grid(alpha=0.3,
                 axis='both',
                 color=grid,
                 linestyle='-',
                 which='major'
                 )  # add the grid
    
    im1 = ax1.imshow(img_array, interpolation='none', clim=lims)
    im1.set_cmap(cmap)
    # cb1 = fig1.colorbar(im1, cax=axcb1, ticks=t, format='%.1f')
    
    cbar = colorbar_v4(fig1,
                       ax1,
                       im1,
                       cblab="",
                       fontsize=colorbar_ticks_font_size,
                       ticks=list(t),
                       ticklab=[str(label) for label in t],
                       cbax_options={'location': 'right', 'pad': 0.05, 'shrink': 1.0, 'aspect': 40},
                       cb_options={'orientation': 'vertical', 'extend': 'neither'}
                       )
    
    ax1.set_title(label=title,
                  fontdict={'fontsize': title_font_size},
                  pad=title_padding
                  )
    
    # Set axis details.
    ax1.set_xlabel(label_xaxis,
                   fontsize=axis_labels_font_size,
                   labelpad=5
                   )
    ax1.set_ylabel(label_yaxis,
                   fontsize=axis_labels_font_size,
                   labelpad=5
                   )
    ax1.tick_params(axis='both',
                    which='major',
                    labelsize=axis_major_ticks_font_size
                    )
    ax1.tick_params(axis='both',
                    which='minor',
                    labelsize=axis_minor_ticks_font_size
                    )
    
    # fig1.tight_layout()
    fig1.savefig(dpi=dpi,
                 fname=output_data_folder / output_file_name
                 )
    plt.close('all')
    
    if debug_verbosity >= DEBUG_VERBOSITY_LEVEL_1:
        print(f'continuous_plotter(): Plotting complete.')
    
    return


def false_colour_composite_plotter(axis_labels_font_size: int,
                                   axis_major_ticks_font_size: int,
                                   axis_minor_ticks_font_size: int,
                                   clim: list | None,
                                   colorbar_ticks_font_size: int,
                                   cpalette: str,
                                   debug_verbosity: int,
                                   dpi: int,
                                   figsize: tuple[float, float],
                                   grid: str | None,
                                   img_3d_stack: np.ndarray,
                                   label_xaxis: str | None,
                                   label_yaxis: str | None,
                                   output_data_folder: Path,
                                   output_file_name: Path,
                                   title: str | None,
                                   title_font_size: float,
                                   title_padding: float
                                   ):
    """
    Plot a 3-band false colour composite image.

    :param axis_minor_ticks_font_size: An int which specifies the axis minor ticks font size.
    :param axis_major_ticks_font_size: An int which specifies the axis major ticks font size.
    :param axis_labels_font_size: An int which specifies the axis labels font size.
    :param clim: Optional two element list, float. If set, creates a contrast stretch
      between these two values. Default = None.
    :param colorbar_ticks_font_size: An int which specifies the colorbar ticks font size.
    :param cpalette: A string to specify the name to use for the continuous colour palette (i.e. 'jet').
    :param debug_verbosity: An integer that defines the verbosity of debugging and status console messages.
    :param dpi: An integer that specifies the dots per inch (resolution) to use for the plot.
    :param figsize: A tuple of two integers to specify the (width, height) of the figure, in inches.
    :param grid: An optional string; if present, add grid lines of this colour.
    :param img_3d_stack: A 3D numpy array of raw data to plot.
    :param label_xaxis: A string label to be used for the x-axis of the plot.
    :param label_yaxis: A string label to be used for the y-axis of the plot.
    :param output_data_folder: A Path object representing the folder location to place the output plot image.
    :param output_file_name: A Path object representing the file name to use for the output plot image.
    :param title: A string to use for the title of the plot.
    :param title_font_size: A float which specifies the plot axis title font size.
    :param title_padding: A float which specifies the amount of padding to apply between the plot axis title
      and the plot itself, in units of "points".
    """
    
    if debug_verbosity >= DEBUG_VERBOSITY_LEVEL_1:
        print('false_colour_composite_plotter(): Plotting {} . . .'.format(title.split("\n")[0]))
    
    plt.close('all')
    
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_axes((0.07, 0.1, 0.8, 0.8))  # (left, bottom, width, height)
    # axcb1 = fig.add_axes((0.88, 0.4, 0.015, 0.2))
    
    # change default colour maps to plot NaN as grey
    cmap = copy.copy(eval('mpl.cm.' + cpalette))
    cmap.set_bad(color='black', alpha=0.9)
    
    # generate ticks for colour bar at evenly spaced points between min - max
    # if not specified, use array min/max
    if not clim:
        minval = np.nanmin(img_3d_stack)
        maxval = np.nanmax(img_3d_stack)
        lims = [minval, maxval]
    else:
        minval = clim[0]
        maxval = clim[1]
        lims = clim
    
    interval = (maxval - minval) / 4.0
    t = np.arange(minval, maxval, interval)
    t = np.round(np.append(t, maxval),
                 decimals=1
                 )
    
    im1 = ax1.imshow(img_3d_stack, interpolation='none', clim=lims)
    im1.set_cmap(cmap)
    
    if grid:
        ax1.grid(which='major',
                 axis='both',
                 linestyle='-',
                 color=grid,
                 alpha=0.3
                 )  # add the grid
    
    colorbar_v4(fig,
                ax1,
                im1,
                cblab="",
                fontsize=colorbar_ticks_font_size,
                ticks=list(t),
                ticklab=[str(label) for label in t],
                cbax_options={'location': 'right', 'pad': 0.05, 'shrink': 1.0, 'aspect': 40},
                cb_options={'orientation': 'vertical', 'extend': 'neither'}
                )
    
    ax1.set_title(label=title,
                  fontdict={'fontsize': title_font_size},
                  pad=title_padding
                  )
    
    ax1.set_xlabel(label_xaxis,
                   fontsize=axis_labels_font_size,
                   labelpad=5
                   )
    ax1.set_ylabel(label_yaxis,
                   fontsize=axis_labels_font_size,
                   labelpad=5
                   )
    ax1.tick_params(axis='both',
                    which='major',
                    labelsize=axis_major_ticks_font_size
                    )
    ax1.tick_params(axis='both',
                    which='minor',
                    labelsize=axis_minor_ticks_font_size)
    
    fig.savefig(output_data_folder / output_file_name,
                dpi=dpi
                )
    plt.close('all')
    
    if debug_verbosity >= DEBUG_VERBOSITY_LEVEL_1:
        print(f'false_colour_composite_plotter(): Plotting complete.')
    
    return


# =================================================================================
# Script
# =================================================================================

print(f"viirs_l1b_delta_and_3band_plotter.py: Creating LWIR, MWIR, MWIR-LWIR delta, and false colour composite images "
      f"using {input_viirs_data_path} and saving outputs to {output_data_folder} . . .")

# Handle the case where input_viirs_data_path is a file.
if input_viirs_data_path.is_file():
    
    # Place the Path file in a list.
    input_viirs_l1b_vnp02img_list = [input_viirs_data_path]

# Handle the case where input_viirs_data_path is a directory.
elif input_viirs_data_path.is_dir():
    
    # Grab all the VIIRS L1B VNP02IMG files from the given directory.
    input_viirs_l1b_vnp02img_list = list(input_viirs_data_path.rglob(r'VNP02IMG.A*.nc'))

else:
    raise ValueError("viirs_l1b_delta_and_3band_plotter.py: The input_viirs_data_path parameter is neither a "
                     "path nor a file; check and try again."
                     )

# Enumerate through the input file list.
for count, input_viirs_l1b_vnp02img_file in enumerate(input_viirs_l1b_vnp02img_list, start=1):
    
    print(f'viirs_l1b_delta_and_3band_plotter.py: Now processing {input_viirs_l1b_vnp02img_file.name}, '
          f'file #{count} of {len(input_viirs_l1b_vnp02img_list)}. . .')
    
    # Open the VIIRS L1B VNP02IMG file.
    viirs_l1b_vnp02img_nc = nC.Dataset(input_viirs_l1b_vnp02img_file)

    # Pull the I04 and I05 band packed integers out into numpy grids; crop if desired.
    # I04.
    viirs_l1b_mwir_rad_grid = \
        viirs_l1b_vnp02img_nc['observation_data']['I04'][:, SAMPLE_CROP_INDICES_MIN:SAMPLE_CROP_INDICES_MAX].data

    # I05.
    viirs_l1b_lwir_rad_grid = \
        viirs_l1b_vnp02img_nc['observation_data']['I05'][:, SAMPLE_CROP_INDICES_MIN:SAMPLE_CROP_INDICES_MAX].data
    
    # Pull the I04 and I05 BT look-up tables.
    # I04 LUT.
    viirs_l1b_mwir_lut = viirs_l1b_vnp02img_nc['observation_data']['I04_brightness_temperature_lut'][:]

    # I05 LUT.
    viirs_l1b_lwir_lut = viirs_l1b_vnp02img_nc['observation_data']['I05_brightness_temperature_lut'][:]
    
    # Apply the I04 and I05 LUTs to the I04 and I05 radiance grids to get their corresponding BT grids.
    # I04 BT.
    viirs_l1b_mwir_bt_grid = viirs_l1b_mwir_lut[viirs_l1b_mwir_rad_grid.astype(np.uint16)]
    viirs_l1b_mwir_bt_grid[viirs_l1b_mwir_bt_grid < 0.0] = np.nan
    
    # I05 BT.
    viirs_l1b_lwir_bt_grid = viirs_l1b_lwir_lut[viirs_l1b_lwir_rad_grid.astype(np.uint16)]
    viirs_l1b_lwir_bt_grid[viirs_l1b_lwir_bt_grid < 0.0] = np.nan
    
    # Calculate the MWIR BT minus LWIR BT grid (the delta grid).
    viirs_mwir_lwir_delta_grid = viirs_l1b_mwir_bt_grid - viirs_l1b_lwir_bt_grid
    
    # Plot the LWIR grid.
    continuous_plotter(axis_labels_font_size=12,
                       axis_major_ticks_font_size=12,
                       axis_minor_ticks_font_size=12,
                       clim=None,
                       colorbar_ticks_font_size=12,
                       cpalette='viridis',
                       debug_verbosity=DEBUG_VERBOSITY_LEVEL_1,
                       dpi=DPI,
                       figsize=FIGSIZE,
                       grid=None,
                       img_array=viirs_l1b_lwir_rad_grid,
                       label_xaxis='Area of interest (x)',
                       label_yaxis='Area of interest (y)',
                       output_data_folder=output_data_folder,
                       output_file_name=Path(input_viirs_l1b_vnp02img_file.stem +
                                             '_lwir.png'
                                             ),
                       title=f'VIIRS I04 (LWIR) Grid\n'
                             f'{input_viirs_l1b_vnp02img_file.name}',
                       title_font_size=TITLE_FONT_SIZE,
                       title_padding=TITLE_PADDING
                       )
    
    # Plot the MWIR grid.
    continuous_plotter(axis_labels_font_size=12,
                       axis_major_ticks_font_size=12,
                       axis_minor_ticks_font_size=12,
                       clim=None,
                       colorbar_ticks_font_size=12,
                       cpalette='viridis',
                       debug_verbosity=DEBUG_VERBOSITY_LEVEL_1,
                       dpi=DPI,
                       figsize=FIGSIZE,
                       grid=None,
                       img_array=viirs_l1b_mwir_rad_grid,
                       label_xaxis='Area of interest (x)',
                       label_yaxis='Area of interest (y)',
                       output_data_folder=output_data_folder,
                       output_file_name=Path(input_viirs_l1b_vnp02img_file.stem +
                                             '_mwir.png'
                                             ),
                       title=f'VIIRS I04 (MWIR) Grid\n'
                             f'{input_viirs_l1b_vnp02img_file.name}',
                       title_font_size=TITLE_FONT_SIZE,
                       title_padding=TITLE_PADDING
                       )
    
    # Plot the MWIR-LWIR delta grid.
    continuous_plotter(axis_labels_font_size=12,
                       axis_major_ticks_font_size=12,
                       axis_minor_ticks_font_size=12,
                       clim=None,
                       colorbar_ticks_font_size=12,
                       cpalette='viridis',
                       debug_verbosity=DEBUG_VERBOSITY_LEVEL_1,
                       dpi=DPI,
                       figsize=FIGSIZE,
                       grid=None,
                       img_array=viirs_mwir_lwir_delta_grid,
                       label_xaxis='Area of interest (x)',
                       label_yaxis='Area of interest (y)',
                       output_data_folder=output_data_folder,
                       output_file_name=Path(input_viirs_l1b_vnp02img_file.stem +
                                             '_mwir_lwir_delta.png'
                                             ),
                       title=f'VIIRS I04-I05 (MWIR-LWIR) Delta Grid\n'
                             f'{input_viirs_l1b_vnp02img_file.name}',
                       title_font_size=TITLE_FONT_SIZE,
                       title_padding=TITLE_PADDING
                       )
    
    # If this is a Day image, pull the I01, I02 and I03 band data out into numpy grids; crop if desired.
    # I01.
    if viirs_l1b_vnp02img_nc.DayNightFlag == 'Day':
        
        # I01.
        viirs_l1b_vis_grid = viirs_l1b_vnp02img_nc['observation_data']['I01'][:,
                                                                              SAMPLE_CROP_INDICES_MIN:
                                                                              SAMPLE_CROP_INDICES_MAX
                                                                              ].data
        
        # I02.
        viirs_l1b_nir_grid = viirs_l1b_vnp02img_nc['observation_data']['I02'][:,
                                                                              SAMPLE_CROP_INDICES_MIN:
                                                                              SAMPLE_CROP_INDICES_MAX
                                                                              ].data
    
        # I03.
        viirs_l1b_swir_grid = viirs_l1b_vnp02img_nc['observation_data']['I03'][:,
                                                                               SAMPLE_CROP_INDICES_MIN:
                                                                               SAMPLE_CROP_INDICES_MAX
                                                                               ].data
        
        # Truncate all reflectance values > 1.0 for all three bands.
        viirs_l1b_vis_grid[viirs_l1b_vis_grid > 1.0] = 1.0
        viirs_l1b_nir_grid[viirs_l1b_nir_grid > 1.0] = 1.0
        viirs_l1b_swir_grid[viirs_l1b_swir_grid > 1.0] = 1.0
        
        # Helper function to scale FCC grid floating point values to [0 ... 255] and set the dtype to uint8.
        def scale_grid_rgb(grid):
            return ((grid - np.nanmin(grid)) *
                    (1 / (np.nanmax(grid) - np.nanmin(grid)) * RGB_MAX)
                    ).astype(np.uint8)
        
        # Apply the helper function to our three FCC grids.
        viirs_l1b_swir_grid = scale_grid_rgb(viirs_l1b_swir_grid)
        viirs_l1b_nir_grid = scale_grid_rgb(viirs_l1b_nir_grid)
        viirs_l1b_vis_grid = scale_grid_rgb(viirs_l1b_vis_grid)

        # Apply histogram equalization to each band of the FCC stack.
        viirs_l1b_swir_grid = cv2.equalizeHist(viirs_l1b_swir_grid)
        viirs_l1b_nir_grid = cv2.equalizeHist(viirs_l1b_nir_grid)
        viirs_l1b_vis_grid = cv2.equalizeHist(viirs_l1b_vis_grid)
        
        # Create False Colour Composite (FCC) 3D stack of SWIR, NIR and VIS.
        viirs_l1b_fcc_stack = np.stack(arrays=[viirs_l1b_swir_grid,
                                               viirs_l1b_nir_grid,
                                               viirs_l1b_vis_grid
                                               ],
                                       axis=2
                                       )
        
        # Plot the 3D stack.
        false_colour_composite_plotter(axis_labels_font_size=12,
                                       axis_major_ticks_font_size=12,
                                       axis_minor_ticks_font_size=12,
                                       clim=None,
                                       colorbar_ticks_font_size=12,
                                       cpalette='viridis',
                                       debug_verbosity=DEBUG_VERBOSITY_LEVEL_1,
                                       dpi=DPI,
                                       figsize=FIGSIZE,
                                       grid=None,
                                       img_3d_stack=viirs_l1b_fcc_stack,
                                       label_xaxis='Area of interest (x)',
                                       label_yaxis='Area of interest (y)',
                                       output_data_folder=output_data_folder,
                                       output_file_name=Path(input_viirs_l1b_vnp02img_file.stem +
                                                             '_fcc_stack.png'
                                                             ),
                                       title=f'VIIRS I03, I02 & I01 (SWIR, NIR & VIS) '
                                             f'False Colour Composite\n'
                                             f'{input_viirs_l1b_vnp02img_file.name}',
                                       title_font_size=TITLE_FONT_SIZE,
                                       title_padding=TITLE_PADDING
                                       )
    
    # Close the NetCDF handle when we are done with it.
    viirs_l1b_vnp02img_nc.close()

print(f'viirs_l1b_delta_and_3band_plotter.py: Script execution complete.')
exit(0)
