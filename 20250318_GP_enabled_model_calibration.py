#---------------------------------------------------------------------------------------------------------------------------------------
# ***Background information***
#-----------------------------------------------------------------------------------------------------------------------------------------
# Article title:    Gaussian Processes enabled model calibration in the context of deep geological disposal
# Authors:          Lennart Paul (lennart.paul@tu-braunschweig.de), Jorge-Humberto Urrea-Quintero, Umer Fiaz, 
#                   Ali Hussein, Hazem Yaghi, Joachim Stahlmann, Ulrich Römer and Henning Wessels
# Journal:          Data-Centric Engineering by Cambridge University Press
# Keywords:         Deep geological disposal, Salt mechanics, Gaussian Processes, Sensitivity analysis, Calibration

# Acknowledgments:              We are grateful for the provision and preparation of monitoring data from the 
#                               Federal Company for Radioactive Waste Disposal (BGE mbH).

# Funding Statement:            This research is funded by the German Federal Ministry for the Environment, 
#                               Nature Conservation, NuclearSafety and Consumer Protection (BMUV) and managed 
#                               by project management agency Karlsruhe (PTKA) under grant number 02E12102.

# Data Availability Statement:  All source code used for the presented benchmark studies will be published at zenodo.org upon
#                               acceptance of this manuscript. The data that support the findings of this study are available 
#                               from the Federal Company for Radioactive Waste Disposal (BGE mbH). Restrictions apply to the 
#                               availability of these data, which were used under licence for this study. Data are available 
#                               from the authors with the permission of the Federal Company for Radioactive Waste Disposal (BGE mbH).

# IMPORTANT NOTE: Monitoring data are only available from the authors with the permission of the Federal Company for Radioactive Waste Disposal (BGE mbH)


#%%---------------------------------------------------------------------------------------------------------------------------------------
# ***Initial configurations***
#-----------------------------------------------------------------------------------------------------------------------------------------
# Import of libraries
from scipy.optimize import differential_evolution
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib import font_manager
from SALib.analyze import sobol
from SALib.sample import saltelli
from sklearn.gaussian_process.kernels import WhiteKernel, Matern, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, ExpSineSquared
from scipy.optimize import basinhopping
from scipy.interpolate import PchipInterpolator
import pandas as pd
import numpy as np
import time

# Percentage of training data in the total data set
percentage_train = 0.75
# Setting the nominal values of the considered parameters
X_nominal = np.array([80000*86400, 30, 0.5, 75, 350000000*86400, 30, 1.5])
# Amount of model calls for convergence of sensitivity analysis
N = 2**14
# Weighting between vertical and horizontal convergence for the optimization
weights = [1.0, 1.0]
# Colors for plotting
color1 = cm.Blues(0.5)
color2 = cm.Blues(0.9)
color3 = cm.Reds(0.9)

#%%---------------------------------------------------------------------------------------------------------------------------------------
# ***Definition of input parameters and import of sampled values for each simulation***
#-----------------------------------------------------------------------------------------------------------------------------------------
# TUBSsalt parameters
p1 = 'etap'  # Viscosity of primary creep
p2 = 'seq0p'  # Start of the slope change
p3 = 'pp'  # Curvature parameter
p4 = 'ep'  # Hardening modulus
p5 = 'etas'  # Viscosity of secondary creep
p6 = 'seq0s'  # Start of the slope change
p7 = 'ps'  # Curvature parameter
# Sets the considered parameters and their range of values
param_list = [p1, p2, p3, p4, p5, p6, p7]
symbol_list = ['$\eta_p$', '$\sigma_{eq,0,p}$', '$p_p$', '$E_p$',
               '$\eta_s$', '$\sigma_{eq,0,s}$', '$p_s$']
# Dictionary with parameter ranges, that where used in FLAC3D simulations
param_dict = {'etap': [5.0e4 * 86400, 60.0e4 * 86400], 'seq0p': [20, 40], 'pp': [0.3, 1.0], 'ep': [60, 90],
              'etas': [5.0e7 * 86400, 50.0e7 * 86400], 'seq0s': [20, 40], 'ps': [0.5, 2.0]}
# Name od the folder where the simulation data is stored
foldername = "flac3d_simulation_data"
# Import of sampled parameter values 
data_f3_para = pd.read_csv(foldername+r'/VD_Gorleben2_Params.txt', sep=', ')
sets = len(data_f3_para)
sets_train = int(sets*percentage_train)

#%%---------------------------------------------------------------------------------------------------------------------------------------
# ***Import of the monitoring data***
#-----------------------------------------------------------------------------------------------------------------------------------------
#***IMPORTANT NOTE: Monitoring data are available from the authors with the permission of the Federal Company for Radioactive Waste Disposal (BGE mbH)***

# Saving monitoring data from excel file in a DataFrame
data_moni = pd.read_excel("Export_MQ 62.xlsx", sheet_name="62_40")
# Determination of time difference between excavation and first meassurement
excavation = time.mktime(time.strptime(str(data_moni.iloc[6, 4]), '%d.%m.%Y'))
start_moni = time.mktime(time.strptime(str(data_moni.iloc[6, 5]), '%d.%m.%Y'))
diff_time = (start_moni-excavation)/(60*60*24)  # diff_time = 1
# Deleting the first 10 rows of the excel sheets to extract the monitoring data
data_moni.drop(index=data_moni.index[0:11], inplace=True)
data_moni.reset_index(drop=True, inplace=True)
# Renames the columns and set the data types for each column
data_moni.rename(columns={'BGE': 'date', 'Gorleben': 'time', 'Unnamed: 2': 'days',
                            'Unnamed: 3': 'K_h [mm]', 'Unnamed: 4': 'DeltaK_h [mm]',
                            'Unnamed: 6': 'K_v [mm]', 'Unnamed: 7': 'DeltaK_v [mm]', }, inplace=True)
data_moni = data_moni.astype({'days': float, 'K_h [mm]': float, 'DeltaK_h [mm]': float, 'K_v [mm]': float, 'DeltaK_v [mm]': float})
# Applys linear interpolation for missing data points of the convergence meassurements
data_moni = data_moni.interpolate(method='linear')
data_moni["years"] = data_moni["days"]/365
time_values_moni = data_moni["years"].tolist()
# Calculates the corrected days and years from the time difference
data_moni["days_corr"] = data_moni["days"] + diff_time


#%%---------------------------------------------------------------------------------------------------------------------------------------
# ***Import of simulation data***
#-----------------------------------------------------------------------------------------------------------------------------------------
# Import of the simulation data from FLAC3D and saved as convergence data in a dictionary
f3_dict = {}
for ind in range(1, sets+1, 1):
    # Saving the simulations data in a Dataframe
    data_f3_vd = pd.read_csv(foldername+r'/VD_Gorleben2_{}.txt'.format(ind), sep=' ')
    data_f3_vd.drop(index=[data_f3_vd.index[0]], inplace=True)
    data_f3_vd.dropna(axis=1, how='all', inplace=True)
    data_f3_vd.rename(columns={'Unnamed: 1': 'steps', 'Unnamed: 3': 'days',
                               'Unnamed: 4': 'disp_z_top', 'Unnamed: 6': 'disp_z_bot', 'Unnamed: 7': 'disp_x_wall'}, inplace=True)
    data_f3_vd = data_f3_vd.astype({'steps': float, 'days': float, 'disp_z_top': float})
    data_f3_vd['years'] = data_f3_vd['days']/365
    # Saving the parameter values for the corresponding simulation as a list
    values = []
    for para in param_list:
        # data_f3_para starts with index 0!!!
        values.append(data_f3_para.loc[ind-1, para])
    # Transformes the displacements into the horizontal and vertical convergence
    data_f3_vd['K_h [mm]'] = (data_f3_vd['disp_x_wall'] + data_f3_vd['disp_x_wall']) * 1000
    data_f3_vd['K_v [mm]'] = (data_f3_vd['disp_z_top'] - data_f3_vd['disp_z_bot']) * 1000
    # Generates cubic spline interpolation function for simulation convergences
    spline_K_v_sim = PchipInterpolator(data_f3_vd['days'], data_f3_vd['K_v [mm]'])
    spline_K_h_sim = PchipInterpolator(data_f3_vd['days'], data_f3_vd['K_h [mm]'])
    # Determination of the simulation convergence where the monitoring starts (day 1)
    conv_v_index0 = spline_K_v_sim(diff_time)
    conv_h_index0 = spline_K_h_sim(diff_time)
    # Creates the dataframe data_sim and store days, years and monitoring data in it
    data_sim = pd.DataFrame()
    # Uses 80 timepoints from monitoring data
    data_sim["days"] = data_moni["days_corr"]
    data_sim["K_v_moni [mm]"] = data_moni["K_v [mm]"]
    data_sim["K_h_moni [mm]"] = data_moni["K_h [mm]"]
    datapoints = len(data_sim)
    # Determines the corresponding convergence value for the specific amount of days in consideration of the start correction
    for j in range(datapoints):
        index_days = data_sim.loc[j, "days"]
        data_sim.at[j, "K_v [mm]"] = spline_K_v_sim(index_days) - conv_v_index0
        data_sim.at[j, "K_h [mm]"] = spline_K_h_sim(index_days) - conv_h_index0
    data_sim['days'] = data_sim['days'] - diff_time
    data_sim["years"] = data_sim["days"]/365
    # Saves the convergence data to the dictionary f3_dict
    f3_dict[ind] = [data_sim, values]
    print(ind)

# List of features in f3_dict
features = ['K_v [mm]', 'K_h [mm]']
features_dict = {'K_v [mm]': 'Vertical Convergence', 'K_h [mm]': 'Horizontal Convergence'}
# Increased figsize and dpi for better resolution
fig, axes = plt.subplots(1, len(features), figsize=(16, 6), dpi=300)
for j, feature in enumerate(features):
    ax = axes[j]
    # Plotting the horizontal convergencies of the simulation data
    for i, key in enumerate(f3_dict):
        ax.plot(f3_dict[key][0]['years'], f3_dict[key][0][feature],
                color=color2, linewidth=0.5, alpha=0.8, 
                label=str(sets)+' FLAC3D simulation samples' if i==0 else None)
    # Plot the convergences of the monitoring data
    ax.plot(data_moni['years'], data_moni[feature], 's', markersize=6,
            color='black', alpha=0.9, label='Monitoring data')
    # Corrected the indexing here
    ax.set_title(f'{features_dict[feature]}', fontsize=17, fontweight='semibold', pad=10)
    ax.set_xlabel(r'Time [years]', fontsize=16)
    if j == 0:
        ax.set_ylabel(r'Convergence [mm]', fontsize=16, labelpad=10)
    # Adjusting the tick font size
    ax.tick_params(axis='both', labelsize=15)
    # Set the axes limits
    ax.set_xlim([-0.5, max(data_moni['years'])+0.5])
    ax.set_ylim([-210, 5])
    ax.yaxis.set_inverted(True)
    # Set the number of ticks for the x and y axes
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.grid(True)
    ax.legend(loc='lower right', fontsize=16)
# Saving the Plot
plt.tight_layout(h_pad=2)
plt.savefig(fname=r'Fig_Simulation samples.pdf', format='pdf', dpi=400)
plt.show()

#%%---------------------------------------------------------------------------------------------------------------------------------------
# ***Processing of simulation data***
#-----------------------------------------------------------------------------------------------------------------------------------------
X_dict = {var: [] for var in param_list}
Y_Horizontal_converg, Y_Vertical_converg = [], []
time_values_all = []  # List to hold all time vectors

for key in f3_dict:
    # Extract the desired Y values from the dataframe
    # data obtain from data frame data_sim
    Horizontal_converg = f3_dict[key][0]['K_h [mm]'].tolist()
    Vertical_converg = f3_dict[key][0]['K_v [mm]'].tolist()
    # Append Y values to respective lists
    Y_Horizontal_converg.append(Horizontal_converg)
    Y_Vertical_converg.append(Vertical_converg)
    # Extract X values and append to X_dict
    values = f3_dict[key][1]
    for i, var in enumerate(param_list):
        X_dict[var].append(values[i])
    # Extract and store time vector
    time_values_all.append(f3_dict[key][0]['years'].tolist())

# Create DataFrame for X_combined
X_combined = pd.DataFrame(X_dict)
Y_combined = []
for Horizontal_converg, Vertical_converg, time_values in zip(Y_Horizontal_converg, Y_Vertical_converg, time_values_all):
    # Ensure lists are of the same length
    min_length = min(len(Horizontal_converg), len(Vertical_converg), len(time_values))
    Horizontal_converg = Horizontal_converg[:min_length]
    Vertical_converg = Vertical_converg[:min_length]
    time_values_trimmed = time_values[:min_length]

    combined_df = pd.DataFrame({'Vertical_converg': Vertical_converg,
                                'Horizontal_converg': Horizontal_converg,
                                'years': time_values_trimmed})
    Y_combined.append(combined_df)

# Determines the time intervals for the original time points
time_values_difference = [time_values_moni[i] - time_values_moni[i-1] for i in range(1, len(time_values_moni))]
# Indices for splitting the monitoring data under consideration of different areas of time intrevals
indices_split = [33, 54]
# Reduced density values for 3 areas
density_timepoints = [11, 2, 1]

# Splits the original 80 datapoints into 3 areas and removes datapoints according to 'density_timepoints'
time_values_moni_1 = (time_values_moni[1:indices_split[0]])                 [::density_timepoints[0]]
time_values_moni_2 = (time_values_moni[indices_split[0]:indices_split[1]])  [::density_timepoints[1]]
time_values_moni_3 = (time_values_moni[indices_split[1]:])                  [::density_timepoints[2]]

# Saves the reduced amount of timepoints in the list 'target_time_values'
target_time_values = sorted(time_values_moni_1 + time_values_moni_2 + time_values_moni_3)
# Determines the new time intervals for the reduced time points
time_values_difference_new = [target_time_values[i] - target_time_values[i-1] for i in range(1, len(target_time_values))]
# Determines the remaining convergence data for the corresponding time in 'target_time_values'
remaining_monitoring_points = [data_moni.loc[data_moni.index[data_moni['years'] == time]]['K_v [mm]'] for time in target_time_values]


# Plots the original density of monitoring points vs the reduced density together with the correspinding time intervals between monitoring points
plot_titles = ['Original', 'Selected']
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

for i, title in enumerate(plot_titles):
    # Plots the time intervals between monitoring points
    axes[0,i].set_title(title+' time intervals of monitoring data', pad=15, fontsize=18, fontweight='bold')
    if i == 0:
        axes[0,i].plot(list(range(1,len(time_values_difference)+1,1)), time_values_difference, linestyle='-', linewidth=1, marker='o', markersize=6, color=color2)
        axes[0,i].plot([indices_split[0]-0.5, indices_split[0]-0.5], [-1, 1], linewidth=1, color=color3)
        axes[0,i].plot([indices_split[1]-0.5, indices_split[1]-0.5], [-1, 1], linewidth=1, color=color3)
        axes[0,i].set_ylabel('Years [a]', fontsize=16, labelpad=10)
    else:
        axes[0,i].plot(list(range(1,len(time_values_difference_new)+1,1)), time_values_difference_new, linestyle='-', linewidth=1, marker='o', markersize=6, color=color2)
        axes[0,i].plot([len(time_values_moni_1)+0.5, len(time_values_moni_1)+0.5], [-1, 1], linewidth=1, color=color3)
        axes[0,i].plot([len(time_values_moni_1+time_values_moni_2)-0.5, len(time_values_moni_1+time_values_moni_2)-0.5], [-1, 1], linewidth=1, color=color3)
    axes[0,i].set_xlabel('Index', fontsize=16)
    #axes[0,i].set_xticks(list(range(0,len(time_values_difference)+2,5)))
    axes[0,i].set_ylim(-0.03, 0.7)
    axes[0,i].grid()
    # Plots the monitoring points over time
    if i == 0:
        axes[1,i].plot(time_values_moni, data_moni['K_v [mm]'], 's', markersize=6, color='black', alpha=0.9, label='Monitoring Data')
        axes[1,i].set_ylabel('Vertical convergence [mm]', fontsize=16, labelpad=10)
    else:
        axes[1,i].plot(target_time_values, remaining_monitoring_points, 's', markersize=6, color='black', alpha=0.9, label='Monitoring Data')
    axes[1,i].plot([data_moni.loc[indices_split[0]]['years'], data_moni.loc[indices_split[0]]['years']], [-150, 10], linewidth=1, color=color3)
    axes[1,i].plot([data_moni.loc[indices_split[1]]['years'], data_moni.loc[indices_split[1]]['years']], [-150, 10], linewidth=1, color=color3)
    axes[1,i].set_xlabel('Years [a]', fontsize=16)
    axes[1,i].set_ylim([-125, 5])
    axes[1,i].yaxis.set_inverted(True)
    axes[1,i].grid()
plt.tight_layout(h_pad=2)
#plt.savefig(foldername+"_Selected datapoints.png", dpi=400)
plt.show()


# Saves indices for the years in 'target_time_values' in list 'list_indices'
list_indices = []
for times in target_time_values:
    for index in range(0, len(data_moni)):
        if data_moni.loc[index]['years'] == times:
            list_indices.append(index)
# Reduces the dataframe 'Y_combined' to the datapoints in 'target_time_values' and saves them in 'Y_selected'
Y_selected = [] 
for df in Y_combined:
    selected_df = df
    selected_df = selected_df[selected_df.index.isin(list_indices)]
    Y_selected.append(selected_df)

# Normalize data to be between 0 and 1 using provided min and max values.
def normalize_data(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)

# Convert X_combined DataFrame to a numpy stack of arrays
X_combined_np = np.stack(X_combined.values)
num_samples = X_combined_np.shape[0]

# Convert Y_selected list of DataFrames to a numpy stack of arrays for each attribute
Horizontal_converg_stack = np.stack([df['Horizontal_converg'].values for df in Y_selected])
Vertical_converg_stack = np.stack([df['Vertical_converg'].values for df in Y_selected])

# Set a seed for reproducibility
np.random.seed(24)

# Find indices of max and min final displacement values for each attribute
max_disp_indices = [np.argmax(attr[:, -1]) for attr in [Vertical_converg_stack, Horizontal_converg_stack]]
min_disp_indices = [np.argmin(attr[:, -1]) for attr in [Vertical_converg_stack, Horizontal_converg_stack]]

# Ensure unique indices and add them to the training set
unique_indices = list(set(max_disp_indices + min_disp_indices))

# Shuffle the remaining indices
remaining_indices = [i for i in range(num_samples) if i not in unique_indices]
np.random.shuffle(remaining_indices)

# Determine the split point based on the desired proportion
split_point = int(percentage_train * num_samples) - len(unique_indices)

# Combine unique indices with the first part of the shuffled indices for training
train_indices = unique_indices + remaining_indices[:split_point]
test_indices = remaining_indices[split_point:]

X_train = X_combined_np[train_indices]
X_test = X_combined_np[test_indices]

# Normalize the X data
min_X = np.min(X_combined_np, axis=0)
max_X = np.max(X_combined_np, axis=0)
print('min X', min_X, 'max X', max_X)

X_train = normalize_data(X_train, min_X, max_X)
X_test = normalize_data(X_test, min_X, max_X)

#X_nominal = np.array([100000*86400, 25, 0.5, 75, 360000000*86400, 25, 1.5])
X_nominal_norm = normalize_data(X_nominal, min_X, max_X)
print('Nominal X = ', X_nominal_norm)

# Split and normalize the Y data for each attribute
attributes = [Vertical_converg_stack, Horizontal_converg_stack]
Y_train = []
Y_test = []
min_Y = np.min([np.min(attr) for attr in attributes])
max_Y = np.max([np.max(attr) for attr in attributes])

for attr_stack in attributes:
    Y_train_attr = normalize_data(attr_stack[train_indices], min_Y, max_Y)
    Y_test_attr = normalize_data(attr_stack[test_indices], min_Y, max_Y)
    Y_train.append(Y_train_attr)
    Y_test.append(Y_test_attr)


# Plots the probability density function of input and output
pdf_color = cm.Reds(0.8)  # Adjusted color for PDFs
# Define plot_2d_projected_pdfs to accept an Axes object
def plot_2d_projected_pdfs(ax, Y_selected, attribute, data_moni, bins=25, remove_first_n=3, time_step=5,
                           fontsize=16, xticks=8, yticks=8, labelpad=15, scale_factor=10):
    def reflect_negative(values):
        return values.apply(lambda x: -x if x < 0 else x)

    time_instances = Y_selected[0]['years'].unique()
    time_instances = time_instances[remove_first_n:]
    selected_time_instances = time_instances[::time_step]
    # Plot PDFs for selected time instances
    for i, time_instance in enumerate(selected_time_instances):
        data_values = []
        for df in Y_selected:
            values_at_time = df[df['years'] == time_instance][attribute]
            data_values.extend(values_at_time.tolist())
        if not data_values:
            continue  # Skip if no data for the time instance
        hist, bin_edges = np.histogram(data_values, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        hist_scaled = hist * scale_factor
        ax.fill_betweenx(bin_centers, time_instance, time_instance + hist_scaled,
                         alpha=0.8, color=pdf_color, edgecolor='k', 
                         label='p.d.f. output data over time' if i==0 else None)
    # Set labels and title
    ax.set_xlabel(r'Time [years]', fontsize=fontsize, labelpad=labelpad)
    ax.set_ylabel(r'Convergence [mm]', fontsize=fontsize, labelpad=labelpad)
    attribute_title = 'Horizontal Convergence' if attribute == 'Horizontal_converg' else 'Vertical Convergence'
    ax.set_title(attribute_title, fontsize=fontsize, fontweight='semibold', pad=15)
    # Customize ticks
    ax.xaxis.set_major_locator(MaxNLocator(nbins=xticks))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=yticks))
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.grid(True)

# Combined plot function
def plot_combined():
    fig = plt.figure(figsize=(16, 12), dpi=300)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2], wspace=0.3)
    # Left Column: Parameter Distributions (Code 3)
    ax_left_title = fig.add_subplot(gs[0])
    ax_left_title.set_title('Histogram Input \n Model Parameters', fontsize=16, fontweight='semibold', pad=10)
    ax_left_title.axis('off')
    gs_left = gridspec.GridSpecFromSubplotSpec(len(param_list), 1, subplot_spec=gs[0], hspace=0.9)
    # Plot each parameter histogram
    for ind, para in enumerate(param_list):
        ax = fig.add_subplot(gs_left[ind])
        list_values = data_f3_para[para]
        ax.hist(list_values, bins=15, color=color1, edgecolor='black')
        ax.set_ylabel('Samples', fontsize=14)
        ax.set_xlabel(symbol_list[ind], fontsize=14)
        ax.tick_params(axis='both', labelsize=14)
        ax.grid(True)

    # Right Column: Time Series and PDFs
    gs_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1], hspace=0.30)
    # --- Subplot 1: Vertical Convergence with PDFs ---
    ax_v = fig.add_subplot(gs_right[0])
    feature_v = 'K_v [mm]'
    for i, key in enumerate(f3_dict):
        ax_v.plot(f3_dict[key][0]['years'], f3_dict[key][0][feature_v],
                  color=color2, linewidth=0.5, alpha=0.06, 
                  label=str(sets)+' FLAC3D simulation samples' if i==0 else None)
    ax_v.plot(data_moni['years'], data_moni[feature_v], 's',
              markersize=6, color='black', alpha=0.9, label='Monitoring data')

    # Integrate PDF overlays for Vertical Convergence
    plot_2d_projected_pdfs(ax=ax_v, Y_selected=Y_selected, attribute='Vertical_converg',
                           data_moni=data_moni, bins=10, remove_first_n=2, time_step=4,
                           fontsize=16, xticks=8, yticks=8, labelpad=15, scale_factor=15)
    ax_v.set_title(features_dict[feature_v], fontsize=16, fontweight='semibold', pad=15)
    ax_v.set_xlabel('Time [years]', fontsize=15, labelpad=5)
    ax_v.set_ylabel('Convergence [mm]', fontsize=15, labelpad=15)
    ax_v.tick_params(axis='both', labelsize=14)
    ax_v.set_xlim([-0.5, max(data_moni['years']) + 0.5])
    ax_v.set_ylim([-350, 10])
    ax_v.invert_yaxis()  # Invert y-axis to reflect convergence direction
    ax_v.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax_v.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax_v.legend(loc='upper left', fontsize=14)
    ax_v.grid(True)

    # --- Subplot 2: Horizontal Convergence with PDFs ---
    ax_h = fig.add_subplot(gs_right[1])
    feature_h = 'K_h [mm]'
    for i, key in enumerate(f3_dict):
        ax_h.plot(f3_dict[key][0]['years'], f3_dict[key][0][feature_h],
                  color=color2, linewidth=0.5, alpha=0.06, 
                  label=str(sets)+' FLAC3D simulation samples' if i==0 else None)
    ax_h.plot(data_moni['years'], data_moni[feature_h], 's',
              markersize=6, color='black', alpha=0.9, label='Monitoring data')

    # Integrate PDF overlays for Horizontal Convergence
    plot_2d_projected_pdfs(ax=ax_h, Y_selected=Y_selected, attribute='Horizontal_converg',
                           data_moni=data_moni, bins=10, remove_first_n=2, time_step=4,
                           fontsize=16, xticks=8, yticks=8, labelpad=15, scale_factor=15)
    ax_h.set_title(features_dict[feature_h], fontsize=16, fontweight='semibold', pad=15)
    ax_h.set_xlabel('Time [years]', fontsize=15, labelpad=5)
    ax_h.set_ylabel('Convergence [mm]', fontsize=15, labelpad=15)
    ax_h.tick_params(axis='both', labelsize=14)
    ax_h.set_xlim([-0.5, max(data_moni['years'])+0.5])
    ax_h.set_ylim([-350, 10])
    ax_h.invert_yaxis()  # Invert y-axis to reflect convergence direction
    ax_h.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax_h.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax_h.grid(True)
    plt.tight_layout()
    plt.savefig(fname='Fig_combined_IO_data.pdf', format='pdf', dpi=400)
    plt.show()
# Ensure that required data (f3_dict, data_moni, Y_selected) is available before running
plot_combined()

#%%---------------------------------------------------------------------------------------------------------------------------------------
# ***Training of the GP-based surrogate model***
#-----------------------------------------------------------------------------------------------------------------------------------------
# Determine the duration for creation of the surrogate model
start_time1 = time.time()

# Function to add noise to the data
def add_noise(data, noise_level=0.025):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

# Simpler kernel: RBF kernel with a constant kernel and a small white noise
kernel_RBF = C(1.0, (1e-3, 1e3)) * RBF(length_scale=0.1, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-4, 1e-1))

# Original kernel
# kernel_matern = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)

# - Combine Matern with a WhiteKernel (noise term).
# - Adjust initial values and bounds of kernel hyperparameters for more flexibility.
kernel_matern = (C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.5, length_scale_bounds=(1e-2, 1e2), nu=1.5)  # + #, nu=1.5
        #   WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-10, 1e+1))
          )

# Complex kernel: Combination of RBF, DotProduct (for polynomial features), and ExpSineSquared (for periodic features)
kernel_complex = (C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) +
          C(1.0, (1e-3, 1e3)) * DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-2, 1e2))**2 +
          C(1.0, (1e-3, 1e3)) * ExpSineSquared(length_scale=1.0, periodicity=1.0,
                                              length_scale_bounds=(1e-2, 1e2),
                                              periodicity_bounds=(1e-1, 1e1)) +
          WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-4, 1e-1)))


# Combined kernel (you can modify this as needed)
# kernel_combined = kernel_RBF
kernel_combined = kernel_matern
# kernel_combined = kernel_complex

# Define the optimizer function
def optimizer(obj_func, initial_theta, bounds):
    result = basinhopping(lambda x: obj_func(x)[0], initial_theta, minimizer_kwargs={"method": "L-BFGS-B", "bounds": bounds})
    return result.x, result.fun


# List of features and their corresponding indices in Y_train and Y_test
features = ['Vertical_converg', 'Horizontal_converg']
indices = [0, 1]  # Assuming you have 3 features, adjust if you have more

# Dictionary to store trained GPs for each feature
trained_GPs = {}

# Noise level to be added to the training data
noise_level = 0.0005  # Adjust this value as needed for your specific use case

for feature, idx in zip(features, indices):
    # Extracting data related to the current feature from Y_train and Y_test
    feature_train = Y_train[idx]
    feature_test = Y_test[idx]

    # Add noise to the training data
    feature_train_noisy = feature_train
    # feature_train_noisy = add_noise(feature_train, noise_level=noise_level)

    # Initialize and train the GP for the current feature with noisy data
    gp = GaussianProcessRegressor(kernel=kernel_combined, n_restarts_optimizer=20, optimizer='fmin_l_bfgs_b')
    gp.fit(X_train, feature_train_noisy)
    # Predicting using the trained GP
    YKRGmean_train, YKRGmean_train_std = gp.predict(X_train, return_std=True)
    # Store the trained GP in the dictionary
    trained_GPs[feature] = gp
    # Print the shape of the predictions (or any other relevant information)
    print(f"Feature: {feature}")
    print(YKRGmean_train.shape)
    print("-" * 50)

# Determine the duration for creation of the surrogate model
end_time1 = time.time()
duration_training = (end_time1 - start_time1)
print(f"Training of the surrogate model took {duration_training:.1f} seconds")
print("-" * 50)

#%%---------------------------------------------------------------------------------------------------------------------------------------
# ***Perform accuracy evaluation of the surrogate model using testing data***
#-----------------------------------------------------------------------------------------------------------------------------------------
# Define the components (time points) you want to visualize in list "target_time_values"
components = [2, 6, 14, 26, 35]  # Example components, adjust as needed
# Titles for each feature
titles = ["vertical convergence", "horizontal convergence"]
# Create a 3x5 grid for plotting
fig, axes = plt.subplots(len(features), len(components), figsize=(16, 10), constrained_layout=True)
# Common font size for all text in the plots
common_font_size = 16
# Number of ticks on the x and y axes
num_x_ticks = 3
num_y_ticks = 4

for j, feature in enumerate(features):
    # Extract the true values for the current feature for both training and validation data
    Y_true_train = Y_train[indices[j]]
    Y_true_test = Y_test[indices[j]]
    # Predict using the trained GP for the current feature
    Y_pred_train, _ = trained_GPs[feature].predict(X_train, return_std=True)
    Y_pred_test, _ = trained_GPs[feature].predict(X_test, return_std=True)

    for i, component in enumerate(components):
        ax = axes[j, i]
        # Plot the true vs. predicted values for the current component for both training and validation data
        ax.scatter(Y_true_train[:, component], Y_pred_train[:, component], marker='o',
                   alpha=.8, s=20, color=color2, label='Training' if i == 0 else "")
        ax.scatter(Y_true_test[:, component], Y_pred_test[:, component], marker='o',
                   alpha=.8, s=20, color=color3, label='Validation' if i == 0 else "")
        ax.plot([np.min(Y_true_train[:, component]), np.max(Y_true_train[:, component])],
                [np.min(Y_true_train[:, component]), np.max(Y_true_train[:, component])], 'k')
        # Compute R^2 coefficients
        r2_train = r2_score(Y_true_train[:, component], Y_pred_train[:, component])
        r2_test = r2_score(Y_true_test[:, component], Y_pred_test[:, component])
        # Convert time to years and round
        time_in_years = target_time_values[component]

        if j == 1:
            # Set labels, title, and ticks
            ax.set_xlabel('True output', fontsize=common_font_size)
        # ax.set_title(f'component at t = {time_in_years} years', fontsize=common_font_size)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=num_x_ticks))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=num_y_ticks))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.tick_params(axis='both', labelsize=common_font_size)

        # Print R^2 values inside the subplot
        bbox = dict(boxstyle='round', facecolor='white', edgecolor='lightgrey', alpha=0.8)
        ax.text(0.05, 0.97, f'$R^2$-train: {r2_train:.3f}\n$R^2$-test: {r2_test:.3f}',
                transform=ax.transAxes, fontsize=common_font_size-1, verticalalignment='top', bbox=bbox)
        ax.grid(True)
        # Add component title to the first subplot of each column
        if j == 0:
            ax.set_title(f'component at\nt = {time_in_years:.2f} years', fontsize=common_font_size+1, fontweight='bold', pad=15)
        # Add feature title to the first subplot of each row
        if i == 0:
            ax.set_ylabel(f'Predicted output for\n{titles[j]}', fontsize=common_font_size+1, labelpad=10)
        # Add legend in the first subplot of the first row
        if j == 0 and i == 0:
            font_properties = font_manager.FontProperties(
                size=common_font_size-1)
            ax.legend(loc='lower right', prop=font_properties)
plt.tight_layout()
# Save the figure as a PDF
fig.savefig(fname='Fig_GPs_accuracy_time_'+str(sets_train)+'samples.pdf', format='pdf', bbox_inches='tight', dpi=400)
plt.show()


#%%---------------------------------------------------------------------------------------------------------------------------------------
# ***Perform global sensitivity analysis using Sobol indices***
#-----------------------------------------------------------------------------------------------------------------------------------------
# Determine the duration for performing the sensitivity analysis
start_time2 = time.time()

# 1. Define the problem for SALib
problem = {'num_vars': X_train.shape[1],        # Assuming X_train is defined and corresponds to your training data
           'names': param_list,                 # Names of the parameters
           'bounds': [(0, 1) for _ in range(len(param_list))]} # Bounds for each parameter
# Generate Sobol' samples with N=2**14
param_values = saltelli.sample(problem, N)  # Generate samples
# Dictionary to store Sobol' indices for each feature
S1_dict = {}
ST_dict = {}
# Loop over each feature and its corresponding trained GP
for feature, gp in trained_GPs.items():
    # 3. Use the GP model to predict the outputs for these samples
    Y_GP_predictions = gp.predict(param_values)

    # 4. Compute Sobol' indices for each time point
    T = Y_GP_predictions.shape[1]  # Number of time points
    S1 = np.zeros((T, problem['num_vars']))
    ST = np.zeros((T, problem['num_vars']))

    for i in range(T):
        Si = sobol.analyze(
            problem, Y_GP_predictions[:, i], print_to_console=False)
        S1[i, :] = Si['S1']
        ST[i, :] = Si['ST']

    # Store the computed Sobol' indices in the dictionaries
    S1_dict[feature] = S1
    ST_dict[feature] = ST


# List of features in f3_dict
features = ['Vertical_converg', 'Horizontal_converg']
# Custom titles for each column
titles = ["Vertical convergence", "Horizontal convergence"]
# Common font size for all text in the plots
common_font_size = 16
# Number of ticks on the x and y axes
num_x_ticks = 8
num_y_ticks = 2
# Assuming you've already defined and populated the following from your previous code:
# param_values, problem, S1_dict, ST_dict, target_time_values
T_years = target_time_values  # Replace max_time with the actual maximum time value
# Corrected plotting code to handle dimension mismatch
fig, axes = plt.subplots(len(param_list), len(
    features), figsize=(16, 12), sharex=True)

if len(features) == 1 or len(param_list) == 1:
    axes = np.array([[axes]])

for i, feature in enumerate(features):
    for j, name in enumerate(symbol_list):
        ax = axes[j, i]
        if len(T_years) == len(S1_dict[feature][:, j]) and len(T_years) == len(ST_dict[feature][:, j]):
            ax.plot(sorted(target_time_values), S1_dict[feature][:, j].reshape(
                -1, 1), '-', color=color1, label='First-order Sobol Index' if i == 0 and j == 0 else "")
            ax.plot(sorted(target_time_values), ST_dict[feature][:, j].reshape(
                -1, 1), '--', color=color2, label='Total-order Sobol Index' if i == 0 and j == 0 else "")
        else:
            print(f"Dimension mismatch for feature '{feature}' and parameter '{name}'")
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.2, max(T_years)+0.2)

        if j == 0:
            ax.set_title(titles[i], fontsize=common_font_size+1, fontweight='bold', pad=15)
        if i == 0:
            ax.set_ylabel(name, fontsize=common_font_size+1, rotation='horizontal', labelpad=35)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=num_x_ticks))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=num_y_ticks))
        ax.grid(True)

        if i == 0 and j == 0:
            font_properties = font_manager.FontProperties(size=common_font_size-1)
            ax.legend(loc='upper right', prop=font_properties)

        # Adjusting the tick font size
        ax.tick_params(axis='both', labelsize=common_font_size-1)
    ax.set_xlabel('Time [years]', fontsize=common_font_size)
plt.tight_layout(h_pad=2, w_pad=2)
plt.savefig(fname='Fig_Sobol_indices_time.pdf', format='pdf', dpi=400)
plt.show()


# Initialize dictionaries to store the calculated indices for each feature
time_averaged_S1 = {}
time_averaged_ST = {}
maximum_S1 = {}
maximum_ST = {}
cumulated_S1 = {}
cumulated_ST = {}
# Loop over each feature to calculate the indices
for feature in trained_GPs.keys():
    # Remove the indices for the first two time instances to avoid initial condition of sensitivity near t=0
    # Time-averaged Sobol' indices
    time_averaged_S1[feature] = np.mean(S1_dict[feature][:], axis=0)
    time_averaged_ST[feature] = np.mean(ST_dict[feature][:], axis=0)
    # Maximum Sobol' indices
    maximum_S1[feature] = np.max(S1_dict[feature][:], axis=0)
    maximum_ST[feature] = np.max(ST_dict[feature][:], axis=0)
    # Integrated Sobol' indices (assuming equal time step size)
    cumulated_S1[feature] = np.sum(S1_dict[feature][:], axis=0)
    cumulated_ST[feature] = np.sum(ST_dict[feature][:], axis=0)

print(S1_dict['Vertical_converg'][2:])

# Print the results for each feature
for feature in trained_GPs.keys():
    print(f"Feature: {feature}")
    print("Time-averaged first-order Sobol' indices:",
          time_averaged_S1[feature])
    print("Time-averaged total Sobol' indices:", time_averaged_ST[feature])
    print("Maximum first-order Sobol' indices:", maximum_S1[feature])
    print("Maximum total Sobol' indices:", maximum_ST[feature])
    print("Integrated first-order Sobol' indices:", cumulated_S1[feature])
    print("Integrated total Sobol' indices:", cumulated_ST[feature])
    print("-" * 50)

# Determine the duration for performing the sensitivity analysis
end_time2 = time.time()
duration_sensitivity = (end_time2 - start_time2)
print(
    f"Performing the sensitivity analysis took {duration_sensitivity:.1f} seconds")
print("-" * 50)


# Function to normalize the integrated Sobol' indices
def normalize_cumulated_indices(cumulated_indices_dict, max_value):
    normalized_dict = {}
    for feature, indices in cumulated_indices_dict.items():
        normalized_dict[feature] = indices / max_value
    return normalized_dict

# Assuming you have the integrated Sobol' indices in dictionaries like integrated_S1 and integrated_ST
# Find the maximum integrated value for normalization
max_cumulated_value = max(max(cumulated_S1[feature].max() for feature in cumulated_S1),
                          max(cumulated_ST[feature].max() for feature in cumulated_ST))
# Normalize the integrated Sobol' indices
normalized_cumulated_S1 = normalize_cumulated_indices(cumulated_S1, max_cumulated_value)
normalized_cumulated_ST = normalize_cumulated_indices(cumulated_ST, max_cumulated_value)
# Update the indices_dicts list with the normalized values
indices_dicts = [time_averaged_S1, maximum_S1, normalized_cumulated_S1,
                 time_averaged_ST, maximum_ST, normalized_cumulated_ST]
# Define a blue color palette using a Matplotlib colormap
n_colors = 6  # Number of different colors you need
blue_palette = cm.get_cmap('Blues', n_colors)
# Generate colors from the palette
colors = blue_palette(range(n_colors))
# Common font size for all text in the plots
common_font_size = 16


def plot_grouped_sobol_indices(indices_dicts, titles, feature_names, param_names, subfigure_titles, colors):
    n_params = len(param_names)  # Number of parameters
    n_indices = len(indices_dicts)  # Total number of indices dictionaries
    fig, axes = plt.subplots(nrows=2, ncols=len(feature_names), figsize=(16, 10))
    bar_width = 0.1
    # Since you have 2 rows (S1 and ST indices), divide n_indices by 2 for positions in each row
    indices_positions = [np.arange(n_params) + i * bar_width for i in range(n_indices // 2)]

    for col_idx, feature in enumerate(feature_names):
        for row_idx, (indices_type, row_title) in enumerate(zip([indices_dicts[:3], indices_dicts[3:]], ['S1 Indices', 'ST Indices'])):
            ax = axes[row_idx, col_idx]

            for i, (indices_dict, title) in enumerate(zip(indices_type, titles)):
                # Ensure indices_dict[feature] returns an array of length n_params
                # This should be of shape (n_params,)
                indices = indices_dict[feature]
                ax.bar(indices_positions[i % (n_indices // 2)], indices, width=bar_width, label=title, color=colors[i+3])

            ax.set_xticks([p + bar_width for p in range(n_params)])
            ax.set_xticklabels(param_names, rotation=55, fontsize=common_font_size+1, fontweight='semibold')

            subfig_title = subfigure_titles[row_idx * len(feature_names) + col_idx]
            ax.set_title(subfig_title, fontsize=common_font_size+1, fontweight='bold', pad=15)
            if col_idx == 0:
                ax.set_ylabel('Sobol Index Value', fontsize=common_font_size, labelpad=10)
            ax.set_ylim([0, 1])  # Set y-axis range from 0 to 1
            ax.grid(True)
            if col_idx == 0 and row_idx == 0:
                ax.legend(fontsize=common_font_size, loc='upper left', bbox_to_anchor=(0.1, 1))
            # Adjusting the tick font size
            ax.tick_params(axis='y', labelsize=common_font_size-1)
    plt.tight_layout(h_pad=2, w_pad=2)
    plt.savefig(fname='Fig_Sobol_indices_grouped.pdf', format='pdf', dpi=400)
    plt.show()


# Define feature names and parameter names
feature_names = list(trained_GPs.keys())
# New titles for subfigures
subfigure_titles = ["First-Order Sobol Indices - Vertical convergence",
                    "First-Order Sobol Indices - Horizontal convergence",
                    "Total-Order Sobol Indices - Vertical convergence",
                    "Total-Order Sobol Indices - Horizontal convergence"]
# Combine the dictionaries in a list
legends_titles = ["Time-averaged", "Maximum", "Cumulated"]
# Call the plotting function with updated parameters
plot_grouped_sobol_indices(indices_dicts, legends_titles, feature_names, symbol_list, subfigure_titles, colors)


#%%---------------------------------------------------------------------------------------------------------------------------------------
# ***Calibration process with monitoring data***
#-----------------------------------------------------------------------------------------------------------------------------------------
#***IMPORTANT NOTE: Monitoring data are available from the authors with the permission of the Federal Company for Radioactive Waste Disposal (BGE mbH)***

# Extract the features
K_h = data_moni['K_h [mm]'].values
K_v = data_moni['K_v [mm]'].values
# Combine features for normalization
data_moni_combined = np.column_stack((K_v, K_h))
# Normalize the combined features
data_moni_normalized = normalize_data(data_moni_combined, min_Y, max_Y)
# Create a time sequence
time_moni = data_moni["years"]
# Assuming target_time_values and Y_train are given and already sorted
# Replace with your actual time sequence if different
time_train = sorted(target_time_values)
# Find the closest time points in data_moni to the time_train
closest_indices = [np.argmin(np.abs(time_moni - t)) for t in time_train]
filtered_time_moni = time_moni[closest_indices]
filtered_data_moni_normalized = data_moni_normalized[closest_indices, :]

# Determine the duration for the calibration of the surrogate model parameters
start_time3 = time.time()

# Defining the objective function to minimize, which is the distance between monitoring data and model prediction
def objective_function(X, trained_GPs, filtered_data_moni_normalized, features, weights):
    total_distance = 0 
    # Loop through each feature
    for j, (feature, weight) in enumerate(zip(features, weights)):
        # Predict using the trained GP for the current feature
        Y_pred, _ = trained_GPs[feature].predict(X.reshape(1, -1), return_std=True)
        # Calculate the distance (mean squared error) for the current feature
        distance = np.mean((filtered_data_moni_normalized[:, j] - Y_pred[:,:].flatten())**2)
        # Multiply the distance by the corresponding weight
        total_distance += weight * distance
    return total_distance

# Define a function to denormalize the parameter values
def denormalize_data(data, min_val, max_val):
    return data * (max_val - min_val) + min_val

# Set the random seed for reproducibility
np.random.seed(255)  # You can use any integer value as the seed
# Initial condition for the optimization
X_nominal_initial = np.array([0.5, X_nominal_norm[1], X_nominal_norm[2], 0.5, 0.5, X_nominal_norm[5], 0.5])
# Generate random perturbations within ±0.5
perturbations = np.random.uniform(-0.5, 0.5, X_nominal_initial.shape)
# Initialize with random values within the ±0.5 range, ensuring within 0-1 range
X_initial = np.clip(X_nominal_initial + perturbations, 0.01, 0.99)
print(X_initial)
# Create bounds for each element in X_nominal_initial
bounds = [(0.05, 0.95) if i not in [1, 2, 5] else (X_nominal_initial[i], X_nominal_initial[i]) for i in range(len(X_nominal_initial))]
print(bounds)

# Set up the differential evolution algorithm
iteration_count = 0
result = differential_evolution(
    objective_function,
    bounds,
    args=(trained_GPs, filtered_data_moni_normalized, features, weights),
    strategy='best1bin',  # Commonly used strategy
    maxiter=1000,  # Maximum number of generations
    popsize=15,  # Population size
    tol=0.01,  # Relative tolerance for convergence
    mutation=(0.5, 1),  # Mutation constant
    recombination=0.7,  # Recombination constant
    seed=42,  # Seed for reproducibility
    callback=lambda xk, convergence: print(f"Current solution: {xk}")
)

# Optimal solution
X_optimal = result.x
# Denormalize X_optimal
X_optimal_denorm = denormalize_data(X_optimal, min_X, max_X)
print("-" * 50)
print("Optimal X:", X_optimal_denorm)
print("Objective function value:", result.fun)
# nit is the number of iterations and nfev the number of call of the surrogate model
print("Optimization result:", result)
# Determine the duration for the calibration of the surrogate model parameters
end_time3 = time.time()
duration_optimizing = (end_time3 - start_time3)
print(f"Calibration of the surrogate model parameters took {duration_optimizing:.2f} seconds")
print("-" * 50)
# Prints the optimal values for each parameter as they should be used in FLAC3D
for i, para in enumerate(param_list):
    print('Optimal ' + str(para) + ': ' + str(round(X_optimal_denorm[i], 2)))


# Remove fixed values from lists
calibrated_params = [0, 3, 4, 6]
param_list_new = [param_list[i] for i in calibrated_params]
symbol_list_new = [symbol_list[i] for i in calibrated_params]
X_optimal_denorm_new = [X_optimal_denorm[i] for i in calibrated_params]
X_optimal_denorm_new = [round(X_optimal_denorm_new[i]/86400, 0) if i in [0, 2] else round(X_optimal_denorm_new[i], 2) for i in range(0, len(X_optimal_denorm_new))]
X_nominal_new = [X_nominal[i] for i in calibrated_params]
X_nominal_new = [X_nominal_new[i]/86400 if i in [0, 2] else X_nominal_new[i] for i in range(0, len(X_nominal_new))]


# Import of the FLAC3D output for the optimized TUBSsalt parameter values
data_f3_opt = pd.read_csv(foldername+r'\VD_Gorleben2_optimal.txt', sep=' ')
data_f3_opt.drop(index=[data_f3_opt.index[0]], inplace=True)
# Dropping the first row and columns with all NaN values
data_f3_opt.dropna(axis=1, how='all', inplace=True)
column_mapping = {'Unnamed: 1': 'steps', 'Unnamed: 3': 'days', 'Unnamed: 4': 'disp_z_top', 'Unnamed: 6': 'disp_z_bot', 'Unnamed: 7': 'disp_x_wall', }
# Rename columns
data_f3_opt.rename(columns=column_mapping, inplace=True)
data_f3_opt.replace('-------------', float('nan'), inplace=True)
# Converting specific columns to float
columns_to_float = ['steps', 'days', 'disp_z_top']
data_f3_opt[columns_to_float] = data_f3_opt[columns_to_float].astype(float)
# Transformes the displacements into the horizontal and vertical convergence
data_f3_opt['Vertical_converg'] = (data_f3_opt['disp_z_top'] - data_f3_opt['disp_z_bot']) * 1000
data_f3_opt['Horizontal_converg'] = (data_f3_opt['disp_x_wall'] + data_f3_opt['disp_x_wall']) * 1000
# Generates cubic spline interpolation function for simulation convergences
spline_K_v_sim = PchipInterpolator(data_f3_opt['days'], data_f3_opt['Vertical_converg'])
spline_K_h_sim = PchipInterpolator(data_f3_opt['days'], data_f3_opt['Horizontal_converg'])
# Determination of the simulation convergence where the monitoring starts (day 1)
conv_v_index0 = spline_K_v_sim(diff_time)
conv_h_index0 = spline_K_h_sim(diff_time)
data_f3_opt['Vertical_converg'] = data_f3_opt['Vertical_converg'] - conv_v_index0
data_f3_opt['Horizontal_converg'] = data_f3_opt['Horizontal_converg'] - conv_h_index0
# Adjusting the dataframe for the giving index_day0
data_f3_opt = data_f3_opt[data_f3_opt['days'] >= 1.0]
data_f3_opt['days'] = data_f3_opt['days'] - diff_time
data_f3_opt['years'] = data_f3_opt['days']/365
data_f3_opt.loc[0, data_f3_opt.columns] = [0]*8
data_f3_opt = data_f3_opt.sort_values(by='days')
data_f3_opt = data_f3_opt.reset_index(drop=True)


# Plotting the prediction for optimized parameter values, the corresponding FLAC3D output and the monitoring data
feature_titles = {'Vertical_converg': 'Vertical Convergence', 'Horizontal_converg': 'Horizontal Convergence'}
fig, axes = plt.subplots(1, len(features), figsize=(16, 6), dpi=300)  # Increased figsize and dpi for better resolution
for j, feature in enumerate(features):
    ax = axes[j]
    # Plot simulation data for optimal values
    ax.plot(data_f3_opt['years'], data_f3_opt[feature], color=color1,
            linestyle='-', linewidth=2, alpha=1, label='FLAC3D output for GP approximation')
    # Extract the true values for the current feature
    Y_true = Y_train[j]
    # Predict using the trained GP for the current feature
    Y_pred_optimal, _ = trained_GPs[feature].predict(X_optimal.reshape(1, -1), return_std=True)
    Y_pred_optimal_denorm = denormalize_data(Y_pred_optimal, min_Y, 0)
    # Denormalization of filtered monitoring data
    filtered_data_moni_denorm = denormalize_data(filtered_data_moni_normalized, min_Y, 0)
    # Plot the optimal GP approximation
    ax.plot(time_train, Y_pred_optimal_denorm.flatten(), 'o', markersize=6, color=color3,
            alpha=0.7, label='Optimal GP approximation')
    # Plot denormalized monitoring data
    ax.plot(filtered_time_moni, filtered_data_moni_denorm[:, j], 's',
            markersize=6, color='black', alpha=0.9, label='Monitoring Data')
    # Corrected the indexing here
    ax.set_title(f'{feature_titles[feature]}', fontsize=18, fontweight='semibold', pad=15)
    ax.set_xlabel(r'Time [years]', fontsize=16)
    if j == 0:
        ax.set_ylabel(r'Convergence [mm]', fontsize=16)
    # Adjusting the tick font size
    ax.tick_params(axis='both', labelsize=15)
    # Set the axes limits
    ax.set_xlim([-0.5, max(data_moni['years'])+0.5])
    ax.set_ylim([-125, 5])
    ax.yaxis.set_inverted(True)
    # Set the number of ticks for the x and y axes
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(True)
    if j == 0:
        ax.legend(loc='lower right', fontsize=16)
        # Textbox with optimal values
        bbox = dict(boxstyle='round', facecolor='white', edgecolor='lightgrey', alpha=0.8)
        ax.text(3.55, -32, 'Optimal values for $\eta_p$, $E_p$, $\eta_s$ and $p_s$: \n'+str(X_optimal_denorm_new), fontsize=16, bbox=bbox)
# Adjust layout for better spacing
plt.tight_layout(h_pad=2)
fig.savefig(fname='Fig_GPs_approximation_denormalized_'+str(sets_train)+'samples.pdf', format='pdf', bbox_inches='tight', dpi=400)
plt.show()

