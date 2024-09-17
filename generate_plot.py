# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 19:42:52 2024

@author: kaueu
"""

import pandas as pd
import numpy as np
from matplotlib.lines import Line2D

# Function to process a single CSV file
def process_csv(csv_filename):
    # Read CSV file
    df = pd.read_csv(csv_filename)
    print(csv_filename)
    # Extract architecture and dataaug from filename
    architecture, dataaug = csv_filename.replace('.csv','').split('_')
    
    # Add columns: Epoch, Architecture, DataAug
    df['Epoch'] = range(1, len(df) + 1)
    df['Architecture'] = architecture
    df['DataAug'] = int(dataaug)

    return df

# Read the list of CSV filenames from txt file
with open('csv_files.txt', 'r') as file:
    csv_filenames = file.read().splitlines()

# Initialize an empty dataframe
main_dataframe = pd.DataFrame()
dataframes = []
# Process each CSV file and append to the main dataframe
for csv_filename in csv_filenames:
    df = process_csv(csv_filename)
    dataframes.append(df)
    #main_dataframe = main_dataframe.append(df, ignore_index=True)

main_dataframe = pd.concat(dataframes, ignore_index=True)

# Display the resulting dataframe
#print(main_dataframe)

import matplotlib.pyplot as plt
import seaborn as sns
# Set seaborn style

choice = 6

if choice == 1:
    sns.set(style="darkgrid")
    
    desired_order = ['VGG16b', 'VGG16a', 'VGG16', 'VGG19', 'ResNet50', 'ResNet152']
    
    # Define a color palette based on unique architectures
    architectures = main_dataframe['Architecture'].unique()
    palette = sns.color_palette("husl", n_colors=len(architectures))
    
    
    
    
    # Loop through DataAug values and create plots
    for dataaug_value in main_dataframe['DataAug'].unique():
        # Filter the dataframe for the current DataAug value
        filtered_df = main_dataframe[main_dataframe['DataAug'] == dataaug_value]
    
        # Loop through accuracy and loss
        #for metric in ['accuracy', 'val_accuracy', 'loss', 'val_loss']:
            # Filter dataframe for the specific metric
        #    metric_df = filtered_df.filter(like=metric)
    
            # Set linestyle and label based on metric
        #linestyle = '--' if 'val_' in metric else '-'
            #label = metric.replace('val_', '')
        from matplotlib.lines import Line2D
            # Plot the data
        plt.figure()
        val_plot = sns.lineplot(x='Epoch', y='val_accuracy', hue='Architecture', 
                     style='Architecture', dashes=[(2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)], data=filtered_df,legend=False,hue_order=desired_order)
        train_plot = sns.lineplot(x='Epoch', y='accuracy', hue='Architecture', 
                     style='Architecture', dashes=False, data=filtered_df,hue_order=desired_order)
        
        plt.ylim(0.4, 1)
        # Create separate legends for "Training" and "Validation"
    
        # Creating custom legend
        custom_legend = plt.legend(handles=[
            Line2D([0], [0], color='black', linestyle=':', label='Validation'),
            Line2D([0], [0], color='black', linestyle='-', label='Training')
        ], title='Line Style')
        
        # Adding the custom_legend to the figure
        #plt.gca().add_artist(train_plot.legend())
        custom_legend.set_bbox_to_anchor((1.315, 0.25))
        
        plt.gca().add_artist(custom_legend)
        
        # Set plot title and show legend
        plt.title(f'Accuracy value - Augmented x{dataaug_value} times')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #plt.show()
        plt.savefig(f'Accuracy_{dataaug_value}.pdf', bbox_inches='tight')
if choice == 2:
    sns.set(style="darkgrid")
    print(main_dataframe['DataAug'].unique())
    main_dataframe['Augmented by'] = main_dataframe['DataAug'].astype(str) + 'x'
    desired_order = ['1x','3x','6x']
    # Define a color palette based on unique architectures
    dataaug = main_dataframe['Augmented by'].unique()
    palette = sns.color_palette("husl", n_colors=len(dataaug))
    
    # Loop through DataAug values and create plots
    for architecture_value in main_dataframe['Architecture'].unique():
        # Filter the dataframe for the current DataAug value
        filtered_df = main_dataframe[main_dataframe['Architecture'] == architecture_value]
    
        # Plot the data
        plt.figure()
        val_plot = sns.lineplot(x='Epoch', y='val_accuracy', hue='Augmented by', 
                     style='Augmented by', dashes=[(2, 2), (2, 2), (2, 2)], data=filtered_df,legend=False,hue_order=desired_order)
        train_plot = sns.lineplot(x='Epoch', y='accuracy', hue='Augmented by', 
                     style='Augmented by', dashes=False, data=filtered_df,hue_order=desired_order)
        
        plt.ylim(0.4, 1)
        # Create separate legends for "Training" and "Validation"
    
        # Creating custom legend
        custom_legend = plt.legend(handles=[
            Line2D([0], [0], color='black', linestyle=':', label='Valid'),
            Line2D([0], [0], color='black', linestyle='-', label='Train')
        ], title='Style')
        
        # Adding the custom_legend to the figure
        #plt.gca().add_artist(train_plot.legend())
        custom_legend.set_bbox_to_anchor((1.215, 0.35))
        
        plt.gca().add_artist(custom_legend)
        
        # Set plot title and show legend
        plt.title(f'Accuracy value - {architecture_value}')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #plt.show()
        plt.savefig(f'Accuracy_{architecture_value}.pdf', bbox_inches='tight')    

if choice == 3:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.lines import Line2D
    
    
    sns.set(style="darkgrid")
    # Define the mapping dictionary
    architecture_mapping = {'VGG16b': 'A', 'VGG16a': 'B', 'VGG16': 'C', 'VGG19': 'D', 'ResNet50': 'E', 'ResNet152': 'F'}
    
    # Map the values in the 'Architecture' column using the defined mapping
    main_dataframe['Architecture'] = main_dataframe['Architecture'].map(architecture_mapping)

    
    print(main_dataframe['DataAug'].unique())
    main_dataframe['Augmented by'] = main_dataframe['DataAug'].astype(str) + 'x'
    desired_order = ['1x', '3x', '6x']
    dataaug = main_dataframe['Augmented by'].unique()
    palette = sns.color_palette("husl", n_colors=len(dataaug))
    
    # Create a 2x3 grid of subplots
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8), sharey=True)
    axes = axes.flatten()
    
    # Loop through DataAug values and create plots
    for i, architecture_value in enumerate(['A','B','C','D','E','F']):
        # Filter the dataframe for the current DataAug value
        filtered_df = main_dataframe[main_dataframe['Architecture'] == architecture_value]
    
        # Plot the data
        val_plot = sns.lineplot(x='Epoch', y='val_accuracy', hue='Augmented by',
                                style='Augmented by', dashes=[(2, 2), (2, 2), (2, 2)], data=filtered_df, legend=False, hue_order=desired_order,
                                ax=axes[i])
        if i==4:
            train_plot = sns.lineplot(x='Epoch', y='accuracy', hue='Augmented by',
                                      style='Augmented by', dashes=False, data=filtered_df, hue_order=desired_order, ax=axes[i])
            
        else:
            train_plot = sns.lineplot(x='Epoch', y='accuracy', hue='Augmented by',
                                      style='Augmented by', dashes=False, data=filtered_df, hue_order=desired_order,legend=False, ax=axes[i])

            
        axes[i].set_ylim(0.4, 1)
        if i==5:
            # Creating custom legend
            custom_legend = axes[i].legend(handles=[
                Line2D([0], [0], color='black', linestyle=':', label='Valid'),
                Line2D([0], [0], color='black', linestyle='-', label='Train')
            ], title='Style')
        
            # Adding the custom_legend to the subplot
            custom_legend.set_bbox_to_anchor((0.75, 0.7))
            axes[i].add_artist(custom_legend)
            #axes[i].set_ylabel(f'Model {architecture_value}')
        # Set subplot title and show legend
        axes[i].set_title(f'Model {architecture_value}')
    axes[0].set_ylabel('Accuracy')
    axes[3].set_ylabel('Accuracy')
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Show or save the figure
    #plt.show()
    plt.savefig('subplots.pdf', bbox_inches='tight')
if choice == 4:
    df = pd.read_csv('C://Users//kaueu//OneDrive//Área de Trabalho//Brainn_conf//accuracy_value_models.csv')
    sns.set_style("whitegrid", {'grid.linestyle': '-', 'grid.color': '#ddd'})
    # Function to filter top 3 for rates 3 and 6, and bottom 3 for rate 1
    def filter_top_bottom(df):
        # Sort by Accuracy within each group
        df_sorted = df.sort_values(by='Accuracy', ascending=False).groupby(['DataAugType', 'NetworkName', 'DataAugRate'])
        
        # Top 3 for rates 3 and 6
        top_3 = df_sorted.apply(lambda x: x.head(3) if x['DataAugRate'].iloc[0] in [3, 6] else pd.DataFrame())
        
        # Bottom 3 for rate 1
        bottom_3 = df_sorted.apply(lambda x: x.tail(3) if x['DataAugRate'].iloc[0] == 1 else pd.DataFrame())
        
        # Combine the filtered data
        new_df = pd.concat([top_3, bottom_3]).drop_duplicates().reset_index(drop=True)
        return new_df
    
    # Filter the DataFrame
    new_df = filter_top_bottom(df)
    architecture_mapping = {'VGG16b': 'A', 'VGG16a': 'B', 'VGG16': 'C', 'VGG19': 'D', 'ResNet50': 'E', 'ResNet152': 'F'}
    
    # Map the values in the 'Architecture' column using the defined mapping
    new_df['NetworkName'] = new_df['NetworkName'].map(architecture_mapping)
    new_df.sort_values('NetworkName',inplace=True)
    print(new_df)
    fig, ax = plt.subplots()
    for i in range(6):  # We have 6 models
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, facecolor='lightblue', alpha=0.3)
    sns.boxplot(new_df[new_df['DataAugType']=='Yes'],y='Accuracy',x='NetworkName',hue='DataAugRate',
                boxprops=dict(facecolor='white',edgecolor='black',zorder=3),
                whiskerprops=dict(color='black',zorder=4),  # Set the whiskers to black
                capprops=dict(color='black',zorder=4),      # Set the caps to black
                medianprops=dict(color='black',zorder=4),   # Set the median line to black
                flierprops=dict(markeredgecolor='black',zorder=4),
                ax=ax)  # Set the outliers to black
    positions = sorted(new_df['NetworkName'].unique())
    plt.ylim(0.57,0.87)
    plt.legend([],[], frameon=False)
    # Draw the vertical dashed lines
    for idx, pos in enumerate(positions):
        x_pos = list(new_df['NetworkName'].unique()).index(pos)
        plt.axvline(x=x_pos-0.27, color='blue', linestyle='--', linewidth=1, alpha=0.3,label='Rate 1')
        plt.axvline(x=x_pos, color='orange', linestyle='--', linewidth=1, alpha=0.3, label='Rate 3')
        plt.axvline(x=x_pos+0.27, color='green', linestyle='--', linewidth=1, alpha=0.3, label='Rate 6')
        # Adding text labels
        if idx%2 == 0:
            colorrr = '#e6f3f7'
        else:
            colorrr = 'white'
        plt.text(x_pos - 0.27, plt.ylim()[0], '1', color='blue', ha='center', va='bottom',fontsize=14,
                 bbox=dict(facecolor=colorrr, edgecolor='none', pad=0))
        plt.text(x_pos, plt.ylim()[0], '3', color='orange', ha='center', va='bottom',fontsize=14,
                 bbox=dict(facecolor=colorrr, edgecolor='none', pad=0))
        plt.text(x_pos + 0.27, plt.ylim()[0], '6', color='green', ha='center', va='bottom',fontsize=14,
                 bbox=dict(facecolor=colorrr, edgecolor='none', pad=0))
        
    plt.xlabel('Models')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('accuracies.pdf')
    # Customize legend
    #handles, labels = plt.gca().get_legend_handles_labels()
    #by_label = dict(zip(labels, handles))
    
    #plt.legend(by_label.values(), by_label.keys())   
if choice == 5:
    df = pd.read_csv('C://Users//kaueu//OneDrive//Área de Trabalho//Brainn_conf//accuracy_value_models.csv')
    # Function to filter top 3 for rates 3 and 6, and bottom 3 for rate 1
    def filter_top_bottom(df):
        # Sort by Accuracy within each group
        df_sorted = df.sort_values(by='Accuracy', ascending=False).groupby(['DataAugType', 'NetworkName', 'DataAugRate'])
        
        # Top 3 for rates 3 and 6
        top_3 = df_sorted.apply(lambda x: x.head(3) if x['DataAugRate'].iloc[0] in [3, 6] else pd.DataFrame())
        
        # Bottom 3 for rate 1
        bottom_3 = df_sorted.apply(lambda x: x.tail(3) if x['DataAugRate'].iloc[0] == 1 else pd.DataFrame())
        
        # Combine the filtered data
        new_df = pd.concat([top_3, bottom_3]).drop_duplicates().reset_index(drop=True)
        return new_df
    
    # Filter the DataFrame
    new_df = filter_top_bottom(df)
    architecture_mapping = {'VGG16b': 'A', 'VGG16a': 'B', 'VGG16': 'C', 'VGG19': 'D', 'ResNet50': 'E', 'ResNet152': 'F'}
    
    # Map the values in the 'Architecture' column using the defined mapping
    new_df['NetworkName'] = new_df['NetworkName'].map(architecture_mapping)
    new_df.sort_values('NetworkName',inplace=True)
    print(new_df)  
    filtered_df = new_df[new_df['DataAugType'] == 'Yes']
    
    # Pivot the DataFrame
    pivot_table = filtered_df.pivot_table(
        index='NetworkName',
        columns='DataAugRate',
        values='Accuracy'
    )
    
    # Create LaTeX table
    latex_table = pivot_table.to_latex(
        column_format='l' + 'c' * len(pivot_table.columns),
        header=True,
        index=True,
        caption='Accuracy of Models with Different Data Augmentation Rates',
        label='tab:accuracy'
    )
    
    # Save the LaTeX table to a .tex file
    with open('accuracy_table.tex', 'w') as f:
        f.write(latex_table)
if choice == 6:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.lines import Line2D
    
    
    sns.set(style="darkgrid")
    # Define the mapping dictionary
    architecture_mapping = {'VGG16b': 'A', 'VGG16a': 'B', 'VGG16': 'C', 'VGG19': 'D', 'ResNet50': 'E', 'ResNet152': 'F'}
    
    # Map the values in the 'Architecture' column using the defined mapping
    main_dataframe['Architecture'] = main_dataframe['Architecture'].map(architecture_mapping)

    
    print(main_dataframe['DataAug'].unique())
    main_dataframe['Augmented by'] = main_dataframe['DataAug'].astype(str) + 'x'
    desired_order = ['1x', '3x', '6x']
    dataaug = main_dataframe['Augmented by'].unique()
    palette = sns.color_palette("husl", n_colors=len(dataaug))
    
    # Create a 2x3 grid of subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharey=True)
    axes = axes.flatten()
    
    # Loop through DataAug values and create plots
    for i, architecture_value in enumerate(['A','B','C','D']):#,'E','F']):
        # Filter the dataframe for the current DataAug value
        filtered_df = main_dataframe[main_dataframe['Architecture'] == architecture_value]
    
        # Plot the data
        val_plot = sns.lineplot(x='Epoch', y='val_accuracy', hue='Augmented by',
                                style='Augmented by', dashes=[(2, 2), (2, 2), (2, 2)], data=filtered_df, legend=False, hue_order=desired_order,
                                ax=axes[i])
        if i==2:
            train_plot = sns.lineplot(x='Epoch', y='accuracy', hue='Augmented by',
                                      style='Augmented by', dashes=False, data=filtered_df, hue_order=desired_order, ax=axes[i])
            
        else:
            train_plot = sns.lineplot(x='Epoch', y='accuracy', hue='Augmented by',
                                      style='Augmented by', dashes=False, data=filtered_df, hue_order=desired_order,legend=False, ax=axes[i])

            
        axes[i].set_ylim(0.4, 1)
        if i==3:
            # Creating custom legend
            custom_legend = axes[i].legend(handles=[
                Line2D([0], [0], color='black', linestyle=':', label='Valid'),
                Line2D([0], [0], color='black', linestyle='-', label='Train')
            ], title='Style')
        
            # Adding the custom_legend to the subplot
            custom_legend.set_bbox_to_anchor((1, 0.3))
            axes[i].add_artist(custom_legend)
            #axes[i].set_ylabel(f'Model {architecture_value}')
        # Set subplot title and show legend
        axes[i].set_title(f'Model {architecture_value}')
    axes[0].set_ylabel('Accuracy')
    axes[2].set_ylabel('Accuracy')
    #axes[4].set_ylabel('Accuracy')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Show or save the figure
    #plt.show()
    plt.savefig('subplots_vertical.pdf', bbox_inches='tight')
    