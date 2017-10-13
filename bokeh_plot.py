# -*- coding: utf-8 -*-
"""
Present an interactive dendrogram for clustered issues.
Use the ``bokeh serve`` command to run the by executing:
    bokeh serve --show CODA_dedn_bokeh.py
@author: Naser Monsefi
"""
# =============================================================================
# Importing needed packages
# =============================================================================
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import set_link_color_palette

from bokeh.io import curdoc
from bokeh.layouts import widgetbox, row, column
from bokeh.models.widgets import Slider, TextInput, DataTable, TableColumn, Button, CheckboxGroup
from bokeh.models.callbacks import CustomJS
from bokeh.plotting import figure
from bokeh.models import LabelSet, ColumnDataSource, Div, Spacer
from bokeh.palettes import Category20_20

# =============================================================================
# importing some functions
# =============================================================================
from data_funcs import load_issues_fast, make_tfidf, create_universe_Z, make_Z, make_R, match

# =============================================================================
# Loading data and setting up some initial variables
# =============================================================================
from flask import Flask, render_template, request
import pandas as pd
from bokeh.charts import Histogram
from bokeh.embed import components


# Load the Iris Data Set

# Index page

#    CODA_export_file = '//longbasp0005/D/Documents/Qlik/Sense/Content/Text Analytics/Data/CODA_Export.csv'

CODA_export_file = 'data/CODA_Export.csv'


issues_df = load_issues_fast(CODA_export_file)

dend_threshold = 3.2

number_of_clusters = 14

    # universe_issues_list = issues_df['CASE_NUMBER'].head(100).tolist()
    # issues_list = ['572', '204', '3191', '202', '709', '626', '858', '98']
issues_dict = dict(issues_list=[""], issues_membership=[""])

    # creat a list for the all issues interesterd called universe issues
    # removing rejected and remediated issues
universe_issues_list = \
issues_df.CASE_NUMBER[~issues_df.DETAILED_DESCRIPTION.isin(["Rejected", "Remediated"])].tolist()

df_tfidf = pd.DataFrame()

    # set the color palette to be used for assigning denrogram colors
set_link_color_palette(Category20_20)

    # =============================================================================
    # Create a new figure with a needed tools
    # =============================================================================
p = figure(tools="pan,wheel_zoom,box_zoom,box_select,save,reset",
               plot_width=1000,
               plot_height=800,
               )
    # =============================================================================
    # Create source data containing and making the plot
    # =============================================================================
    # adding the threshold line as an exrta line in the last entry
dend_source = ColumnDataSource(data=dict(x=[], y=[], leg_color=[]))
leaves_source = ColumnDataSource(data=dict(leaves_x=[], leaves_y=[],
                                               leaves_names=[], leaves_color=[],
                                               leaves_size=[], clusters=[]))
image_source = ColumnDataSource(data=dict(
            url=["static/static_dend.png", "", ""],
            text=["", "Enter your issues IDs", ""],
            text_x=[0, 40, 100],
            text_y=[0, 40, 80]))
    # making fake source data for dend slider
source_for_slider = ColumnDataSource(data=dict(value=[]))

    # add the static dend
p.image_url(url="url", x=0, y=0, w=100, h=80, anchor="bottom_left",
                source=image_source)
p.text(x="text_x", y="text_y", text="text", source=image_source)

    # Plot dendrogram U shapes using multi_line function
p.multi_line("x", "y", color="leg_color", source=dend_source)

    # remove the x axis tickers by setting them to empty
p.xaxis.ticker = []

    # Add label to axex
p.yaxis.axis_label = "Distance"
p.xaxis.axis_label = "Issue ID"

    # creating corcles at the end of each leaf
p.circle(x="leaves_x", y="leaves_y", fill_alpha=0.7,
             size="leaves_size", fill_color="leaves_color",
             source=leaves_source)
    # Adding lables to plot
labels = LabelSet(x="leaves_x", y="leaves_y", text="leaves_names", angle=-60,
                      angle_units="deg", y_offset=-10, x_offset=-5,
                      source=leaves_source, render_mode='canvas')
p.add_layout(labels)

    # =============================================================================
    # Set up widgets
    # =============================================================================
    # using the fake data source method for CustumJS callback
    # look at https://stackoverflow.com/questions/38375961/throttling-in-bokeh-application/38379136
dend_threshold_slider = Slider(title="Dendrogram Threshold over distance",
                                   value=2.0,
                                   start=0.0,
                                   end=10,
                                   step=0.05,
                                   callback_policy='mouseup',
                                   callback=CustomJS(
                                           args=dict(source=source_for_slider),
                                           code="""
                                           source.data = { value: [cb_obj.value] }
                                           """))
issues_list_input = TextInput(title="Enter issues IDs")
clipboard_text = TextInput(title="Selected issues IDs")
columns = [
            TableColumn(field="leaves_names", title="Issue ID"),
            TableColumn(field="clusters", title="Cluster"),
            ]
data_table = DataTable(source=leaves_source, columns=columns, width=280)
submit_button = Button(label="Submit", button_type="success")
universe_checkbox = CheckboxGroup(
            labels=["Universe issues dendrogram (please allow 30s to complete)"])
    # =============================================================================
    # Set up callbacks
    # =============================================================================


def update_issue_list():
        """
        Callback function that updates issues_list dict, tfidf and Z
        """
        global issues_dict
        global Z
        global dend_threshold
        change_dend = False
        # get new input issues list
        issues_list = issues_list_input.value.strip()
        issues_list = issues_list.replace('\n',',').replace('|',',').replace('\t',',').replace(' ',',').replace(',,',',').split(",")
        # universe is selected
        if (0 in universe_checkbox.active):
            # set issues membership to work for node color
            # all universe nodes turns to light blue first
            issues_dict["issues_membership"] = \
                ["lightblue"] * len(universe_issues_list)
            # if there is any user list, turn their membership to black
            for x in match(issues_list, universe_issues_list):
                issues_dict["issues_membership"][x] = "black"
            issues_dict["issues_list"] = universe_issues_list
            Z = create_universe_Z(CODA_export_file, issues_df,
                                  universe_issues_list, number_of_clusters)
            # the epsilon (1e-10) was added for correct collering of dendrogram
            # The exact Z[-k,2] value results in an extra cluster colored
            dend_threshold = Z[-number_of_clusters, 2] + 1e-10
            change_dend = True
        elif (issues_list != [""]):
            issues_dict["issues_list"] = issues_list
            # if there is only user list, all of them have black memebership
            issues_dict["issues_membership"] = ["black"] * len(issues_list)
            # use match function to find the input issue list in the universe list
            # avoid using pandas isin as it keep the reference df order not the
            # input issues list order
            df_tfidf = make_tfidf(
                    issues_df.iloc[match(issues_dict['issues_list'],
                                         issues_df.CASE_NUMBER.tolist()), ])
            Z = make_Z(df_tfidf)
            # arbitrary dend threshold for non-unverse dednrigrams
            dend_threshold = max(Z[:, 2]) / 1.7
            change_dend = True
        if change_dend:
            # clearing the static image
            image_source.data = dict(url=[], text=[], text_x=[], text_y=[])
            # change slider end point
            dend_threshold_slider.update(end=max(Z[:, 2]),
                                         value=dend_threshold,
                                         )
            # update the dnedrogram
            update_dend()


def update_dend():
        """
        Callback function that change the dendrogram by calling make_R

        Updates whole dendrogram, including legs colors
        Updates the leaves, including leaves colors, clusters and membership
        """
        # Use the unpdated dend_threshold
        # Creat new dendrogram
        R = make_R(Z, dend_threshold, issues_dict["issues_list"])
        # Generate the new curve
        dend_source.data = dict(
                x=R['icoord'] + [[0, max(max(R['icoord']))]],
                y=R['dcoord'] + [[dend_threshold, dend_threshold]],
                leg_color=R['color_list'] + ["#FF0000"])
        # find the correct order of nodes for the membership assignment
        membership_index = match(R['ivl'], issues_dict["issues_list"])
        # use the membership_index to find correct color and size for nodes
        # base on their universe or user list membership
        leaves_color = \
            [issues_dict["issues_membership"][x] for x in membership_index]
        # if the node color is black set size to 9 otherwise set it to 5
        leaves_size =  \
            np.where(np.array(leaves_color) == "black", 9, 5).tolist()
        leaves_source.data = dict(
                leaves_x=list(range(5, len(R["ivl"]) * 10, 10)),
                leaves_y=[0] * len(R["ivl"]),
                leaves_names=R['ivl'],
                leaves_color=leaves_color,
                leaves_size=leaves_size,
                clusters=R['Clusters'],
                )


def update_clipboard_text(attr, old, new):
        """
        Callback function that put the selected issues in the clipboard text box
        for exporting.
        """
        clipboard_df = pd.DataFrame({"Issues" : [leaves_source.data['leaves_names'][i] for i in new['1d']['indices']],
                                     "Clusters" : [leaves_source.data['clusters'][i] for i in new['1d']['indices']]})
        # [leaves_source.data['leaves_names'][i] for i in new['1d']['indices']]
        clipboard_text.value = " ".join(clipboard_df['Issues'])
        clipboard_df.to_csv("clipboard_df.csv", sep=',', encoding='utf-8')

    # use an intermidiate function to pass new threshold to dend update


def slider_delay(attr, old, new):
        global dend_threshold
        dend_threshold = source_for_slider.data["value"][0]
        update_dend()
source_for_slider.on_change("data", slider_delay)

    # calling call back function upon changes in interactive widgets
issues_list_input.on_change('value', lambda attr, old, new: update_issue_list())
submit_button.on_click(update_issue_list)
    #dend_threshold_slider.on_change('value', lambda attr, old, new: update_dend())
leaves_source.on_change('selected', update_clipboard_text)
universe_checkbox.on_change('active', lambda attr, old, new: update_issue_list())
    # =============================================================================
    # Set up layouts and add to document
    # =============================================================================
    # making the descrption for the page
page_header = Div(text="""<h1>Data Quality Issue Clustering</h1>
                      Start creating your issues dendrogram by entering your issues
                      IDs. Fine tune your clusters by changing the <b>Dendrogram
                      Threshold</b>. (Click on the link for more information about
                      <a href="https://en.wikipedia.org/wiki/Dendrogram" target="_blank">dendrograms</a>).
                      The following DQ Direct fields were used for
                      this clustering: <b>Source System, Issue Detection Point,
                      Issue Summary, Detailed Description</b> and <b>Attribute Document
                      Header</b>. Check the tick box for <b>Universe issues
                      dendrogram</b> if you want to compare your issues list with
                      all other non-rejected and non-remediated issues.
                      """, width=1300)
    # have to move to directory file system to use static images
link_to_qlik = Div(text="""
                       <a href="https://www.google.com" target="_blank">
                           <img float:right src="static/qlik_link.png" alt="DQ Direct MI Dashboard" height="50" width="70", align="right">
                           <h3>DQ Direct MI Dashboard</h3>
                       </a>
                        """)
page_footer = Div(text="""Developed by
                      <a href="https://www.google.com">DQ Analytics</a>.
                      Any questions please contact \
                      <a href="https://www.google.com" target="_blank">Andrew McGeough</a> or
                      """, width=1200)

    # laying out widgets and figure
page_layout = column(
            page_header,
            row(column(widgetbox([issues_list_input, submit_button, universe_checkbox],
                                 width=300),
                       dend_threshold_slider,
                       Spacer(width=300, height=20),
                       data_table,
                       Spacer(width=300, height=20),
                       clipboard_text,
                   link_to_qlik), p), page_footer)


                # Determine the selected feature

                # Embed plot into HTML via Flask Render
#script, div = components(page_layout)
#return render_template("bokeh.html", script=script, div=div)

# With debug=True, Flask server will auto-reload
# when there are code changes



curdoc().add_root(page_layout)
curdoc().add_root(source_for_slider)

curdoc().title = "Data Quality Issue Clustering"
