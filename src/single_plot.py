# author: Lukas Preis
# date: 4.3.2021
# version: 1.1
# update: plots for composition of database

import os
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import math
from scipy.stats import linregress
import numpy as np

# papers per year relative to size of database (RYX): vertical bar chart
# interpretation as papers and trend as "weight" and "growth" - absolute (XYG): vertical bar chart with linear regression
# interpretation as papers and trend as "weight" and "growth" - relative (RYG): vertical bar chart with linear regression
# composition of database over years (CYX): stacked vertical bar chart
# composition of database total (CXX): pie chart
# composition of database relative (CRX): stackplot


# @todo: adapt colors of stacked bar chart; and change order from top to bottom

### INPUT ###

filename = "Plotfile.txt"
analysis_title = "UAM_Vertiport"
title = 'Rate of UAM- and Vertiport-related Publications'
n_years = 7
sort_items = False
running_avg_length = 3

dir = os.path.dirname(os.path.dirname(__file__))
file = os.path.join(dir, 'logfiles', filename)
log_text = open(file,'r').read()


### FUNCTIONS ###

def read_data(text):

    # token_length = 2
    snippets = text.replace(". ",".").split()
    cleaned_text = " ".join(snippets)
    broken_text = cleaned_text.split()
    points = []
    keyword = ""
    for p in broken_text:
        if p.isnumeric():
            if keyword != "":
                points.append(keyword)
                keyword = ""
            points.append(p)
        elif keyword != "":
            keyword = keyword + " " + str(p)
        else:
            keyword = p
    return points


def extract_points(points, type, n_years):

    if type[0] == 'X':
        if type[1] == 'Y':
            x = [int(p) for p in points[0::2]]
            y = [int(p) for p in points[1::2]]
            data = [x,y]

    elif type[0] == 'R':
        if type[1] == 'Y':
            x_original = [int(p) for p in points[0:int(len(points)/2):2]]
            y_original = [int(p) for p in points[1:int(len(points)/2):2]]
            x_filtered = [int(p) for p in points[int(len(points)/2)::2]]
            y_filtered = [int(p) for p in points[int(len(points)/2)+1::2]]
            if not x_original == x_filtered:
                print("issue with years, not identical!")
                exit()
            data = [x_original, [f/o*100 for o,f in zip(y_original,y_filtered)]]

    elif type[0] == 'C':
        if type[1] == 'Y':
            t = []
            x = []
            y = []
            d = int(len(points) / (n_years * 2 + 1))
            for i in range(0, len(points)-d, n_years*2):
                t.append(points[i])
                points.pop(i)
            e = n_years * 2
            for j in range(d):
                x.append([int(p) for p in points[j*e:(j+1)*e:2]])
                y.append([int(p) for p in points[1+j*e:(j+1)*e:2]])
            for x_sub in x[1:]:
                if not x_sub == x[0]:
                    print("issue with years, not identical!")
                    exit()
            data = [t,x,y]

        elif type[1] == 'X':
            x = [p for p in points[0::2]]
            y = [int(p) for p in points[1::2]]
            if sort_items:
                sorted_data = sorted(dict(zip(x,y)).items())
                x = [p[0] for p in sorted_data]
                y = [p[1] for p in sorted_data]
            data = [x,y]

        elif type[1] == 'R':
            t = []
            y = []
            d = int(len(points) / (n_years * 2 + 1))
            for i in range(0, len(points)-d, n_years*2):
                t.append(points[i])
                points.pop(i)
            e = n_years * 2
            x = [int(p) for p in points[0:e:2]]
            for j in range(d):
                y.append([int(p) for p in points[1+j*e:(j+1)*e:2]])
            for i in range(n_years):
                y_sum = sum([y_d[i] for y_d in y])
                temp = 0
                for k in range(d):
                    y[k][i] = y[k][i] / y_sum
                    temp += y[k][i]
            data = [t,x,y]

    return data

def set_params(style):
    plt.rcParams.update(plt.rcParamsDefault)
    plt.style.use(style)
    plt.rcParams.update({'figure.autolayout': True})

def set_pie_label(data):
    def my_autopct(pct):
        if pct > 5:
            label = "{:d}\n({:.1f}%)".format(int(pct*sum(data)/100), pct)
        else:
            label = "{:d}".format(int(pct*sum(data)/100))
        return label
    return my_autopct

def set_color(n, ax):
    if n > 10:
        cm = plt.get_cmap('tab20')
        ax.set_prop_cycle(color=[cm(1.*i/20) for i in range(20)])
    elif n > 6:
        cm = plt.get_cmap('tab10')
        ax.set_prop_cycle(color=[cm(1.*i/10) for i in range(10)])


### READ DATA FROM LOGFILE ###

marker_start = '!P'
marker_end = 'P!'
pos = 0
plot_type = []
plot_data = []

while True:
    pos = log_text.find(marker_start,pos+1) + len(marker_start)
    if pos < len(marker_start):
        break
    end = log_text.find(marker_end,pos)
    plot_type.append(list(log_text[pos:pos+3]))
    plot_data.append(extract_points(read_data(log_text[pos+3:end]), plot_type[-1], n_years))


### GENERATE PLOTS ###

for type,data in zip(plot_type,plot_data):

    if type[0] == 'X': # looking at (number of) papers
        if type[1] == 'Y': # looking at progression over years
            if type[2] == 'G': # Analyse trend in terms of weight and growth
                set_params('ggplot')
                fig, ax = plt.subplots()
                ax.bar(data[0], data[1])
                slope, intercept, _, _, _ = linregress(data[0], data[1])
                ax.plot(data[0], [intercept+slope*x for x in data[0]], label='linear regression', color='blue', linewidth=4, linestyle='--')
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                ax.set_title('Rate of Publication of ' + analysis_title, pad=30)
                ax.set_xlabel('year')
                ax.set_ylabel('papers')
                ax.legend()
                fig.suptitle('(Weight: {}, Growth: {:.1f}/year)'.format(sum(data[1]), slope), x=0.54, y=0.9, size='medium', fontweight='light')
                fig.savefig(os.path.join(dir, 'plots', analysis_title + "_weight_and_growth_absolute"), dpi=300)


    elif type[0] == 'R': # looking at relative numbers between original and filtered database

        if type[1] == 'Y': # looking at progression over years
            if type[2] == 'X': # simple stating of paper ratios
                set_params('ggplot')
                fig, ax = plt.subplots()
                ax.bar(data[0], data[1])
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                y_limits = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]
                ax.set_ylim([0, min([y for y in y_limits if y > max(data[1])])])
                # ax.set_ylim([0, 10 ** math.ceil(math.log10(max(data[1])))])
                ax.set_title('Ratio of Papers containing Keywords of ' + analysis_title)
                ax.set_xlabel('year')
                ax.set_ylabel('hits [%]')
                fig.savefig(os.path.join(dir, 'plots', analysis_title + "_relative_hits_per_year"), dpi=300)

            elif type[2] == 'Y': # include 3 year trend line
                set_params('ggplot')
                fig, ax = plt.subplots()
                ax.bar(data[0], data[1])
                margin = int((running_avg_length-1) / 2)
                data_avg = [sum(data[1][max([i-margin,0]):min([i+margin+1,len(data[1])])]) / len(data[1][max([i-margin,0]):min([i+margin+1,len(data[1])])])
                            for i in range(len(data[1]))]
                ax.plot(data[0][:margin+1], data_avg[:margin+1], color='blue', linewidth=4, linestyle=':')
                ax.plot(data[0][margin:-margin], data_avg[margin:-margin], label='{} year running average'.format(running_avg_length), color='blue', linewidth=4)
                ax.plot(data[0][-margin-1:], data_avg[-margin-1:], color='blue', linewidth=4, linestyle=':')
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                y_limits = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]
                ax.set_ylim([0, min([y for y in y_limits if y > max(data[1])])])
                ax.set_title('Ratio of Papers containing Keywords of ' + analysis_title)
                ax.set_xlabel('year')
                ax.set_ylabel('hits [%]')
                ax.legend()
                fig.savefig(os.path.join(dir, 'plots', analysis_title + "_relative_hits_per_year_with_trend"), dpi=300)

            elif type[2] == 'G': # Analyse trend in terms of weight and growth
                set_params('ggplot')
                fig, ax = plt.subplots()
                ax.bar(data[0], data[1])
                slope, intercept, _, _, _ = linregress(data[0], data[1])
                ax.plot(data[0], [intercept+slope*x for x in data[0]], label='linear regression', color='blue', linewidth=4, linestyle='--')
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                y_limits = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]
                ax.set_ylim([0, min([y for y in y_limits if y > max(data[1])])])
                # ax.set_ylim(0, 12.0)
                magnitude = int('{:e}'.format(min([y for y in y_limits if y > max(data[1])]))[-3:])*-1
                ax.set_title('Ratio of Papers containing Keywords of ' + analysis_title, pad=30)
                # ax.set_title(title, pad=30)
                ax.set_xlabel('year')
                ax.set_ylabel('hits [%]')
                ax.legend()
                fig.suptitle('(Weight: {:.{}f}%, Growth: {:.{}f}%/year)'.format(np.mean(data[1]), magnitude+1, slope, magnitude+2),
                             x=0.54, y=0.9, size='medium', fontweight='light')
                fig.savefig(os.path.join(dir, 'plots', analysis_title + "_weight_and_growth_relative"), dpi=300)
                # plt.show()

    elif type[0] == 'C': # looking at composition of databases

        if type[1] == 'Y': # looking at progression over years
            # https://towardsdatascience.com/stacked-bar-charts-with-pythons-matplotlib-f4020e4eb4a7
            set_params('ggplot')
            fig, ax = plt.subplots()
            set_color(len(data[0]), ax)
            ax.bar(data[1][0], data[2][0], label=data[0][0])
            bottom = data[2][0]
            for d in range(1, len(data[0])):
                ax.bar(data[1][d], data[2][d], bottom=bottom, label=data[0][d])
                bottom = [b1+b2 for b1,b2 in zip(bottom, data[2][d])]
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            # ax.set_title('Composition of Database over years')
            ax.set_title(title)
            ax.set_xlabel('year')
            ax.set_ylabel('number of publications')
            handles, labels = ax.get_legend_handles_labels()
            # ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.0, 1.0))
            ax.legend(handles[::-1], labels[::-1], loc='upper left')
            # fig.set_size_inches(1+len(data[0])*0.5, 5)
            fig.set_size_inches(6,4)
            # ax.legend()
            fig.savefig(os.path.join(dir, 'plots', analysis_title + "_composition_over_years"), dpi=300)

        elif type[1] == 'X': # looking at overall composition
            set_params('fast')
            fig, ax = plt.subplots()
            pieces = len(data[1])
            set_color(pieces, ax)
            ax.pie(data[1], startangle=90, textprops={'color':"w", 'weight':'bold'}, labels=data[0],
                   autopct=set_pie_label(data[1]), pctdistance=min(0.8,0.5+0.03*pieces))
            ax.set_title('Composition of Database')
            ax.axis('equal')
            ax.legend(title='Source', bbox_to_anchor=(1.1, 0.8))
            fig.set_size_inches(4+pieces*0.25, 3+pieces*0.25)
            fig.savefig(os.path.join(dir, 'plots', analysis_title + "_composition"), dpi=300)

        elif type[1] == 'R': # looking at relative composition over years
            fig, ax = plt.subplots()
            ax.stackplot(data[1], data[2], labels=data[0])
            ax.set_title('Relative Composition of Database over years')
            ax.set_xlabel('year')
            ax.set_ylabel('ratio of papers')
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.2, 0.9))
            ax.set_ylim(0,1)
            ax.set_xlim(min(data[1]), max(data[1]))
            fig.set_size_inches(1+len(data[0])*0.5, 5)
            fig.savefig(os.path.join(dir, 'plots', analysis_title + "_relative_composition_over_years"), dpi=300)