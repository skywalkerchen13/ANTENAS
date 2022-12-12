# author: Lukas Preis
# date: 16.11.2020
# version: 1.1.1
# update: 3 year trend line included in year overview

import pandas as pd
import numpy as np
from statistics import *
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import heapq
from itertools import dropwhile
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# https://matplotlib.org/3.1.1/gallery/style_sheets/style_sheets_reference.html

# my_title = 'UAM ARC'
filename = "Logfile.txt"
analysis_title = "Aviation"
token_length = 2
word_clouds = False
full_name = True
ranks_base = 'mixed' # options: all_time_top | year_winners | mixed
running_avg_length = 5
sort_items = False

dir = os.path.dirname(os.path.dirname(__file__))
file = os.path.join(dir, 'logfiles', filename)
log_text = open(file,'r').read()

# papers per year all times (XYX): vertical bar chart (mark most and least active)
# papers per year all times with 3 year trend (XYY): vertical bar chart with trend line
# papers per year as accumulated percentage (XYR): vertical bar chart
# papers of author per year (AYP): vertical bar chart
# top keywords all times in title (KXT): horizontal bar chart
# top keywords all times in abstract (KXA): horizontal bar chart
# top keywords all times of person/author (KXP): horizontal bar chart
# top keywords per year in title (KYT): line plot with ranks
# top keywords per year in abstract (KYA): line plot with ranks
# top authors all times (AXX): horizontal bar chart
# favorite co-authors of author (AXP): horizontal bar chart
# top authors per year (AYX): line plot with ranks
# composition of database total (CXX): pie chart



### FUNCTIONS ###

def prep_data(text,type,years):

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

    # for i in range(int(len(broken_text)/(token_length+1))):
    #     words = broken_text[i*(token_length+1):i*(token_length+1)+token_length]
    #     count = broken_text[i*(token_length+1)+token_length]
    #     points.append(" ".join(words))
    #     points.append(count)

    if type[0] == 'X':
        if type[1] == 'Y':
            x = [int(p) for p in points[0::2]]
            y = [int(p) for p in points[1::2]]
            if type[2] == 'R':
                x.reverse()
                y.reverse()
                y = np.cumsum([y_i/sum(y)*100 for y_i in y]).tolist()
            data = [x,y]

    elif type[0] == 'K':
        if type[1] == 'X':
            x = [p for p in points[0::2]]
            y = [int(p) for p in points[1::2]]
            data = [x,y]
        elif type[1] == 'Y':
            t = []
            x = []
            y = []
            d = int(len(points)/years)-1
            for p in points:
                if p.isnumeric() and len(p) == 4:
                    t.append(int(p))
                    x.append([])
                    y.append([])
                elif p.isnumeric():
                    y[-1].append(int(p))
                else:
                    x[-1].append(p)
            # for i in range(0,len(points)-years,d):
            #     t.append(int(points[i]))
            #     points.pop(i)
            # for j in range(years):
            #     x.append([p for p in points[j*d:(j+1)*d:2]])
            #     y.append([p for p in points[1+j*d:(j+1)*d:2]])
            data = [t,x,y]

    elif type[0] == 'A':
        if type[1] == 'X':
            x = [p.replace(".",". ") for p in points[0::2]]
            y = [int(p) for p in points[1::2]]
            data = [x,y]
        elif type[1] == 'Y':
            if type[2] == 'X':
                t = [p for p in points[0::3]]
                x = [p.replace(".",". ") for p in points[1::3]]
                y = [int(p) for p in points[2::3]]
                data = [t,x,y]
            elif type[2] == 'P':
                if full_name:
                    a = snippets[0].replace("."," ")
                else:
                    a = snippets[0].replace(".",". ")
                points = list(dropwhile(lambda x: x != 'and', points))
                points.pop(0)
                points.pop(0)
                x = [int(p) for p in points[0::2]]
                y = [int(p) for p in points[1::2]]
                data = [x,y,a]

    elif type[0] == 'C':
        x = [p for p in points[0::2]]
        y = [int(p) for p in points[1::2]]
        if sort_items:
            sorted_data = sorted(dict(zip(x,y)).items())
            x = [p[0] for p in sorted_data]
            y = [p[1] for p in sorted_data]
        data = [x,y]

    return data

def set_params(style):
    plt.rcParams.update(plt.rcParamsDefault)
    plt.style.use(style)
    plt.rcParams.update({'figure.autolayout': True})

def create_word_cloud(x,y):
    d = dict(zip(x,y))
    wordcloud = WordCloud(background_color="white").generate_from_frequencies(frequencies=d)
    plt.close('all')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(os.path.join(dir, 'plots', analysis_title + "_top_keywords_wordcloud_" + source.replace(" ","_")), dpi=1000)

def set_color(n, ax):
    if n > 10:
        cm = plt.get_cmap('tab20')
        ax.set_prop_cycle(color=[cm(1.*i/20) for i in range(20)])
    elif n > 6:
        cm = plt.get_cmap('tab10')
        ax.set_prop_cycle(color=[cm(1.*i/10) for i in range(10)])

def set_pie_label(data):
    def my_autopct(pct):
        if pct > 5:
            label = "{:d}\n({:.1f}%)".format(int(round(pct*sum(data)/100,0)), pct)
        else:
            label = "{:d}".format(int(round(pct*sum(data)/100,0)))
        return label
    return my_autopct



### READ DATA FROM LOGFILE ###

marker_start = '!P'
marker_end = 'P!'
temp_pos = log_text.find('Publications between years ') + len('Publications between years ')
year_range = [int(log_text[temp_pos:temp_pos+4]),int(log_text[temp_pos+9:temp_pos+13])]
temp_pos = log_text.find('most active: ') + len('most active: ')
year_max = int(log_text[temp_pos:temp_pos+4])
temp_pos = log_text.find('least active: ') + len('least active: ')
year_min = int(log_text[temp_pos:temp_pos+4])
plot_type = []
plot_data = []
pos = 0

while True:
    pos = log_text.find(marker_start,pos+1) + len(marker_start)
    if pos < len(marker_start):
        break
    end = log_text.find(marker_end,pos)
    plot_type.append(list(log_text[pos:pos+3]))
    plot_data.append(prep_data(log_text[pos+3:end],plot_type[-1],year_range[1]-year_range[0]+1))



### GENERATE PLOTS ###

for type,data in zip(plot_type,plot_data):

    if type[0] == 'X': # looking at (number of) papers

        if type[2] == 'X': # simple stating of papers
            set_params('ggplot')
            temp_fig, temp_ax = plt.subplots()
            marked_extremes = ['r']*len(data[0])
            marked_extremes[data[0].index(year_min)] = 'b'
            marked_extremes[data[0].index(year_max)] = 'g'
            labeled_extrems = [None]*len(data[0])
            labeled_extrems[data[0].index(year_min)] = data[1][data[0].index(year_min)]
            labeled_extrems[data[0].index(year_max)] = data[1][data[0].index(year_max)]
            temp_ax.bar(data[0],data[1],color=marked_extremes)
            # temp_fig.text(data[1][data[0].index(year_min)],data[0].index(year_min),str(data[0].index(year_min)))
            temp_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            temp_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            temp_ax.set_title('Rate of Publication')
            # temp_ax.set_title('Publications related to "{}"'.format(analysis_title))
            temp_ax.set_xlabel('year')
            # temp_ax.set_ylabel('papers')
            temp_ax.set_ylabel('number of publications')
            temp_fig.savefig(os.path.join(dir, 'plots', analysis_title + "_paper_per_year"), dpi=300)

        elif type[2] == 'Y': # include 3 year trend line
            set_params('ggplot')
            temp_fig, temp_ax = plt.subplots()
            temp_ax.bar(data[0], data[1])
            # temp = [sum(data[1][i-1:i+2])/3 for i in [1,2,3]]
            margin = int((running_avg_length-1) / 2)
            # data_avg = [sum(data[1][:2])/2] + [sum(data[1][i-1:i+2])/3 for i in range(1, len(data[1])-1)] + [sum(data[1][-2:])/2]
            # data_avg = [0]*margin + [sum(data[1][i-margin:i+margin+1])/running_avg_length for i in range(margin, len(data[1])-margin)] + [0]*margin
            data_avg = [sum(data[1][max([i-margin,0]):min([i+margin+1,len(data[1])])]) / len(data[1][max([i-margin,0]):min([i+margin+1,len(data[1])])])
                        for i in range(len(data[1]))]
            # temp_ax.plot(data[0][:2], data_avg[:2], color='blue', linewidth=4, linestyle=':')
            temp_ax.plot(data[0][:margin+1], data_avg[:margin+1], color='blue', linewidth=4, linestyle=':')
            temp_ax.plot(data[0][margin:-margin], data_avg[margin:-margin], label='{} year running average'.format(running_avg_length), color='blue', linewidth=4)
            temp_ax.plot(data[0][-margin-1:], data_avg[-margin-1:], color='blue', linewidth=4, linestyle=':')
            # temp_ax.plot(data[0][-2:], data_avg[-2:], color='blue', linewidth=4, linestyle=':')
            temp_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            temp_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            temp_ax.set_title('Rate of Publication')
            temp_ax.set_xlabel('year')
            temp_ax.set_ylabel('number of publications')
            temp_ax.legend()
            temp_fig.savefig(os.path.join(dir, 'plots', analysis_title + "_paper_per_year_with_trend"), dpi=300)

        elif type[2] == 'R': # accumulative distribution of papers
            set_params('ggplot')
            temp_fig, temp_ax = plt.subplots()
            temp_ax.bar(data[0], data[1])
            temp_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            temp_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            temp_ax.set_ylim(0,100)
            temp_ax.set_title('Accumulated Reverse Percentage of Publications')
            temp_ax.set_xlabel('year')
            temp_ax.set_ylabel('accumulated publications [%]')
            temp_fig.savefig(os.path.join(dir, 'plots', analysis_title + "_accumulated_paper_per_year"), dpi=300)

    elif type[0] == 'K': # looking at keywords (and papers where they occur)
        if type[1] == 'X': # looking at entire time period
            set_params('fivethirtyeight')
            temp_fig, temp_ax = plt.subplots()
            # temp_fig.tight_layout()
            temp_ax.barh(data[0],data[1])
            temp_ax.invert_yaxis()
            if type[2] == 'T': # looking at titles
                source = 'in title'
            elif type[2] == 'A': # looking at abstracts
                source = 'in abstract'
            elif type[2] == 'P': # looking at a person/author
                source = 'of ' + author_label
            temp_ax.set_title('Top Keywords ({} - {})'.format(year_range[0],year_range[1]))
            temp_ax.set_xlabel('occurence')
            temp_ax.set_ylabel('keyword ' + source)
            temp_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            temp_fig.set_size_inches(4+2*token_length, 2+len(data[0])/2)
            temp_fig.savefig(os.path.join(dir, 'plots', analysis_title + "_top_keywords_" + source.replace(" ","_")), dpi=300)
            if word_clouds:
                create_word_cloud(data[0],data[1])
        elif type[1] == 'Y': # looking at progression over years
            set_params('seaborn-deep')
            temp_fig, temp_ax = plt.subplots()
            keywords = {}
            occurences = []
            ranks = []
            temp_keyword = []
            i = 0
            if ranks_base == 'all_time_top':
                temp_k = plot_data[plot_type.index(['K', 'X', type[2]])][0]
                for k in temp_k:
                    keywords[k] = None
            elif ranks_base == 'year_winners':
                while len(keywords.keys()) < 20 and len(data[1][0]) > i:
                    for k,n in zip(data[1],data[2]):
                        if int(n[i]) > 1:
                            keywords[k[i]] = None
                    i += 1
            elif ranks_base == 'mixed':
                temp_k = plot_data[plot_type.index(['K', 'X', type[2]])][0]
                for k in temp_k:
                    keywords[k] = None
                while len(keywords.keys()) < 20:
                    for k,n in zip(data[1],data[2]):
                        if int(n[i]) > 1:
                            keywords[k[i]] = None
                    i += 1
            for keyword in keywords.keys():
                # n_years = year_range[1]-year_range[0]+1
                n_years = len(data[0])
                n_occurence = [0]*n_years
                rank = [None]*n_years
                for n in range(n_years):
                    if keyword in data[1][n]:
                        n_occurence[n] = int(data[2][n][data[1][n].index(keyword)])
                        rank[n] = data[1][n].index(keyword)+1
                occurences.append(mean(n_occurence))
                ranks.append(rank)
                temp_keyword.append(keyword)
            cm = plt.get_cmap('tab20')
            # n_keywords = len(keywords.keys())
            temp_ax.set_prop_cycle(color=[cm(1.*i/20) for i in range(20)])
            i_top = heapq.nlargest(20, range(len(occurences)), key=occurences.__getitem__)
            # drop indeces that are not in t_top; then go through all remaining ranks>
            top_ranks = []
            top_keywords = []
            for i in i_top:
                top_ranks.append(ranks[i])
                top_keywords.append(temp_keyword[i])
            for rank,keyword in zip(top_ranks,top_keywords):
                temp_ax.plot(data[0], rank, label=keyword, marker='s', linestyle='--')
            if type[2] == 'T': # looking at titles
                source = 'title'
            elif type[2] == 'A': # looking at abstracts
                source = 'abstract'
            temp_ax.invert_yaxis()
            temp_ax.set_title('Top Keywords over Years')
            temp_ax.set_xlabel('year')
            temp_ax.set_ylabel('ranks in ' + source)
            temp_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            temp_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            temp_ax.yaxis.grid()
            temp_ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
                           ncol=4, fancybox=True, shadow=True)
            temp_fig.set_size_inches(10, 5)
            temp_fig.savefig(os.path.join(dir, 'plots', analysis_title + "_keywords_ranks_in_" + source), dpi=300)

    elif type[0] == 'A': # looking at authors (and the amount of papers they published)
        if type[1] == 'Y': # looking at the publication history
            if type[2] == 'P': # looking at one author
                set_params('ggplot')
                temp_fig, temp_ax = plt.subplots()
                temp_ax.bar(data[0],data[1])
                author_name = data[2]
                temp_ax.set_title('Publications of ' + author_name)
                temp_ax.set_xlabel('year')
                temp_ax.set_ylabel('papers')
                temp_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                temp_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                if full_name:
                    author_label = author_name.split(" ")[-1]
                else:
                    author_label = author_name[3:]
                temp_fig.savefig(os.path.join(dir, 'plots', analysis_title + "_publications_" + author_label), dpi=300)
            elif type[2] == 'X': # looking at progression of over years
                set_params('seaborn-deep')
                temp_fig, temp_ax = plt.subplots()
                temp_ax.bar(data[0], data[2])
                temp_ax.set_title('Top Authors over Years')
                temp_ax.set_ylabel('papers')
                temp_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                temp_ax.yaxis.grid()
                temp_ax.set_xticklabels([a + " - " + str(y) for y,a in zip(data[0],data[1])], rotation=60, ha='right')
                temp_fig.set_size_inches(10, 4)
                temp_fig.savefig(os.path.join(dir, 'plots', analysis_title + "_authors_ranks"), dpi=300)
        elif type[1] == 'X': # looking at entire time period
            if type[2] == 'X': # looking at top authors in general
                set_params('fivethirtyeight')
                temp_fig, temp_ax = plt.subplots()
                temp_ax.barh(data[0],data[1])
                temp_ax.invert_yaxis()
                temp_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                temp_ax.set_title('Top Authors ({} - {})'.format(year_range[0],year_range[1]))
                # temp_ax.set_title('Top Authors "{}" ({} - {})'.format(analysis_title, year_range[0], year_range[1]))
                temp_ax.set_xlabel('publications')
                temp_fig.set_size_inches(6, 7)
                temp_fig.set_size_inches(6+int(full_name), 2+len(data[0])/2)
                temp_fig.savefig(os.path.join(dir, 'plots', analysis_title + "_top_authors"), dpi=300)
            elif type[2] == 'P': # looking at favorite co-authors
                set_params('fivethirtyeight')
                temp_fig, temp_ax = plt.subplots()
                temp_ax.barh(data[0],data[1])
                temp_ax.invert_yaxis()
                temp_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                temp_ax.set_title('Co-Authors of ' + author_name)
                temp_ax.set_xlabel('shared papers')
                temp_fig.savefig(os.path.join(dir, 'plots', analysis_title + "_coauthors_" + author_name[3:]), dpi=300)

    elif type[0] == 'C': # looking at composition of databases
        set_params('fast')
        fig, ax = plt.subplots()
        pieces = len(data[1])
        set_color(pieces, ax)
        fig.set_size_inches(10, 3+pieces*0.25)
        ax.pie([d for d in data[1]], startangle=90, textprops={'color':"w", 'weight':'bold'},
               labels=[d.replace('_',' ') for d in data[0]],
               autopct=set_pie_label(data[1]), pctdistance=min(0.8,0.5+0.03*pieces))
        ax.set_title('Composition of Database')
        ax.legend(title='Journals', bbox_to_anchor=(1.1, 0.8))
        # ax.legend(title='Sources', bbox_to_anchor=(0.9, 0.8))
        ax.axis('equal')
        # fig.set_size_inches(4+pieces*0.25, 3+pieces*0.25)
        fig.savefig(os.path.join(dir, 'plots', analysis_title + "_composition"), dpi=300)


# plt.show()