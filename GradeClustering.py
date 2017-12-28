# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 15:59:12 2017

@author: David
"""

import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import defaultdict

def import_xlsx():
    """Import xlsx files to concatenate into a dataframe.
    Columns represent assignments/exams, and rows represent students."""
    path = "C:\\Users\\David\\Desktop\\Teaching\\*\\*\\"
    files = []
    for file in glob.glob(path + "*.xlsx"):
        if file.split("\\")[-1].startswith("~"):
            continue
        else:
            files.append(file)
    return files

def normalize(x):
    """Min-Max Scaling for lab grades."""
    columns = x.columns[2:]
    for col in columns:
        x[col] = (x[col]-x[col].min())/(x[col].max()-x[col].min())
    return

def grade_tuples(filtered=True):
    """Sort out student grades for plotting."""
    tuples = []
    for idx in range(len(data)):
        y = data.ix[idx, 2:14]
        x = [i+1 for i in range(len(y))]
        tuples.extend([i for i in list(zip(x,y))]) #(assignment, score)
    
    if filtered:
        tuples = list(filter(lambda x: x[1] > 0.0, tuples))
    
    d = defaultdict(list) #collect assignment number as key, score as value
    for key, value in tuples:
        d[key].append(value)
    
    x, y = zip(*tuples) # unpack (assignment, score) into two lists
    
    means = [(key, np.mean(d[key])) for key in d] # average score per assignment
    means = sorted(means, key = lambda x: x[1])
    mean_x, mean_y = zip(*means) # unpack (assignment, average score) into two lists
    return x, y, mean_x, mean_y

def score_plot(save=False, filtered=True):
    """Plot grades over the course of each semester."""
    font = {"fontname":"Helvetica", "fontsize":16}
    plt.figure(figsize=(10,8))
    plt.title("Assignment Scores", **font)
    plt.xlabel("Assignment / Week", **font)
    plt.ylabel("Score", **font)
    x, y, mean_x, mean_y = grade_tuples(filtered)
    plt.xticks(x) # location, labels
    plt.scatter(x, y, color="blue", edgecolor="white")
    plt.scatter(mean_x, mean_y, color="red", s=150)
    if save:
        if filtered:
            plt.savefig("filtered_scores.png", format="png", dpi=300)
        else:
            plt.savefig("scores.png", format="png", dpi=300)
        

def difficulty_plot(save=False, filtered=True):
    """Rank assignments by average grade."""
    font = {"fontname":"Helvetica", "fontsize":16}
    plt.figure(figsize=(10,8))
    plt.title("Ranked Assignments", **font)
    plt.xlabel("Assignment", **font)
    plt.ylabel("Mean Score", **font)
    _, _, _, mean_y = grade_tuples(filtered)
    x = [i for i in range(12)]
    if filtered:
        assignments = ["Sonometer", "Acceleration", "Oscillations",
                       "Pressure", "Pos. & Vel.", "Sig. Figures",
                       "Momentum", "Energy", "Vectors", "Forces",
                       "Work & Heat", "Calorimetry"]
        plt.ylim([0.8, 1.0])
        plt.xticks(x, assignments, rotation='45')
        plt.bar(x, mean_y, color="blue", alpha=0.65)
        plt.tight_layout()
        if save:
            plt.savefig("filtered_ranks.png", format="png", dpi=300)
    else:
        assignments = ["Calorimetry", "Pressure", "Sonometer", 
                       "Oscillations", "Work & Heat", "Sig. Figures", 
                       "Acceleration", "Momentum", "Energy", "Pos. & Vel.", 
                       "Vectors", "Forces"]
        plt.ylim([0.1, 1.0])
        plt.xticks(x, assignments, rotation='45')
        plt.bar(x, mean_y, color="red", alpha=0.65)
        plt.tight_layout()
        if save:
            plt.savefig("ranks.png", format="png", dpi=300)

def mean_variance_plot(save=False):
    """Assignment Variance vs. Grade"""
    font = {"fontname":"Helvetica", "fontsize":16}
    plt.figure(figsize=(10,8))
    plt.title("Assignment Variance vs. Grade", **font)
    plt.xlabel("Grade", **font)
    plt.ylabel("Variance", **font)
    x = data['Average'].values
    y = data['Variance'].values
    plt.scatter(x, y, color="green", s=100, edgecolor="white")
    plt.tight_layout()
    if save:
        plt.savefig("variance_grade.png", format="png", dpi=300)

def grade_clustering():
    """k-clustering of student groups."""
    #define the data for clustering
    x = data['Average'].values
    y = data['Variance'].values
    points = list(zip(x,y))
    
    # initialize clusters
    kmeans = KMeans(n_clusters=6)
    
    # fit clusters to x,y data
    kmeans = kmeans.fit(points)
    
    # predict labels for clusters
    labels = kmeans.predict(points)
    
    # obtain cluster centroids
    centroids = kmeans.cluster_centers_
    return x, y, labels, centroids

def cluster_plot(save=False):    
    """Plot clusters"""
    x, y, labels, centroids = grade_clustering()
    
    font = {"fontname":"Helvetica", "fontsize":16}
    plt.figure(figsize=(10,8))
    plt.title("Grade Clusters", **font)
    plt.xlabel("Grade", **font)
    plt.ylabel("Variance", **font)
    plt.scatter(x, y, edgecolor="white", s=150, c=labels)
    plt.scatter(centroids[:,0], centroids[:,1], 
                marker="*", edgecolor="white", 
                c=np.arange(len(centroids)), s=1250)
    plt.tight_layout()
    if save:
        plt.savefig("variance_grade.png", format="png", dpi=300)
                
if __name__=="__main__":
    
    """Data importing"""
    files = import_xlsx() # import csv files then concatenate them
    data = pd.concat((pd.read_excel(f, header=0, usecols=range(0,14)) for f in files))
    
    """Data processing"""
    data = data.fillna(0) # fill nan data with zeroes
    data['Average'] = data.mean(axis=1) # make a column of averages
    data['Variance'] = data.var(axis=1)
    data = data[~(data['Average'] == 0)] # ignore students who dropped before the first day
    data = data.sort("Student Name").reset_index(drop=True)
    normalize(data) # MinMax Scaling of grades.
    
    """Data plotting"""
    score_plot(save=True, filtered=True) #no zeroes or missed assignments included
    difficulty_plot(save=True, filtered=True)
    
    score_plot(save=True, filtered=False) #zeroes and missed assignments included
    difficulty_plot(save=True, filtered=False)
    
    mean_variance_plot(save=True)
    
    cluster_plot(save=True)
