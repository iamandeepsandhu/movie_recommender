# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 22:11:22 2020

@author: Amandeep Sandhu
"""

import pandas as pd

#read movie ratings file, change the path below to where ever you have downloaded that file
r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('c:/ml-100k/u.data', sep='\t', names=r_cols, usecols=range(3), encoding="ISO-8859-1")

#read another file with movie titles, change the path below to where ever you have downloaded that file
m_cols = ['movie_id', 'title']
movies = pd.read_csv('c:/ml-100k/u.item', sep='|', names=m_cols, usecols=range(2), encoding="ISO-8859-1")

#merge both dataframes
ratings = pd.merge(movies, ratings)


#make pivot table
userRatings = ratings.pivot_table(index=['user_id'],columns=['title'],values='rating')

#compute correlation score for every column
corrMatrix = userRatings.corr()

#keep movies which were rated by atleast 100 persons
corrMatrix = userRatings.corr(method='pearson', min_periods=100)

#enter user id for whom you want to find recommendations
userid=input("Enter user id: ")
userid=int(userid)
myRatings = userRatings.loc[userid].dropna()

print()
print("Movies rated by user with above user id:\n")

for i in myRatings.index:
    print(i)
    
simCandidates = pd.Series()
for i in range(0, len(myRatings.index)):
    
    # Retrieve similar movies to this one that the user rated
    sims = corrMatrix[myRatings.index[i]].dropna()
    # Now scale its similarity by how well the user rated this movie
    sims = sims.map(lambda x: x * myRatings[i])
    # Add the score to the list of similarity candidates
    simCandidates = simCandidates.append(sims)

simCandidates.sort_values(inplace = True, ascending = False)

#add together the scores from movies that show up more than once, so they'll count more
simCandidates = simCandidates.groupby(simCandidates.index).sum()

simCandidates.sort_values(inplace = True, ascending = False)

#drop movies which i have already rated(we assume if a user had rated a movie, he must have watched it)
for i in myRatings.index:
    if(i in simCandidates):
        filteredSims=simCandidates.drop(i)
        
print ("\nGetting movie recommendations for you...\n\nTitle\t\t\t\t\tSimilarity score (relative)\n")

print(filteredSims.head(10))       
