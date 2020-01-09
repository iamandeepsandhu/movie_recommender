# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 22:11:22 2020

@author: Amandeep Sandhu
"""

import pandas as pd

r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('c:/MLCourse/ml-100k/u.data', sep='\t', names=r_cols, usecols=range(3), encoding="ISO-8859-1")

m_cols = ['movie_id', 'title']
movies = pd.read_csv('c:/MLCourse/ml-100k/u.item', sep='|', names=m_cols, usecols=range(2), encoding="ISO-8859-1")

ratings = pd.merge(movies, ratings)



userRatings = ratings.pivot_table(index=['user_id'],columns=['title'],values='rating')


corrMatrix = userRatings.corr()


corrMatrix = userRatings.corr(method='pearson', min_periods=100)

userid=input("Enter user id: ")
userid=int(userid)
myRatings = userRatings.loc[userid].dropna()

print()
print("Movies rated by user with above user id:\n")

for i in myRatings.index:
    print(i)
    
simCandidates = pd.Series()
for i in range(0, len(myRatings.index)):
    
    # Retrieve similar movies to this one that I rated
    sims = corrMatrix[myRatings.index[i]].dropna()
    # Now scale its similarity by how well I rated this movie
    sims = sims.map(lambda x: x * myRatings[i])
    # Add the score to the list of similarity candidates
    simCandidates = simCandidates.append(sims)
    
#Glance at our results so far:
print ("\nGetting movie recommendations for you...\n\nTitle\t\t\t\t\tSimilarity score (relative)\n")
simCandidates.sort_values(inplace = True, ascending = False)


simCandidates = simCandidates.groupby(simCandidates.index).sum()

simCandidates.sort_values(inplace = True, ascending = False)


for i in myRatings.index:
    if(i in simCandidates):
        filteredSims=simCandidates.drop(i)
 
print(filteredSims.head(10))       