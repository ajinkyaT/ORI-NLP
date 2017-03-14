import numpy as np
import pandas as pd
from math import sqrt

df=pd.read_csv('/home/ajinkya/Documents/NLP/final1.csv')
sentiment_columns = ['service','installation','cooling']

def wilson_score_confidence(ups, downs):
	
	# Total votes
	n = ups + downs

	# If total votes is 0, return 0
	if n == 0:
		return 0

	# Calculate Wilson score
	# 1.44 = 85%, 1.96 = 95%
	z = 1.0 
	phat = float(ups) / n
	return ((phat + z*z/(2*n) - z * sqrt((phat*(1-phat)+z*z/(4*n))/n))/(1+z*z/n))

def calculate_wilson_score(column_positive, column_negative):
	

	wilson_scores = np.array([0.0] * len(df))

	for i in range(len(df)):
		wilson_scores[i] = wilson_score_confidence(column_positive[i], column_negative[i])
	return wilson_scores

def append_column(column, column_name):
	global df
	s1 = pd.Series(column, name=column_name)
	df = pd.concat([df, s1], axis=1)

def calcute_all_wilson_scores_and_append_result():
	

	# Service, installation and cooling - Wilson score
	for i in range(len(sentiment_columns)):
		column_positive_name = 'good_' + sentiment_columns[i]
		column_negative_name = 'bad_' + sentiment_columns[i]
		wilson_scores = calculate_wilson_score(
											   		df[[column_positive_name]].values,
											   		df[[column_negative_name]].values)
		wilson_column_name = sentiment_columns[i] + '_wilson_score'
		append_column(wilson_scores, wilson_column_name)

	# Reviews - Wilson score	
	column_positive = df.star_5.values.astype(float) + df.star_4.values.astype(float)
	column_negative = df.star_3.values.astype(float) + df.star_2.values.astype(float) + df.star_1.values.astype(float)
	column_positive[np.isnan(column_positive)] = 0
	column_negative[np.isnan(column_negative)] = 0
	wilson_scores = calculate_wilson_score(
												column_positive,
												column_negative)
	append_column(wilson_scores, 'reviews_wilson_score')
	df.to_csv("final2.csv", index=False)

calcute_all_wilson_scores_and_append_result()