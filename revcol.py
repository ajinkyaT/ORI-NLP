import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize


pos_words=[]
neg_words=[]
with open('/home/ajinkya/Documents/NLP/pos_words.txt') as f:
			pos_words = f.read().splitlines()
with open('/home/ajinkya/Documents/NLP/neg_words.txt') as f:
			neg_words = f.read().splitlines()
# List of negation words

negation_words = ['never', 'neither', 'no', 'not']
sentiment_word_list = ['service', 'instal', 'cool', 'noi']
df=pd.read_csv('/home/ajinkya/Documents/NLP/amazon_kora_1.csv')
reviews=df[['Review_1','Review_2','Review_3','Review_4','Review_5','Review_6','Review_7','Review_8']].values.tolist()
for i in range(len(reviews)):
	for j in range(len(reviews[i])):
		reviews[i][j]=str(reviews[i][j])

def is_in_review(word, tokenized_sent_review):
	indices = {'Sentence': None}

	# For each element in sentence tokenized review, if word found return it's index
	for sent in tokenized_sent_review:
		if word in sent:
			indices['Sentence'] = tokenized_sent_review.index(sent)

	return indices['Sentence']

def find_sentiment_around(word, review):
	# Lower case all characters of review
	review = review.lower()

	# Replace comma with space
	review = review.replace(',', ' ')

	# Tokenize review into list of sentences
	tokenized_sent_review = sent_tokenize(review)

	# Check whether sent is found in review
	index_of_sent = is_in_review(word, tokenized_sent_review)
	# If not found, return False
	if index_of_sent is None:
		return None

	# Word tokenized text containing mentioned word
	tokenized_review = word_tokenize(tokenized_sent_review[index_of_sent])

	# If word is 'noise' and word is present in reviews, 
	# return negative sentiment 
	if word == 'noise':
		for w in tokenized_review:
			if word in w:
				sentiment = 'negative'
				return sentiment

	# Find index of word in tokenized review
	index_of_word = None
	for w in tokenized_review:
		if word in w:
			index_of_word = tokenized_review.index(w)

	# Get list of indices of positive & negative words in tokenized review
	postive_word_indices = []
	negative_word_indices = []
	for i in range(len(tokenized_review)):
		if i != index_of_word:
			if tokenized_review[i] in pos_words:
				postive_word_indices.append(i)
			elif tokenized_review[i] in neg_words:
				negative_word_indices.append(i)
	
	# Get closest positive & negative word's index to given word
	closest_positive_word_index = None
	closest_negative_word_index = None
	if len(postive_word_indices) is not 0:
		closest_positive_word_index = min(x for x in postive_word_indices if x is not None)
	if len(negative_word_indices) is not 0:
		closest_negative_word_index = min(x for x in negative_word_indices if x is not None)
	
	# If no positive & negative words exits near given word
	if closest_positive_word_index == closest_negative_word_index == None:
		sentiment = 'neutral'
		return sentiment

	# If only negative word found near given word
	if closest_negative_word_index is not None and closest_positive_word_index is None:
		# If a negation word is followed by negative word, then sentiment is positive
		# and negative otherwise
		if tokenized_review[closest_negative_word_index] in negation_words and closest_negative_word_index < len(tokenized_review)-1:
			if tokenized_review[closest_negative_word_index+1] in neg_words:
				sentiment = 'positive'
			else:
				sentiment = 'negative'
		else:
			sentiment = 'negative'

		return sentiment

	# If only positive word found near given word	
	elif closest_positive_word_index is not None and closest_negative_word_index is None:
		# If positive word is preceeded by negative word 
		# and index of negative word is less that index of given word
		if closest_positive_word_index < index_of_word and tokenized_review[closest_positive_word_index-1] in negation_words:
			sentiment = 'negative'
		else:
			sentiment = 'positive'

		return sentiment

	# If positive word is closer to given word
	if closest_positive_word_index < closest_negative_word_index:
		# If positive word is preceeded by negative word 
		# and index of negative word is less that index of given word
		if closest_positive_word_index < index_of_word and tokenized_review[closest_positive_word_index-1] in negation_words:
			sentiment = 'negative'
		else:
			sentiment = 'positive'

		return sentiment

	# If negative word is closer to given word
	else:
		# If a negation word is followed by negative word, then sentiment is positive
		# and negative otherwise
		if tokenized_review[closest_negative_word_index] in negation_words and closest_negative_word_index < len(tokenized_review)-1:
			if tokenized_review[closest_negative_word_index+1] in neg_words:
				sentiment = 'positive'
			else:
				sentiment = 'negative'
		else:
			sentiment = 'negative'

		return sentiment

def find_brand_indices():
		"""

		Use
		---
		Find starting and ending index of each brand in database

		Parameters
		----------
		None

		Returns
		-------
		start_indices : numpy array (int)
			Starting index of each brand in database

		end_indices : numpy array (int)
			Ending index of each brand in database

		"""

		# Brand in record of current iteration through dataframe
		current_brand = df.loc[0, 'Brand']

		# List of starting index of each brand in dataframe
		start_indices = np.array([0])

		# List of ending index of each brand in dataframe
		end_indices = np.array([])

		# Loop through each record to find starting index of each brand
		for i in range(1, len(df)):
			if df.loc[i, 'Brand'] != current_brand:
				start_indices = np.append(start_indices, i)
				current_brand = df.loc[i, 'Brand']

		# Loop through starting indices of each brand to find ending indices
		for i in range(len(start_indices)):
			# If it is last brand
			if i == len(start_indices)-1:
				# Ending index of last brand is index of last record in dataframe
				end_indices = np.append(end_indices, len(df)-1)
			else:
				# Ending index of current brand is index prior to starting index of next brand
				end_indices = np.append(end_indices, start_indices[i+1]-1)


		# Return starting and ending indices of all brands
		return start_indices.astype(int), end_indices.astype(int)

def find_sentiment_in_given_reviews( word, start_index, end_index):
		

	positive_sentiment_count = 0
	negative_sentiment_count = 0

	# Extract reviews withing start index and end index
	revs = reviews[start_index:end_index+1]

	# For each review, check for sentiment about given word
	for i in range(len(revs)):
		####
		for j in range(len(revs[i])):
			sentiment = find_sentiment_around(word, revs[i][j])
			if sentiment is 'negative':
				negative_sentiment_count += 1
			elif sentiment is 'positive':
				positive_sentiment_count += 1

	return positive_sentiment_count, negative_sentiment_count

def find_all_sentiments():
	global df

	# Find start and end indices of brand in dataframe
	start_indices, end_indices = find_brand_indices()

	# For each word in sentiment word list which consists of
	# service, installation, cooling and noise
	good_service = np.array([0] * len(df))
	bad_service = np.array([0] * len(df))
	good_installation = np.array([0] * len(df))
	bad_installation = np.array([0] * len(df))
	good_cooling = np.array([])
	bad_cooling = np.array([])
	noise = np.array([])
	for i in range(len(sentiment_word_list)):

		# Service - at Brand level
		if i == 0: 
			for j in range(len(start_indices)):
				num_good_service, num_bad_service = find_sentiment_in_given_reviews(sentiment_word_list[i], int(start_indices[j]), int(end_indices[j]))
				for n in range(int(start_indices[j]), int(end_indices[j])+1):
					good_service[n] = num_good_service
					bad_service[n] = num_bad_service

				# self.__good_service = np.append(self.__good_service, good_service)
				# self.__bad_service = np.append(self.__bad_service, bad_service)

		# Installation - at Brand level
		elif i == 1:
			for j in range(len(start_indices)):
				num_good_installation, num_bad_installation = find_sentiment_in_given_reviews(sentiment_word_list[i], int(start_indices[j]), int(end_indices[j]))
				for n in range(int(start_indices[j]), int(end_indices[j])+1):
					good_installation[n] = num_good_installation
					bad_installation[n] = num_bad_installation

		# Cooling - at Product level
		elif i == 2:
			for j in range(len(df)):
				num_good_cooling, num_bad_cooling = find_sentiment_in_given_reviews(sentiment_word_list[i], j, j)
				good_cooling = np.append(good_cooling, num_good_cooling)
				bad_cooling = np.append(bad_cooling, num_bad_cooling)
		
		# Noise - at Product level
		elif i == 3:
			# Extract reviews withing start index and end index
			for j in range(len(df)):
				temp, num_noise = find_sentiment_in_given_reviews(sentiment_word_list[i], j, j)
				noise = np.append(noise, num_noise)
	s1 = pd.Series(good_service, name='good_service')
	df = pd.concat([df, s1], axis=1)
	s1 = pd.Series(bad_service, name='bad_service')
	df = pd.concat([df, s1], axis=1)
	s1 = pd.Series(good_installation, name='good_installation')
	df = pd.concat([df, s1], axis=1)
	s1 = pd.Series(bad_installation, name='bad_installation')
	df = pd.concat([df, s1], axis=1)
	s1 = pd.Series(good_cooling, name='good_cooling')
	df = pd.concat([df, s1], axis=1)
	s1 = pd.Series(bad_cooling, name='bad_cooling')
	df = pd.concat([df, s1], axis=1)
	s1 = pd.Series(noise, name='noise')
	df = pd.concat([df, s1], axis=1)
	df.to_csv("improved"+".csv", index=False)

find_all_sentiments()