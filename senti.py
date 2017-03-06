import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize

# List of positive and negative words

pos_words=[]
neg_words=[]
with open('/home/ajinkya/Documents/NLP/pos_words.txt') as f:
			pos_words = f.read().splitlines()
with open('/home/ajinkya/Documents/NLP/neg_words.txt') as f:
			neg_words = f.read().splitlines()
# List of negation words

negation_words = ['never', 'neither', 'no', 'not']

review="he ac works like a charm, got it delivered as promised on the same day , well packed. Ac cools the room within minutes. According to my room size 1.8 to 2 was required but i took a chance and my luck it was a worthwhile buy. 10 to 12 mins the whole room gets cooled up. Go for it if you are looking for 1.5 ton 3 star. Got the product in 2days after placing order . packing was perfect.,Truly a master piece at this price range.,After initial hiccups got the installation done. Good product,installation was done within 3-4 hrs of the product delivery, installation charges are Rs 2000 and not 1500 as informed by carrier team as Rs 500 is seperately charged for iron hanging stand/unit on which the outdoor unit is installed. Cooling is superb.It was a fresh unit manufactured in mar 2016. Overall Happy and Satisfied with Amazon and Carrier.,Timely delivered and installed at given appointment by me.,Performance of product is really good.,Very bad ac.Colling is null.Noise level of both unit is very high.Don't buy.Wrost service of carrior.After 5 day my installation is done and also demo is getting another 7 day.very poor response of carrior cc,Very good product with good installation service from carrier team,Very Good AC @ reasonable price from Amazon  with 100% copper.,Satisfactory performance."



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



def main():
	
	print 'Sentiment around word "installation" is:', find_sentiment_around('instal',review)
	print 'Sentiment around word "service" is:', find_sentiment_around('service',review)
	print 'Sentiment around word "cooling" is:', find_sentiment_around('cool',review)
	print 'Sentiment around word "noise" is:', find_sentiment_around('noise',review)

main()