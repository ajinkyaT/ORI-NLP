# ORI-NLP

Using NLP for sentiment analysis

## Code

Open [senti.py](../master/senti.py)

## Code implementation

Change file path of  pos_words.txt and neg_words.txt in code as shown in below:

```python 
pos_words=[]
neg_words=[]
with open('/home/ajinkya/Documents/NLP/pos_words.txt') as f:
			pos_words = f.read().splitlines()
with open('/home/ajinkya/Documents/NLP/neg_words.txt') as f:
			neg_words = f.read().splitlines()
```
Also you may change the review in review field

# Test Code
Run in terminal 
```python 
python senti.py
```
