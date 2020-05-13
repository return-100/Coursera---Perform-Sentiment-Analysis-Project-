import re
import nltk
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


# Task 1: data read
data = pd.read_csv("IMDb_Reviews.csv")
print(data['sentiment'][0])
print(data['review'][0])


# Task 2: Bag of words
count = CountVectorizer()
docs = np.array(['The sun is shining',
                 'The weather is sweet',
                 'The sun is shining, the weather is sweet and one and one is two'])
bag = count.fit_transform(docs)
print(count.vocabulary_)
print(bag.toarray())


# Task 3: term frequency-inverse document frequency
np.set_printoptions(2)
tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
tfidf_arr = tfidf.fit_transform(bag)
print(tfidf_arr.toarray())


# Task 4: data preparation
def process(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    return text

data['review'] = data['review'].apply(process)


# Task 5: Tokenization of document
porter = PorterStemmer()

def tokenizer(text):
    return text.split()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

print(tokenizer('runners like running and thus they run'))
print(tokenizer_porter('runners like running and thus they run'))
nltk.download('stopwords')
stop = stopwords.words('english')

for w in tokenizer_porter('runners like running and thus they run in rain'):
    if w not in stop:
        print(w)


# Task 6: Transform actual review data to tf-idf vector
tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None,
                        tokenizer=tokenizer_porter,
                        use_idf=True,
                        norm='l2',
                        smooth_idf=True)
y = data.sentiment.values
x = tfidf.fit_transform(data.review)


# Task 7: Document classification using logistic regression
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.5, shuffle=False)
clf = LogisticRegressionCV(cv=5,
                           random_state=0,
                           n_jobs=-1,
                           verbose=3,
                           max_iter=300).fit(x_train, y_train)
saved_model = open('saved_model.sav', 'wb')
pickle.dump(clf, saved_model)
saved_model.close()


# Task 8: Performance evaluation
saved_file = pickle.load(open('saved_model.sav', 'rb'))
print(saved_file.score(x_test, y_test))

exit(0)
