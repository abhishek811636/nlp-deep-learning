# --------------
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import eli5

# Start of code

# Read the file with the given encoding and do not throw any error, ignore it.
df = pd.read_csv(path,encoding = "ISO-8859-1",error_bad_lines=False)



#The following shows the first 5 rows of the dataset
df.head()

agg = df.groupby("Tag").count()
print(agg)



# Let's plot a bar graph showing the Entities and the number of words each of them have
fig, ax = plt.subplots(figsize=(16,4))
plt.bar(agg.index,agg.Word,width=1)
plt.xlabel('Tag', fontsize=10)
plt.ylabel('No of Occurances', fontsize=10)
#plt.xticks(5, label, fontsize=5, rotation=30)
plt.title('Tag distribution in the dataset')
plt.show()


df = df.fillna(method='ffill')

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]


class Sentence(object):
    data = None
    sent = None
    grouped = None
    def __init__(self, data):
        self.data = data
        # Take the data, extract out the word, part of speech associated and the Tag assigned and convert it
        # into a list of tuples.
        list_vals = lambda row: [(word, pos, tag) for word, pos, tag in list(zip(row['Word'],row['POS'],row['Tag']))]
        # Group the collected values according to the Sentence # column in the dataframe so that all the words
        # in a sentence are gouped together
        self.grouped = self.data.groupby('Sentence #').apply(list_vals)
        
        #Add the rows to the 'sent' list.
        self.sent = [row for row in self.grouped]


# We will now pass our dataset to the Sentence class that we just wrote which will then convert the dataset for 
# further processing
sObject = Sentence(df)
sentences = sObject.sent


# Convert all the sentences into features
X = [sent2features(s) for s in sentences]

# Get all the labels from the dataset
y = [sent2labels(s) for s in sentences]

# Split the data into Training data and Testing data. We keep 33% of the data/rows for testing our learned model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)



# Initialising and predicting using crfsuite model
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)

crf.fit(X_train, y_train)

eli5.show_weights(crf, top=10)

y_pred = crf.predict(X_test)

# Drawing the metrics of the Test Data with that of the predicted data
print(metrics.flat_classification_report(y_test, y_pred))

print(metrics.flat_f1_score(y_test, y_pred,average='weighted'))



# End of Code

"""
NER USING KERAS
"""


# Import pandas library and fill the null values
# import numpy as np
# import pandas as pd
# df = pd.read_csv("data/NER_Data.csv",encoding = "ISO-8859-1",error_bad_lines=False)
# df = df.fillna(method="ffill")

# # Same as last tutorial. To extract out values in Word, Tag and POS columns
# class Sentence(object):
#     data = None
#     sent = None
#     grouped = None
#     def __init__(self, data):
#         self.data = data
#         list_vals = lambda row: [(word, pos, tag) for word, pos, tag in list(zip(row['Word'],row['POS'],row['Tag']))]
#         self.grouped = self.data.groupby('Sentence #').apply(list_vals)
#         self.sent = [row for row in self.grouped]
        
#     def get_next(self):
#         try:
#             s = self.grouped["Sentence: {}".format(self.n_sent)]
#             self.n_sent += 1
#             return s
#         except:
#             return None
        
# # The input vector needs to be of equal & fixed length as defined by 'max_len'. 
# # We will use padding the sentences to 'max_len'
# max_len = 50

# # Get the words in form of a list and add the string "ENDPAD" at the end of the lsit
# words = list(set(df["Word"].values))
# n_words = len(words)
# words.append("ENDPAD")

# # Get all the tags as a list
# tags = list(set(df["Tag"].values))

# # As in the last turorial(Ner with CRF) we will reconstruct the input vectors
# sObject = Sentence(df)
# sentences = sObject.sent

# word2idx = {w: i for i, w in enumerate(words)}
# tag2idx = {t: i for i, t in enumerate(tags)}


# # We will be using keras built in function 'pad_sequences' to pad the input vectors to 'max_lan'
# # This will ensure that all sequences in a list have the same length
# from keras.preprocessing.sequence import pad_sequences
# X = [[word2idx[w[0]] for w in s] for s in sentences]
# X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=n_words - 1)

# # Using the built-in function to_categorical to convert a class vector (integers) to binary class matrix.
# from keras.utils import to_categorical
# y = [[tag2idx[w[2]] for w in s] for s in sentences]
# y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])
# y = [to_categorical(i, num_classes=len(tags)) for i in y]


# # Load the train_test_split model so that we can split the data into training data and test data
# from sklearn.model_selection import train_test_split

# # Convert all the sentences into features
# #X = [sent2features(s) for s in sentences]

# #Get all the labels from the dataset
# #y = [sent2labels(s) for s in sentences]

# # Split the data into Training data and Testing data. We keep 33% of the data/rows for testing our learned model
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# # Import Keras related modules and build the model
# from keras.models import Model, Input
# from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional

# # We will now be building the model

# #Input is used to instantiate a Keras tensor. A Keras tensor is a tensor object from the underlying backend(Tensorflow)
# # which we augment with certain attributes that allow us to build a Keras model just buy knowing the inputs and 
# # output of the model
# input = Input(shape=(max_len,))

# # 'Embedding' turns positive integers into dense vectors of a fixed size
# # Therefore, we supply to it the input/output dimesions, and the input length
# model = Embedding(input_dim=n_words, output_dim=50, input_length=max_len)(input)

# model = Dropout(0.1)(model)

# #Initialize bi-directional LSTM cells
# model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)

# # Initialize a time distributed layer while building the sequential model
# out = TimeDistributed(Dense(len(tags), activation="softmax"))(model)  # softmax output layer
# model = Model(input, out)
# model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
# trained = model.fit(X_train, np.array(y_train), batch_size=32, epochs=5, validation_split=0.1, verbose=1)


# p = model.predict(np.array([X_test[1234]]))
# p = np.argmax(p,axis=1)

# # Print the predictions of the sample # 1234
# print("{:15} ({:5}): {}".format("Word", "True", "Pred"))
# for w, pred in zip(X_test[1234], p[0]):
#     try:
#         print("{:15}: {}".format(words[w], tags[pred]))
#     except:
#         pass
    



