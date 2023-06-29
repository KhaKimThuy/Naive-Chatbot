import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

# Pretrained package for tokenization
# nltk.download('punkt')
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] += 1
    return bag

if __name__ == "__main__":
    # a = "Hello, thanks for visiting"
    # print(a)
    # t = tokenize(a)
    # print(t)
    s = ["I", 'am', 'a', 'human']
    al = ['hi', 'hello', 'a', 'kernel', 'am', 'I', 'car', 'human']
    print(bag_of_words(s, al))