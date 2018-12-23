from glob import glob
import sys, os
import argparse
import codecs

import numpy as np
from collections import namedtuple
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import confusion_matrix

from chemdataextractor.nlp.tokenize import ChemWordTokenizer
cwt = ChemWordTokenizer()

PARA_DIR = "./para/para-annotated"

# start_symbol = ["<para sub="]
start_symbol = ["<para sub=\"opt\"", "<para sub=\"scope\""]
end_symbol = ["</para>"]

def tokenize(para):
    return cwt.tokenize(" ".join(para))

def match(s, pattern):
    for pat in pattern:
        if s.find(pat) >= 0:
            return True
    return False

def generate_data(args):
    data = []
    n_positive = n_negative = 0

    for f in glob(PARA_DIR + "/*"):
        para = []
        label = 0
        for i, l in enumerate(codecs.open(f, 'r', "utf-8")):
            l = l.strip()
            if match(l, start_symbol):
                if label != 0:
                    print f, i
                assert (label == 0)
                if len(para) > 0: # paragraph before <para sub=xxx>
                    data.append((label, tokenize(para)))
                    n_negative += 1
                para = []
                label = 1
                continue
            if match(l, end_symbol) and label == 1:
                para.append(l[:l.find("</para>")])
                data.append((label, tokenize(para)))
                para = []
                n_positive += 1
                label = 0
                continue

            if len(l) == 0 or (not match(l, start_symbol) and l.find("<para sub=") >= 0):
                data.append((label, tokenize(para)))
                if label == 0: n_negative += 1
                if label == 1: n_positive += 1
                para = []
            else:
                para.append(l)

    print >> sys.stderr, "Detected: %d positive examples, %d negative examples" % (n_positive, n_negative)
    n_positive =  n_negative = 0
    with codecs.open(args.save_data, 'w', "utf-8") as fw:
        for label, para in data:
            if len(para) > 10:
                if label == 0: n_negative += 1
                else: n_positive += 1
                para = " ".join(para).strip()
                print >> fw, "%d\t%s" % (label, para)
    print >> sys.stderr, "Selected: %d positive examples, %d negative examples" % (n_positive, n_negative)

def load_dataset(filename, percent=0.8):
    train = namedtuple('Dataset', ['texts', 'labels'])([], [])
    test  = namedtuple('Dataset', ['texts', 'labels'])([], [])
    for p in codecs.open(filename, 'r', 'utf-8'):
        label, text = p.strip().split('\t')
        if np.random.rand() < percent:
            train.texts.append(text)
            train.labels.append(label)
        else:
            test.texts.append(text)
            test.labels.append(label)
    print "Load %d examples for Train, and %d examples for Test" % (len(train.texts), len(test.texts))
    return train, test

def train_and_test(args):
    train, test = load_dataset(args.train, percent=0.8)
    # text_clf = Pipeline([('vect', CountVectorizer()),
    #                      ('tfidf', TfidfTransformer()),
    #                      ('clf', SGDClassifier(loss='hinge', penalty='l2',
    #                                            alpha=1e-3, random_state=42,
    #                                            max_iter=5, tol=None))
    #                     ])

    # text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
    #                      ('tfidf', TfidfTransformer()),
    #                      ('clf', LogisticRegression(penalty='l2', C=2.0, class_weight={'0':1.0, '1':1.0}))])

    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MLPClassifier(activation="tanh", alpha=0.1))])
    text_clf.fit(train.texts, train.labels)
    predicted = text_clf.predict(test.texts)
    acc = np.mean(predicted == test.labels)
    print "Accuracy = %f vs. Majority baseline = %f" % (acc, 1 - sum(map(float, test.labels)) / len(test.labels))
    print "Confusion matrix:"
    print confusion_matrix(test.labels, predicted)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="meaningful paragraph recognization")
    argparser.add_argument("--gen", action="store_true")
    argparser.add_argument("--save_data", type=str, default="./para/allpara.txt")
    argparser.add_argument("--train", type=str)
    argparser.add_argument("--test", type=str)
    argparser.add_argument("--save_model", type=str)

    args = argparser.parse_args()
    if args.gen:
        generate_data(args)
    elif args.train is not None:
        train_and_test(args)
    else:
        pass

