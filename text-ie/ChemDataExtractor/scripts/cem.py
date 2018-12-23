from chemdataextractor.nlp.cem import CrfCemTagger, CemTagger
import six, codecs

def train(args):
    """Train the CRF tagging model"""
    sentences = []
    for line in args.input:
        sentence = []
        for t in line.split():
            token, tag, iob = t.rsplit('/', 2)
            sentence.append(((token, tag), iob))
        if sentence:
            sentences.append(sentence)

    tagger = CrfCemTagger(clusters = True)
    tagger.train(sentences, args.model)

def evaluate(args):
    """Evaluate the CRF tagging model"""
    sentences = []
    # for line in open('../../data/cde-ner/chemdner-development-tag.txt'):
    for i, line in enumerate(open(args.input)):
        # if i >= 1000: break
        sentence = []
        for t in line.split():
            token, tag, iob = t.rsplit('/', 2)
            sentence.append(((token.decode('utf-8'), tag.decode('utf-8')), iob))
            # labels.append(iob)
        if sentence:
            sentences.append(sentence)

    # tagger = CrfCemTagger(model = args.model, clusters = True)
    tagger = CrfCemTagger()
    # tagger = CemTagger()
    # acc = tagger.evaluate(sentences)
    tagged_sents = tagger.tag_sents([token for (token, label) in sent] for sent in sentences)
    # [[((w, t), l), ((w, t), l), ...], ...]
    if isinstance(tagger, CemTagger):
        tagged_sents = [[(token, label or 'O') for (token, label) in sent] for sent in tagged_sents]
    with codecs.open(args.output, 'w', 'utf-8') as fw:
        for gold_sent, pred_sent in zip(sentences, tagged_sents):
            for gold_token, pred_token in zip(gold_sent, pred_sent):
                # gold_token = ((w, t), l)
                assert (gold_token[0] == pred_token[0])
                print >> fw, " ".join([gold_token[0][0], gold_token[0][1], gold_token[1], pred_token[1]])
            print >> fw

    # accuracy = float(sum(x == y for x, y in six.moves.zip(gold_tokens, pred_tokens))) / len(pred_tokens)
    # print "Accuracy = %f" % (accuracy)

    gold_tokens = sum(sentences, [])
    pred_tokens = sum(tagged_sents, [])
    """ when CrfCemTagger is used, None should be 'O' instead.
    """
    n_gold_cems = sum(label != 'O' for token, label in gold_tokens)
    n_pred_cems = sum(label != 'O' for token, label in pred_tokens)
    n_correct_cems = sum((x == y and x[1] != 'O') for x, y in six.moves.zip(gold_tokens, pred_tokens))
    precision = float(n_correct_cems) / n_pred_cems
    recall = float(n_correct_cems) / n_gold_cems
    print "Precision (token) = %d/%d = %f" % (n_correct_cems, n_pred_cems, precision)
    print "Recall (token) = %d/%d = %f" % (n_correct_cems, n_gold_cems, recall)
    print "F1 (token) = %f" % (2 * precision * recall / (precision + recall))

import sys
import argparse
if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Chemical Named Entity Recognizer")
    argparser.add_argument("--train", action="store_true")
    argparser.add_argument("--eval", action="store_true")
    argparser.add_argument("--input", type=str, required=True)
    argparser.add_argument("--model", type=str)
    argparser.add_argument("--output", type=str)

    args = argparser.parse_args()
    if args.train:
        train(args)
    elif args.eval:
        evaluate(args)
    else:
        pass

