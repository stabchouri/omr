# -*- coding: utf-8 -*-
"""
chemdataextractor.cli.cem
~~~~~~~~~~~~~~~~~~~~~~~~~

Chemical entity mention (CEM) commands.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import click

from ..nlp.cem import CrfCemTagger


@click.group()
@click.pass_context
def cem(ctx):
    """Chemical NER commands."""
    pass


@cem.command()
@click.argument('input', type=click.File('r', encoding='utf8'), required=True)
@click.option('--output', '-o', help='Output model file.', required=True)
@click.option('--clusters/--no-clusters', help='Whether to use cluster features', default=True)
@click.pass_obj
def train_crf(ctx, input, output, clusters):
    """Train CRF CEM recognizer."""
    click.echo('chemdataextractor.crf.train')
    sentences = []
    for line in input:
        sentence = []
        for t in line.split():
            token, tag, iob = t.rsplit('/', 2)
            sentence.append(((token, tag), iob))
        if sentence:
            sentences.append(sentence)

    tagger = CrfCemTagger(clusters=clusters)
    tagger.train(sentences, output)

@cem.command()
@click.argument('input', type=click.File('r', encoding='utf8'), required=True)
@click.option('--model', '-m', help='Input model file.', required=True)
@click.option('--output', '-o', help='output file.', required=True)
@click.option('--clusters/--no-clusters', help='Whether to use cluster features', default=True)
@click.pass_obj
def eval_crf(ctx, input, output, clusters):
    """Evaluate CRF CEM recognizer."""
    click.echo('chemdataextractor.crf.eval')
    sentences = []
    labels = []
    for line in input:
        sentence = []
        for t in line.split():
            token, tag, iob = t.rsplit('/', 2)
            sentence.append(((token.decode('utf-8'), tag.decode('utf-8')), iob))
        if sentence:
            sentences.append(sentence)

    tagger = CrfCemTagger(model=model, clusters=clusters)
    acc = tagger.evaluate(sentences)
    print "Accuracy = %f" % (acc)

    # preds = tagger.tag_sents(sentences)
    # accuracy = 0.0
    # with open(output, 'w') as fw:
    #     for p in preds:
    #         print >> fw, ' '.join(["%s/%s/%s" % (token, tag, iob) for (token, tag), iob in p])

