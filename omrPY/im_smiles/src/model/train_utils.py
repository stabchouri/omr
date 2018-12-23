import time, math
#import matplotlib.pyplot as plt
#plt.switch_backend('agg')
#import matplotlib.ticker as ticker
import numpy as np



'''
def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
'''

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '{}m {}s'.format(m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '{} (- {})'.format(asMinutes(s), asMinutes(rs))

def chunks(l,n):
    for i in range(0, len(l), n):
        yield l[i:i +n]

