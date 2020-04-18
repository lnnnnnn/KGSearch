import json
from collections import defaultdict

# WEIGHT_THRESHOLD=0.2
# FREQ_THREHOLD=2000
WEIGHT_THRESHOLD = 0
FREQ_THREHOLD = 0
COOCC_WINDOW = 10

def calc_weight(batch):
    '''
    batch:[['项目','项目','软件','PPT']]
    '''
    r2cnt = defaultdict(dict)#两个词的共现次数
    r1cnt = defaultdict(int)#单个词出现的次数

    for docTags in batch:
        for i in range(len(docTags)):
            w1=docTags[i]
            tag_window=docTags[i+1:i+1+COOCC_WINDOW] if COOCC_WINDOW else docTags[i+1:]
            for w2 in tag_window:
                if w1 in w2 or w2 in w1:
                    continue
                if w2 < w1:
                    w1,w2=w2,w1
                if w2 not in r2cnt[w1]:
                    r2cnt[w1][w2] = 1
                else:
                    r2cnt[w1][w2] += 1

                r1cnt[w1] += 1
                r1cnt[w2] += 1

    '''
    r2cnt:{'软件':{'项目':2},'PPT':{'软件':3}}
    r1cnt:{'软件':5,'项目':2,'PPT':3}
    '''

    results=[]

    for w1,coocc_words in r2cnt.items():
        for w2,freq in coocc_words.items():
            if freq > FREQ_THREHOLD:
                sc_w1w2=freq / r1cnt[w1]
                if sc_w1w2 >= WEIGHT_THRESHOLD:
                    results.append((w1,w2,sc_w1w2))
                sc_w2w1=freq / r1cnt[w2]
                if sc_w2w1 >= WEIGHT_THRESHOLD:
                    results.append((w2,w1,sc_w2w1))

    return results

if __name__ =='__main__':
    res = calc_weight([['项目', '项目', '软件', 'PPT']])
    print(res)
    '''
    [('软件', '项目', 0.4), ('项目', '软件', 1.0), ('PPT', '软件', 1.0), ('软件', 'PPT', 0.6)]
    '''