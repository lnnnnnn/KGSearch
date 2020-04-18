import os, sys, time, random, re, json
import subprocess
import ljqpy

dir = r'C:\Users\liuyue\GitHub\KManagement'


def MakeTxt():
    for file in os.listdir(dir):
        fn = os.path.join(dir, file)
        print(file)
        objfile = os.path.join('txts', file + '.txt')
        subprocess.call('python ../totxt/totxt.py "%s" "%s"' % (fn, objfile))


def MakeJson():
    dd = 'txts'
    if not os.path.isdir('training'): os.mkdir('training')
    with open('training/all_data.txt', 'w', encoding='utf-8') as fout:
        for file in os.listdir(dd):
            fn = os.path.join(dd, file)
            print(file)
            segments = [];
            segment = []
            with open(fn, encoding='utf-8') as fin:
                for lln in fin:
                    line = lln.strip()
                    if ('pptx' not in file and line == '') or line.startswith('-' * 15):
                        if len(segment) > 0:
                            segments.append(''.join(segment))
                            segment = []
                    else:
                        segment.append(line)
                if len(segment) > 0:
                    segments.append(''.join(segment))
            if len(segments) == 0: continue
            for i, seg in enumerate(segments):
                seg = re.sub('[ ]{2,}', '  ', seg)
                seg = seg.replace('ﬁ', 'fi')
                jj = {'id': '%s@%d' % (file, i), 'text': seg}
                jj = json.dumps(jj, ensure_ascii=False, sort_keys=True)
                fout.write(jj + '\n')


def MakeMerged():
    txts = []
    for xx in ljqpy.LoadList('training/all_data.txt'):
        xx = json.loads(xx)
        txts.append(xx['text'])
    ljqpy.SaveList(txts, 'training/merged_text.txt')


if __name__ == '__main__':
    # MakeTxt()
    MakeJson()
    MakeMerged()
    print('done')
