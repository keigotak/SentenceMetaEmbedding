from pathlib import Path

lines = [['model', 'method', 'STSB-dev', 'STSB-test', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16']]
tables = {}
file_names = [
    # 'info-20220228083323330673.txt',
    'info-20220312212221927583.txt',
    'info-20220313120405683814.txt',
    'info-20220313120431997468.txt',
    'info-20220313120445887950.txt',
    'info-20220313174613032216.txt',
    'info-20220313174637738639.txt',
    'info-20220313174640090798.txt',
    'info-20220313174656300956.txt'
]

for file in file_names:
    with Path(f'../results/vec_attention/{file}').open('r') as f:
        texts = f.readlines()
    t = [text.strip() for text in texts]

    line = ['|'.join([t[0], t[1].replace(',', '')]), '|'.join([t[22], t[5], t[6]])]
    tables['STSBenchmark-dev'] = {}
    tables['STSBenchmark-dev']['pearson'] = t[37].split(': ')[1]
    tables['STSBenchmark-dev']['spearman'] = '-'
    tables[t[38]] = {}
    tables[t[38]]['pearson'] = t[39].split(': ')[1]
    tables[t[38]]['spearman'] = t[40].split(': ')[1]
    tables[t[41]] = {}
    tables[t[41]]['pearson'] = t[42].split(': ')[1]
    tables[t[41]]['spearman'] = t[43].split(': ')[1]
    tables[t[58]] = {}
    tables[t[58]]['pearson'] = t[59].split(': ')[1]
    tables[t[58]]['spearman'] = t[60].split(': ')[1]
    tables[t[71]] = {}
    tables[t[71]]['pearson'] = t[72].split(': ')[1]
    tables[t[71]]['spearman'] = t[73].split(': ')[1]
    tables[t[90]] = {}
    tables[t[90]]['pearson'] = t[91].split(': ')[1]
    tables[t[90]]['spearman'] = t[92].split(': ')[1]
    tables[t[107]] = {}
    tables[t[107]]['pearson'] = t[108].split(': ')[1]
    tables[t[107]]['spearman'] = t[109].split(': ')[1]

    header = ['STSBenchmark-dev', t[38], t[41], t[58], t[71], t[90], t[107]]
    for k in header:
        line.append(f"{tables[k]['pearson']}/{tables[k]['spearman']}")

    lines.append(line)


with Path('../results/summary_vecatt.csv').open('w') as f:
    for line in lines:
        f.write(f'{",".join(line)}\n')



