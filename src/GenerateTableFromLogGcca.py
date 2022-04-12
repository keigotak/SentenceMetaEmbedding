from pathlib import Path

lines = [['model', 'method', 'STSB-dev', 'STSB-test', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16']]
tables = {}
file_names = [
    "info-20220312203614924114.txt"
]

for file in file_names:
    with Path(f'../results/gcca/{file}').open('r') as f:
        texts = f.readlines()
    t = [text.strip() for text in texts]

    line = ['gcca', 'four models']
    tables[t[0]] = {}
    tables[t[0]]['pearson'] = t[1].split(': ')[1]
    tables[t[0]]['spearman'] = t[2].split(': ')[1]
    tables[t[3]] = {}
    tables[t[3]]['pearson'] = t[4].split(': ')[1]
    tables[t[3]]['spearman'] = t[5].split(': ')[1]
    tables[t[6]] = {}
    tables[t[6]]['pearson'] = t[7].split(': ')[1]
    tables[t[6]]['spearman'] = t[8].split(': ')[1]
    tables[t[23]] = {}
    tables[t[23]]['pearson'] = t[24].split(': ')[1]
    tables[t[23]]['spearman'] = t[25].split(': ')[1]
    tables[t[36]] = {}
    tables[t[36]]['pearson'] = t[37].split(': ')[1]
    tables[t[36]]['spearman'] = t[38].split(': ')[1]
    tables[t[55]] = {}
    tables[t[55]]['pearson'] = t[56].split(': ')[1]
    tables[t[55]]['spearman'] = t[57].split(': ')[1]
    tables[t[72]] = {}
    tables[t[72]]['pearson'] = t[73].split(': ')[1]
    tables[t[72]]['spearman'] = t[74].split(': ')[1]

    header = [t[0], t[3], t[6], t[23], t[36], t[55], t[72]]
    for k in header:
        line.append(f"{tables[k]['pearson']}/{tables[k]['spearman']}")

    lines.append(line)


with Path('../results/summary_gcca.csv').open('w') as f:
    for line in lines:
        f.write(f'{",".join(line)}\n')



