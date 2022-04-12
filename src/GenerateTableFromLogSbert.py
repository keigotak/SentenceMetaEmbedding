from pathlib import Path

lines = [['model', 'method', 'STSB-dev', 'STSB-test', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16']]
tables = {}
for file in Path(f'../results/sberts').iterdir():
    with file.open('r') as f:
        texts = f.readlines()
    if len(texts) < 91:
        continue
    t = [text.strip() for text in texts]
    line = [t[0], t[1]]
    tables[t[2]] = {}
    tables[t[2]]['pearson'] = t[3].split(': ')[1]
    tables[t[2]]['spearman'] = t[4].split(': ')[1]
    tables[t[5]] = {}
    tables[t[5]]['pearson'] = t[6].split(': ')[1]
    tables[t[5]]['spearman'] = t[7].split(': ')[1]
    tables[t[8]] = {}
    tables[t[8]]['pearson'] = t[9].split(': ')[1]
    tables[t[8]]['spearman'] = t[10].split(': ')[1]
    tables[t[25]] = {}
    tables[t[25]]['pearson'] = t[26].split(': ')[1]
    tables[t[25]]['spearman'] = t[27].split(': ')[1]
    tables[t[38]] = {}
    tables[t[38]]['pearson'] = t[39].split(': ')[1]
    tables[t[38]]['spearman'] = t[40].split(': ')[1]
    tables[t[57]] = {}
    tables[t[57]]['pearson'] = t[58].split(': ')[1]
    tables[t[57]]['spearman'] = t[59].split(': ')[1]
    tables[t[74]] = {}
    tables[t[74]]['pearson'] = t[75].split(': ')[1]
    tables[t[74]]['spearman'] = t[76].split(': ')[1]

    header = [t[2], t[5], t[8], t[25], t[38], t[57], t[74]]
    for k in header:
        line.append(f"{tables[k]['pearson']}/{tables[k]['spearman']}")

    lines.append(line)


with Path('../results/summary_sbert.csv').open('w') as f:
    for line in lines:
        f.write(f'{",".join(line)}\n')



