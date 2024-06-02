import csv
from os import path as osp

if __name__ == '__main__':
    path = 'logs/lightning_logs/version_4/metrics.csv'
    fig_path = 'figures'
    reader = list(csv.reader(open(path, 'r')))
    metric = dict()
    converter = dict()
    
    for i, name in enumerate(reader[0][2:]):
        metric[name] = list()
        converter[i] = name
        
    for row in reader[1:]:
        for i, value in enumerate(row[2:]):
            if(value == ''): continue
            metric[converter[i]].append(value)

    for key, value in metric.items():
        if len(value) > 1:
            print(key)
            value = list(map(float, value))
            import matplotlib.pyplot as plt
            plt.plot(range(1, len(value)+1), value)
            plt.title(key)
            key = key.replace('/', '-')
            plt.savefig(osp.join(fig_path, key))
            plt.clf()