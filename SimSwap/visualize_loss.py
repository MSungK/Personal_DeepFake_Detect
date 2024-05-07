import matplotlib.pyplot as plt


if __name__ == '__main__':
    loss_log = 'checkpoints/k-face_224/loss_log.txt'
    f = open(loss_log, 'r')
    lines = f.readlines()
    G_list = list()
    D_list = list()
    for line in lines[1:]:
        line = line.strip()
        G_loss = float(line[line.find('G_Loss: '):line.find('G_ID: ')].split(' ')[1])
        D_loss = float(line.split(' ')[-1])
        G_list.append(G_loss)
        D_list.append(D_loss)
    plt.plot(range(1, len(G_list)+1), G_list, label='G_Loss')
    plt.plot(range(1, len(D_list)+1), D_list, label='D_Loss')
    plt.legend()
    plt.savefig('loss.png')