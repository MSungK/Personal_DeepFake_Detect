import matplotlib.pyplot as plt


if __name__ == '__main__':
    loss_log = 'checkpoints/asian_face/loss_log.txt'
    # loss_log = 'checkpoints/people/loss_log.txt'
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
    
    avg_D_list = list()
    avg_G_list = list()
    idx = 41
    avg_D = 0
    avg_G = 0
    for i in range(1, len(G_list)):
        avg_D += D_list[i]
        avg_G += G_list[i]
        if i % idx == 0:
            avg_D_list.append(avg_D/idx)
            avg_G_list.append(avg_G/idx)
            avg_D = 0
            avg_G = 0

    # plt.plot(range(1, len(G_list)+1), G_list, label='G_Loss')
    # plt.plot(range(1, len(D_list)+1), D_list, label='D_Loss')
    plt.plot(range(1, len(avg_G_list)+1), avg_G_list, label='G_Loss')
    plt.plot(range(1, len(avg_D_list)+1), avg_D_list, label='D_Loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('loss.png')