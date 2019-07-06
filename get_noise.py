import multiprocessing

import joblib
import numpy as np
from Ocspy.ReceiverDsp.dsp_tools import normal_sample
from scipy.io import loadmat


def get_amp_pha(names, index):
    cnt = len(names)
    fea = np.ones((cnt, 4, 1025),dtype=np.complex)
    for cnt, name in enumerate(names):
        # pass
        demod = loadmat(name)['demod_symbol']
        ori = loadmat(name)['symbol']
        target = loadmat(name)['snr_x'][0, 0]
        demod = normal_sample(demod)
        ori = normal_sample(ori)
        amp = np.abs(demod[:, 1024:1024 * 2]) - np.abs(ori[:, 1024:1024 * 2])
        pha = np.angle(demod[:, 1024:1024 * 2]) / np.angle(ori[:, 1024:1024 * 2])
        fea[cnt, 0] = np.hstack((amp[0], target))
        fea[cnt, 1] = np.hstack((amp[1], target))

        fea[cnt, 2] = np.hstack((pha[0], target))
        fea[cnt, 3] = np.hstack((pha[1], target))
    print(index)
    joblib.dump(fea, f'./data/{index}')


def split_mission(names, number=8):
    total = len(names)
    each = int(np.ceil(total / 8))

    workers = []
    for i in range(each):
        workers.append(names[i * number:i * number + number])
    return workers


if __name__ == '__main__':
    pass

    # x = joblib.load('fea')
    # print(x)
    # #
    # #
    # import os
    # # names = os.listdir('./data/')
    # # fea = np.ones((301*8, 4, 1025),dtype=np.complex)
    # # # fea = np.ones((1,4,1025),dtype=np.complex)
    # # for i,name in enumerate(names):
    # #     x = joblib.load('./data/'+name)
    # #     fea[i * 8:i * 8 + 8] = x
    # # joblib.dump(fea,'fea')
    #
    # # import os
    # #
    # # PATH = '/Volumes/MyPassport/jltdata/cGN_model(1)/cgndata_right/data/'
    # # names = os.listdir('/Volumes/MyPassport/jltdata/cGN_model(1)/cgndata_right/data/')
    # # names = filter(lambda name: '.mat' in name and not name.startswith('.'), names)
    # # names = map(lambda name: PATH + name, names)
    # # names = list(names)
    # # workers = split_mission(names)[:-1]
    # #
    # # pool = multiprocessing.Pool(processes=8)
    # # for index_, worker in enumerate(workers):
    # #     # print(index_)
    # #     pool.apply_async(get_amp_pha, args=(worker, index_))
    # #
    # # pool.close()
    # # pool.join()
    # # print('hello')
    #
