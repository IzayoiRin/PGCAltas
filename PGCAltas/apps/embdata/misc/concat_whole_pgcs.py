import os
import django

import numpy as np
import pandas as pd

from embdata.misc.reader import BinomialDataReader

if os.environ.get("DJANGO_SETTINGS_MODULE", None) is None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "PGCAltas.settings.dev")
    django.setup()

from embdata.models import CellsInfo


def trans2stage_data_frame(dataset, samples, psize=None, nsize=None):
    ctype = dataset.loc[:, 'ctype'].to_numpy(np.str)
    label = dataset.loc[:, 'label'].to_numpy(np.int8)
    header = dataset.columns.to_numpy(np.str)[213:]

    def cat(lab, size):
        idx = np.argwhere(label == lab).reshape(-1)
        if size:
            idx = np.random.choice(idx, size, False)
        cat_samples = samples[idx]
        expr = dataset.iloc[idx, 213:].to_numpy(np.float32)
        if lab == 1:
            qs = CellsInfo.query.in_bulk(list(cat_samples))
            stage = np.array([qs[int(cid)].stage.stage for cid in cat_samples])
        else:
            stage = np.array(['Non' for _ in cat_samples])
        return np.hstack([cat_samples.reshape(-1, 1), stage.reshape(-1, 1), expr])

    pos = cat(1, psize)

    return pd.DataFrame(pos, columns=np.hstack(['cid', 'stage', header]))


def trans2stage_data_frame_per(dataset, samples, pid, stage='E6.5'):
    pdata, psample = dataset.iloc[pid, 213:], samples[pid]
    header = pdata.columns.to_numpy(np.str)
    stage = np.array([stage for _ in range(len(pid))]).reshape(-1, 1)
    data = np.hstack([psample.reshape(-1, 1), stage.reshape(-1, 1), pdata])
    return pd.DataFrame(data, columns=np.hstack(['cid', 'stage', header]))


def trans_and_concat(save=True):
    dirn = "EMTAB6967"
    pklfile_1k1 = '0308-FIoMrmEQEeq1JSjxDhxCxQ=='
    pklfile_2k = '0308-F2Y2pGEWEeqSfyjxDhxCxQ=='
    reader_1k = BinomialDataReader.init_from_pickle(dirn, pklfile_1k1)
    df1k = trans2stage_data_frame(reader_1k.dataset, reader_1k.samples)

    path = r'D:\D\Desktop\SIBS-S324-IZ@YOI\work\project1018\PGCAltas\dataset\EMTAB6967\0217-large\texts'
    file = 'SVMClassifierBinomialFlowPredict.txt'
    pre_lab = pd.read_table(os.path.join(path, file), header=0, index_col=0)
    pre_l = pre_lab.iloc[:, -1].to_numpy()
    pos_id = np.argwhere(pre_l == 1).reshape(-1)
    reader_2k = BinomialDataReader.init_from_pickle(dirn, pklfile_2k)
    df2k = trans2stage_data_frame_per(reader_2k.dataset, reader_2k.samples, pid=pos_id)
    df = pd.concat([df1k, df2k], ignore_index=True)  # type: pd.DataFrame
    print(df.shape)
    print(df.index.to_numpy())
    if save:
        stpath = path
        sppath = path.rsplit('\\', 1)[0] + r'\pickles'
        df.to_pickle(os.path.join(sppath, 'WholePGCs.pkl'))
        with open(os.path.join(stpath, 'WholePGCs.txt'), 'w') as f:
            df.to_csv(f, sep='\t', header=True, index=True)
        return


if __name__ == '__main__':
    trans_and_concat()
