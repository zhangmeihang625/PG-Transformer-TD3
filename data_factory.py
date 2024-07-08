from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, MD
from torch.utils.data import DataLoader

data_dict = {
    'ETTm2_high_analysis': Dataset_ETT_hour,
    'ETTm2': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTh3': Dataset_ETT_hour,
    'ETTh4': Dataset_ETT_hour,
    'ETTh51': Dataset_ETT_hour,
    'ETTh6': Dataset_ETT_hour,
    'ETTh8d0802': Dataset_ETT_hour,
    'CFEC011_high2': Dataset_ETT_hour,
    'CFEC0': Dataset_ETT_hour,
    'CFEC0_corr': Dataset_ETT_hour,
    'CFEC0_high1': Dataset_ETT_hour,
    'CFEC0_low': Dataset_ETT_hour,
    'CFEC15_low': Dataset_ETT_hour,
    'CFEC15_high1': Dataset_ETT_hour,
    'CFEC15': Dataset_ETT_hour,
    'CFEC15_corr': Dataset_ETT_hour,
    'CFEC30_low': Dataset_ETT_hour,
    'CFEC30_high1': Dataset_ETT_hour,
    'CFEC30': Dataset_ETT_hour,
    'CFEC30_corr': Dataset_ETT_hour,
    'CFEC45': Dataset_ETT_hour,
    'CFEC45_low': Dataset_ETT_hour,
    'CFEC45_high1': Dataset_ETT_hour,
    'CFEC45_corr': Dataset_ETT_hour,
    'CFEC60': Dataset_ETT_hour,
    'CFEC60_low': Dataset_ETT_hour,
    'CFEC60_high1': Dataset_ETT_hour,
    'CFEC60_corr': Dataset_ETT_hour,
    'CFEC75': Dataset_ETT_hour,
    'CFEC75_low': Dataset_ETT_hour,
    'CFEC75_high1': Dataset_ETT_hour,
    'CFEC75_corr': Dataset_ETT_hour,
    'ALEC01': Dataset_ETT_hour,
    'ALEC01_high1': Dataset_ETT_hour,
    'ALEC01_corr': Dataset_ETT_hour,
    '45steelEC02': Dataset_ETT_hour,
    '45steelEC02_high1': Dataset_ETT_hour,
    '45steelEC02_corr': Dataset_ETT_hour,
    'PEC02': Dataset_ETT_hour,
    'PEC02_corr': Dataset_ETT_hour,
    'PEC02_high1': Dataset_ETT_hour,
    'ECL': Dataset_ETT_hour,
    'WTH': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'ETTm4': Dataset_ETT_minute,
    'custom': Dataset_Custom,

}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
