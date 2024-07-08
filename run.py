import argparse
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Non-stationary Transformers for Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=str, required=False, default=1, help='status')
parser.add_argument('--model_id', type=str, required=False, default='train', help='model id')
parser.add_argument('--model', type=str, required=False, default='ns_Transformer',
                    help='model name, options: [ns_Transformer, Transformer，Informer，Autoformer]')

# data loader
parser.add_argument('--data', type=str, required=False, default='CFEC75_high1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='CFEC75_high1.csv', help='data file')
parser.add_argument('--features', type=str, default='MS',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
#parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
#parser.add_argument('--pred_len', type=int, default=128, help='prediction sequence length')
#parser.add_argument('--pred_len', type=int, default=256, help='prediction sequence length')
#parser.add_argument('--pred_len', type=int, default=336, help='prediction sequence length')

# model define
parser.add_argument('--enc_in', type=int, default=4, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=4, help='decoder input size')
parser.add_argument('--c_out', type=int, default=4, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--fac'
                    ''
                    ''
                    'tor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')     #TimeFeatureEmbedding
parser.add_argument('--activation', type=str, default='gelu', help='activation')   #激活函数为gelu, 即高斯误差线性单元
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=40, help='train epochs')  #epochs=100的效果和epochs=30 差不多
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')

# parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')   #自适应学习率优化算法有 AdaGrad， RMSProp， Adam 以及AdaDelta，本代码用的是啥？
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)  #自动混合精度训练


# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# de-stationary projector params   #去平稳化注意力
parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128], help='hidden layer dimensions of projector (List)')
parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

fix_seed = args.seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

if args.use_gpu:
    if args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    else:
        torch.cuda.set_device(args.gpu)

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        # if args.do_predict:
        if False:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)


        # plt.rcParams['font.family'] = 'Times New Roman'
        # trues = np.load('./results/' + setting + '/true.npy')
        # preds = np.load('./results/' + setting + '/pred.npy')
        # plt.figure(dpi=600)
        # plt.plot(trues[:, 0, -1].reshape(-1), label='GroundTruth')
        # plt.plot(preds[:, 0, -1].reshape(-1), label='Prediction')
        # font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 14}
        # plt.legend(prop=font)
        # plt.xticks(fontsize=14)
        # plt.yticks(fontsize=14)
        # plt.xlabel('Sequential point', fontdict={'fontname': 'Times New Roman', 'fontsize': 16})
        # plt.ylabel('Resulting value', fontdict={'fontname': 'Times New Roman', 'fontsize': 16})
        # plt.savefig('example1_plot.png')
        # plt.show()


        # zmh adding
        plt.rcParams['font.family'] = 'Times New Roman'
        trues = np.load('./results/' + setting + '/true.npy')
        preds = np.load('./results/' + setting + '/pred.npy')
        plt.figure(dpi=600)
        plt.plot(trues[:, 0, -1].reshape(-1), label='GroundTruth')
        plt.plot(preds[:, 0, -1].reshape(-1), label='Prediction')
        font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 18}
        plt.legend(prop=font)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel('Sequential point', fontdict={'fontname': 'Times New Roman', 'fontsize': 18})
        plt.ylabel('Resulting value', fontdict={'fontname': 'Times New Roman', 'fontsize': 18})
        plt.savefig('example1_plot.png')
        plt.legend()
        plt.show()

        # # # #wrp adding
        # trues = np.load('./results/' + setting + '/true.npy')
        # preds = np.load('./results/' + setting + '/pred.npy')
        # plt.figure()
        # plt.plot(trues[:, 0, -1].reshape(-1), label='GroundTruth')
        # plt.plot(preds[:, 0, -1].reshape(-1), label='Prediction')
        # plt.legend()
        # plt.show()
        #
        # torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                  args.model,
                                                                                                  args.data,
                                                                                                  args.features,
                                                                                                  args.seq_len,
                                                                                                  args.label_len,
                                                                                                  args.pred_len,
                                                                                                  args.d_model,
                                                                                                  args.n_heads,
                                                                                                  args.e_layers,
                                                                                                  args.d_layers,
                                                                                                  args.d_ff,
                                                                                                  args.factor,
                                                                                                  args.embed,
                                                                                                  args.distil,
                                                                                                  args.des, ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
