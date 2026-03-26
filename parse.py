
from param_utils import params, ratio

def parser_add_main_args(parser):
    # setup and protocol
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--ood_type', type=str, default='structure', choices=['structure', 'label', 'feature'],
                        help='only for cora/amazon/arxiv datasets')
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--device', type=int, default=params('device'),
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=params('seed'))
    parser.add_argument('--train_prop', type=float, default=ratio('train_prop'),
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=ratio('valid_prop'),
                        help='validation label proportion')
    parser.add_argument('--runs', type=int, default=params('runs'), help='number of distinct runs')
    parser.add_argument('--epochs', type=int, default=params('epochs'))
    parser.add_argument('--use_saved', action='store_true', help='use saved model')
    parser.add_argument('--save_model', action='store_true', help='save model')

    # model network
    parser.add_argument('--method', type=str, default='tide', choices=['msp', 'nodesafe', 'gnnsafe', 'tide'])
    parser.add_argument('--backbone', type=str, default='gcnib', choices=['gcn', 'gcnib', 'mlp'])
    parser.add_argument('--hidden_channels', type=int, default=params('hidden_channels'))
    parser.add_argument('--num_layers', type=int, default=params('num_layers'),
                        help='number of layers for GNN classifiers')
    parser.add_argument('--gat_heads', type=int, default=params('gat_heads'),
                        help='attention heads for gat')
    parser.add_argument('--out_heads', type=int, default=params('out_heads'),
                        help='out heads for gat')
    parser.add_argument('--hops', type=int, default=params('hops'),
                        help='power of adjacency matrix for sgc')

    # gnnsafe hyper
    parser.add_argument('--T', type=float, default=ratio('temperature_energy'), help='temperature for Softmax')
    parser.add_argument('--use_reg', action='store_true', help='whether to use energy regularization loss')
    parser.add_argument('--alpha', type=float, default=ratio('lamda2'), help='weight for regularization')
    parser.add_argument('--m_in', type=float, default=params('m_in'), help='upper bound for in-distribution energy')
    parser.add_argument('--m_out', type=float, default=params('m_out'), help='lower bound for in-distribution energy')
    parser.add_argument('--use_prop', action='store_true', help='whether to use energy belief propagation')
    parser.add_argument('--use_UB', action='store_true', help='whether to use bounded loss and uniform loss')
    parser.add_argument('--K', type=int, default=params('hops'), help='number of layers for energy belief propagation')
    parser.add_argument('--eta', type=float, default=ratio('eta'), help='weight for residual connection in propagation')
    parser.add_argument('--lamda1', type=float, default=ratio('lamda1'), help='weight for bounded loss and uniform loss')
    parser.add_argument('--lamda2', type=float, default=ratio('lamda2'), help='weight for mloss')
    parser.add_argument('--beta', type=float, default=params('device'), help='weight for compression and prediction')
    parser.add_argument('--gamma', type=float, default=params('device'), help='weight for conditional independence')
    parser.add_argument('--std_w', type=float, default=params('std_w'), help='weight for controlling the std of the latent space')
    parser.add_argument('--pmi_w', type=float, default=ratio('pmi_weight'), help='weight for pairwise mutual information')
    parser.add_argument('--use_pairwise', action='store_true', help='whether to use pairwise loss')
    parser.add_argument('--reset', action='store_true', help='if reset parameters')
    parser.add_argument('--train_structure', action='store_true', help='whether to train structure model')
    parser.add_argument('--train_feature', action='store_true', help='whether to train feature model')
    parser.add_argument('--train_model', action='store_true', help='whether to train model')

    # training
    parser.add_argument('--weight_decay', type=float, default=ratio('weight_decay'))
    parser.add_argument('--dropout', type=float, default=ratio('dropout'))
    parser.add_argument('--lr', type=float, default=ratio('lr'))
    parser.add_argument('--use_bn', default = True, action='store_true', help='use batch norm')

    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=params('display_step'), help='how often to print')
    parser.add_argument('--cached', action='store_true',
                        help='set to use faster sgc')
    parser.add_argument('--print_prop', action='store_true',
                        help='print proportions of predicted class')
    parser.add_argument('--print_args', action='store_true',
                        help='print args for hyper-parameter searching')
    parser.add_argument('--mode', type=str, default='detect', choices=['classify', 'detect'])


