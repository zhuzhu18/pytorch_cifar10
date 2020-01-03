from .general import parser

def get_configs():
    parser.description = "PyTorch ImageNet Training"

    # dataset related
    parser.add_argument('data', metavar='DIR', nargs='?', default="/media/zhuzhu/ec114170-f406-444f-bee7-a3dc0a86cfa2/dataset/cifar10",
                        help='path to datasets')

    parser.add_argument('--dataset', '-d',type=str, default='cifar10',
                        choices=["cifar10", "cifar100"],
                        help='use which Dataset (default: cifar10)')


    # models / network architectures
    # import torchvision.models as models
    model_names = ['resnet1001', 'resnet101', 'resnet146', 'resnet20', 'resnet200', 'resnet56']

    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet20)')

    return parser.parse_args()