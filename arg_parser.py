
import argparse
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Run ChaoRec.")

    parser.add_argument('--Model', nargs='?', default='MHRec', help='Model name')

    parser.add_argument('--data_path', nargs='?', default='beauty', help='Input data path.')
    parser.add_argument('--learning_rate', type=float, nargs='+', default=1e-3, help='Learning rates')
    parser.add_argument('--feature_embed', type=int, default=64, help='Feature Embedding size')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    parser.add_argument('--reg_weight', type=float, nargs='+', default=1e-3, help='Weight decay.')
    parser.add_argument('--dim_E', type=int, default=64, help='Embedding dimension.')
    parser.add_argument('--num_epoch', type=int, default=1000, help='Epoch number.')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout.')
    parser.add_argument('--n_layers', type=int, default=2, help='conv_layers.')
    parser.add_argument('--ii_topk', type=int, default=10, help='the number of item-item graph topk.')
    parser.add_argument('--uu_topk', type=int, default=10, help='the number of user-user graph topk.')
    parser.add_argument('--ssl_temp', type=float, default=0.9, help='temperature coefficient.')
    parser.add_argument('--ssl_alpha', type=float, default=0.9, help='ssl coefficient.')
    # MHRec
    parser.add_argument('--h_layers', type=int, default=2, help='hypergraph layers.')
    parser.add_argument('--num_hypernodes', type=int, default=10, help='hypergraph num_hypernodes.')
    parser.add_argument('--beta1', type=float, default=0.5, help='MHRec beta1')
    parser.add_argument('--beta2', type=float, default=0.5, help='MHRec beta2')

    parser.add_argument('--seed', type=int, default=42, help='Number of seed')
    parser.add_argument('--num_workers', type=int, default=1, help='Workers number.')
    parser.add_argument('--topk', type=float, nargs='+', default=[5, 10, 20], help='topK')

    return parser.parse_args()


def load_yaml_config(model_name):
    yaml_file = f"Model_YAML/{model_name}.yaml"
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)
