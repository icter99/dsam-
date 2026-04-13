import os


def print_tree(start_path='.', prefix=''):
    """简化的目录树打印函数"""
    items = os.listdir(start_path)
    items = [item for item in items if not item.startswith('.')]  # 过滤隐藏文件

    for i, item in enumerate(items):
        path = os.path.join(start_path, item)
        is_last = i == len(items) - 1

        if is_last:
            print(prefix + '└── ' + item)
            new_prefix = prefix + '    '
        else:
            print(prefix + '├── ' + item)
            new_prefix = prefix + '│   '

        if os.path.isdir(path):
            print_tree(path, new_prefix)


if __name__ == "__main__":
    print('.')
    print_tree()