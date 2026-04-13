import os
from pathlib import Path


def generate_folder_only_tree(root_dir='.', prefix='', ignore_dirs=None, show_empty=True):
    """
    只显示文件夹的目录树

    参数:
    root_dir: 根目录路径
    prefix: 前缀，用于构建树状结构
    ignore_dirs: 要忽略的目录列表
    show_empty: 是否显示空文件夹
    """
    if ignore_dirs is None:
        ignore_dirs = ['.git', '__pycache__', '.idea', '.vscode', 'node_modules']

    root_path = Path(root_dir)

    try:
        items = sorted(root_path.iterdir(),
                       key=lambda x: (not x.is_dir(), x.name.lower()))
    except PermissionError:
        print(f"{prefix}[权限拒绝: {root_path.name}]")
        return

    # 只获取文件夹
    folders = [item for item in items
               if item.is_dir() and not item.name.startswith('.')
               and item.name not in ignore_dirs]

    # 检查文件夹是否为空
    for i, folder in enumerate(folders):
        is_last = i == len(folders) - 1

        if is_last:
            current_prefix = prefix + "└── "
            next_prefix = prefix + "    "
        else:
            current_prefix = prefix + "├── "
            next_prefix = prefix + "│   "

        # 获取子文件夹数量
        sub_items = list(folder.iterdir())
        sub_folders = [item for item in sub_items
                       if item.is_dir() and not item.name.startswith('.')
                       and item.name not in ignore_dirs]

        # 显示文件夹信息
        if sub_folders or show_empty:
            folder_info = f"{folder.name}/"
            if sub_folders:
                folder_info += f" ({len(sub_folders)})"
            print(f"{current_prefix}{folder_info}")

            # 递归处理子文件夹
            generate_folder_only_tree(folder, next_prefix, ignore_dirs, show_empty)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='生成只包含文件夹的目录树')
    parser.add_argument('path', nargs='?', default='.',
                        help='要生成目录树的路径（默认当前目录）')
    parser.add_argument('--ignore', nargs='+',
                        default=['.git', '__pycache__'],
                        help='要忽略的目录（用空格分隔）')
    parser.add_argument('--hide-empty', action='store_true',
                        help='隐藏空文件夹')

    args = parser.parse_args()

    root_path = Path(args.path)
    print(f"{root_path.absolute()}/")
    generate_folder_only_tree(args.path,
                              ignore_dirs=args.ignore,
                              show_empty=not args.hide_empty)


if __name__ == "__main__":
    main()