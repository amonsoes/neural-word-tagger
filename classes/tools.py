import argparse
import os

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def handle_path_coll(path):
    if os.path.exists(path):
        print('WARNING: file at {} already exists'.format(path))
        while os.path.exists(path):
            user_input = input('Do you want to overwrite <o> or keep both <c> ?')
            if user_input == 'c':
                while os.path.exists(path):
                    path_wo, ext = os.path.splitext(path)
                    path = path_wo + '_copy' + ext
                return path
            elif user_input == 'o':
                os.remove(path)
                print(path)
                return path
            else:
                print('please enter valid option')
                continue
    else:
        return path