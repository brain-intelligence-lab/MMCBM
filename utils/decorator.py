# -*- encoding: utf-8 -*-
"""
@Author :   liuyang
@github :   https://github.com/ly1998117/MMCBM
@Contact :  liu.yang.mine@gmail.com
"""

import os
import pandas as pd
from functools import wraps


# -------------------------------------- decorators -------------------------------------- start
def decorator_args(function):
    @wraps(function)
    def decorated():
        from params import get_args
        args = get_args()
        function(args)
        if 'fold' not in args.dir_name:
            if args.name != "":
                args.dir_name += f'_{args.name}'
            args.dir_name += f'/fold_{args.k}'
            if args.mark != "":
                args.dir_name += f'_{args.mark}'
            args.pre_embeddings = True  # used for MMCBM
        return args

    return decorated


def run_model(pm, init_model_fn, multi=False, fold=None):
    def decorator(function=None):
        @wraps(function)
        def decorated():
            def _run(p, k):
                pm, function = p
                model, args = init_model_fn(k=k, **pm)
                result = function(args, k, model)
                name = args.name
                dir_base_name = os.path.join('result', *args.dir_name.split('/')[:-1])
                del model, args  # 释放资源
                return result, name, dir_base_name

            all_retrievals = []
            name = ''
            dir_base_name = ''
            if fold is not None:
                result, name, dir_base_name = _run((pm, function), fold)
                all_retrievals.extend(result)
            else:
                if multi:
                    import mpire as mpi
                    with mpi.WorkerPool(5, shared_objects=[pm, function], daemon=False) as p:
                        results = p.map(_run, range(5))
                        [all_retrievals.extend(r[0]) for r in results]
                        name = results[0][1]
                        dir_base_name = results[0][2]
                else:
                    for k in range(5):
                        result, name, dir_base_name = _run((pm, function), k)
                        all_retrievals.extend(result)
            if len(all_retrievals) > 0:
                all_retrievals = pd.concat(all_retrievals)
            return all_retrievals, name, dir_base_name

        return decorated

    return decorator


def cache_df(save_path=None, cache=True):
    gsave_path = save_path

    def decorator(function):
        @wraps(function)
        def decorated(*args, save_path=None, csv_name=None, **kwargs):
            if save_path is None:
                save_path = gsave_path
            if csv_name is not None:
                save_path = f'{save_path}/{csv_name}'
                kwargs['csv_name'] = csv_name
            if not os.path.exists(save_path) or cache is False:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                df = function(*args, **kwargs)
                if isinstance(df, (list, tuple)):
                    df = pd.concat([d for d in df if isinstance(d, pd.DataFrame)])
                df.to_csv(save_path, index=False)
            else:
                df = pd.read_csv(save_path)
            return df

        return decorated

    return decorator

# -------------------------------------- decorators -------------------------------------- end
