import os
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings(action='ignore')
from IPython.core.display import display, HTML
# display_html = lambda df: display(HTML(df.to_html()))
display_html = print

# DIR_EXPS = "./exps/"
DIR_EXPS = "./report/"
pd.options.display.float_format = '{:.4f}'.format
pd.options.display.max_colwidth = 200

str_list = ['dataset', 'backbone', 'conv', 'linear', 'loss', 'opt', 'lr_schedule']
num_list = ['lr max', 'wd', 'epochs', 'test acc', 'emp lip', 'emp robust', 'cert robust', 'parameters']

df = pd.DataFrame(columns = str_list+num_list+['seed', 'prefix', 'exp name'])
for dir_name in sorted(os.listdir(DIR_EXPS)):
    if any([key in dir_name for key in ['debug', 'No', 'Exp0105']]): 
        continue
    if "Exp" not in dir_name:
        continue
    file_path = os.path.join(DIR_EXPS, dir_name, "log.txt")
    if not os.path.exists(file_path): 
        continue

    with open(file_path, "r") as f:
        lines = f.readlines()
    if not 'Elapsed Time' in lines[-1]: 
        continue

    with open(os.path.join(DIR_EXPS, dir_name, "options.json"), "rb") as arg_json:
        args = json.load(arg_json)

    dataset = args['dataset'] if 'dataset' in args else 'cifar10'
    epochs = args['epochs']
    lr_schedule = args['lr_schedule'] if 'lr_schedule' in args else 'tri'
    cert_rob = None

    for line in lines:
        if f'{epochs}/{epochs}' in line:
            train_acc = float(line.split('Train Acc')[-1].split()[0])
            test_acc = float(line.split('Test Acc')[-1].split()[0])
        if f'EPOCH {epochs}' in line:
            emp_lip = float(line.split('Lipschitzity:')[-1])
        if 'Certifiably Robust' in line:
            cert_rob = float(line.split(', Cert. Wrong')[0].split()[-1])
        if 'Robust accuracy' in line:
            emp_rob = float(line.split()[-1])
        if 'parameters :' in line:
            num_params = float(line.split()[-1].split('M')[0])
    
    loss = args['loss'] if 'loss' in args else 'margin'
    prefix = args['exp_name'].split(dataset)[0] if 'dataset' in args else args['exp_name'].split(args['backbone'])[0]
    if 'lam' in args:
        lam  = str(args['lam'])
        scale = str(args['scale']) if 'scale' in args else str(1.0)
        exp_name = '_'.join([prefix, dataset, args['backbone'], args['conv'], args['linear'], lam, scale, loss, args['opt'], lr_schedule])
    else:
        exp_name = '_'.join([prefix, dataset, args['backbone'], args['conv'], args['linear'], loss, args['opt'], lr_schedule])

    dict_line = {
        'backbone': args['backbone'], 'dataset': dataset, 'conv': args['conv'], 'linear': args['linear'], 'eps': args['eps'], 
        'lr max': args['lr_max'], 'wd': args['weight_decay'], 'loss': loss, 'opt': args['opt'], 'lr_schedule': lr_schedule,
        'epochs': epochs, 'seed': args['seed'], 'exp name': exp_name, 'prefix': prefix,
        'test acc': test_acc, 'emp lip': emp_lip, 'emp robust': emp_rob, 'cert robust': cert_rob, 'train acc': train_acc,
        'parameters': num_params,
    }

    df.loc[len(df)] = pd.Series(dict_line)
df.to_csv("results_org.csv")

## Aggregate
num_dict = {}
str_dict = {}
for key in num_list:
    num_dict[key] = 'mean'
for key in str_list + ['prefix']:
    str_dict[key] = ' '.join
agg_dict = {'exp name': 'count'}
agg_dict.update(num_dict)
agg_dict.update(str_dict)

agg = df.groupby('exp name').agg(agg_dict)

agg.rename(columns={'exp name': 'count'}, inplace=True)
for key in str_list + ['prefix']:
    agg[key] = agg[key].apply(lambda x: x.split()[0])

agg = agg.sort_values(by = ['dataset', 'backbone', 'test acc'], ascending=False)
agg['exp name'] = agg.index
agg.reset_index(drop=True, inplace=True)
agg = agg[['count'] + str_list + num_list + ['prefix']]
agg.to_csv("results.csv")
# print(agg.drop(columns=['exp name']))
agg.drop(columns=['lr max'], inplace=True)
print(agg)

## Show the results on datasets and backbones.
for dataset in ['cifar10', 'cifar100']:
    for backbone in ['ResNet9', 'LipConvNet_N5']:
        print(f"{dataset} {backbone}")
        select = agg[(agg['dataset'] == dataset) & (agg['backbone'] == backbone)]
        select = select.sort_values(by=['opt', 'test acc'])
        display_html(select[['count', 'conv', 'linear', 'parameters', 'loss', 'opt', 'test acc', 'emp robust', 'cert robust', 'emp lip', 'prefix']])

## Generate Tables for LaTex
pd.options.display.float_format = '{:.2f}'.format
from collections import OrderedDict
def make_logs():
    val = []
    return OrderedDict({
        'Cayley' : val.copy(),
        'SOC' : val.copy(),
        'ECO' : val.copy(),
        'SES(Ours)' : val.copy(),
    })

sch_dict = {'step': 'MultiStep', 'tri': 'Piecewise Triangle'}
backbone_list = ['LipConvNet_N5', 'ResNet9']
dataset_list = ['cifar10', 'cifar100']
for opt, lr_sch in zip(['sgd', 'adam'], ['step', 'tri']):
    print(f"{opt.upper()} {sch_dict[lr_sch]}")
    df_main = pd.DataFrame([])
    select_main = df[(df['opt'] == opt) & (df['lr_schedule'] == lr_sch) & (df['conv'] != 'PlainConv') & (df['wd'] == 0.0)]
    if len(select_main) < 1: continue
    # display_html(select)
    for backbone in backbone_list:
        for dataset in dataset_list:
            select = select_main[(select_main['backbone'] == backbone) & (select_main['dataset'] == dataset)]
            select = select.sort_values(by=['opt', 'test acc'], ascending=False)
            select['conv'] = select['conv'].replace({'CayleyConv':'Cayley', 'SESConv': 'SES(Ours)'})
            select['test acc'] = select['test acc'] * 100
            select['emp robust'] = select['emp robust'] * 100
            select['cert robust'] = select['cert robust'] * 100
            select['test acc std']    = select['test acc']
            select['emp robust std']  = select['emp robust']
            select['cert robust std'] = select['cert robust']
            select['count'] = select['conv']
            static = select.groupby('conv').agg({
                'count': 'count', 
                'test acc': 'mean', 'test acc std': 'std', 
                'cert robust': 'mean', 'cert robust std': 'std',
                'emp robust': 'mean', 'emp robust std': 'std', 
            })
            
            res_dict = {
                'count': static['count'], 'dataset': dataset, 'backbone': backbone,
                'acc': static['test acc'], 'acc std': static['test acc std'],
                'cert': static['cert robust'], 'cert std': static['cert robust std'],
                'emp': static['emp robust'], 'emp std': static['emp robust std'],                
            }

            df_main = df_main.append(pd.DataFrame(res_dict).reset_index())

    display_html(df_main.sort_values(by=['dataset', 'backbone', 'acc']))

    # Log
    logs = make_logs()
    msgAll = []
    for conv in logs.keys():
        lines = []
        for backbone in backbone_list:
            for dataset in dataset_list:
                line = df_main[(df_main['backbone'] == backbone) & (df_main['dataset'] == dataset) & (df_main['conv'] == conv)]
                if 'Res' in backbone:
                    lines += line[['acc', 'acc std', 'emp', 'emp std']].values[0].tolist()
                else:
                    lines += line[['acc', 'acc std', 'cert', 'cert std']].values[0].tolist()
        logs[conv] = np.array(lines)
    
    # Get Rank
    vals_merge = np.array([]).reshape(0, 8)
    for vals in logs.values():
        vals_merge = np.concatenate([vals_merge, vals.reshape(8, -1)[:, 0][None, ...]], axis=0)
    rank_list = vals_merge.argmax(axis=0)

    # Generate Tables for LaTex
    msgAll = []
    for conv_idx, (conv, vals) in enumerate(logs.items()):
        msg_list = ["\\texttt{" + conv + "}"]
        for i in range(len(vals)//2):
            mean = vals[2*i]
            std = vals[2*i+1]
            std = 0 if np.isnan(std) else std
            std = f"{std:.2f}"[1:] if std < 1 else f"{std:.2f}"
            msg = f"{mean:.2f} (\\footnotesize" + "{" + "$\pm$}" + f"{std})"
            if rank_list[i] == conv_idx:
                msg = "\\textbf{"  + msg + "}"
            msg_list.append(msg)
        msgAll.append(" & ".join(msg_list))
    print(" \\\\\n".join(msgAll), end=" \\\\ \n")

    print("\n\n")

