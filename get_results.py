import os
import json
import pandas as pd

DIR_EXPS = "./exps/"

df = pd.DataFrame(columns=['backbone', 'conv', 'linear', 'eps', 'std', 'lr max', 'epochs', 'seed', 
                           'test acc', 'emp lip', 'robust acc', 'cert robust', 'train acc', 'exp name', 'parameters'])

for dir_name in sorted(os.listdir(DIR_EXPS)):
    if 'debug' in dir_name: 
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

    epochs = args['epochs']
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
            rob_acc = float(line.split()[-1])
        if 'parameters :' in line:
            num_params = float(line.split()[-1].split('M')[0])

    dict_line = {
        'backbone': args['backbone'], 'conv': args['conv'], 'linear': args['linear'], 'eps': args['eps'], 
        'std': bool(args['stddev']), 'lr max': args['lr_max'], 
        'epochs': epochs, 'seed': args['seed'], 'exp name': args['exp_name'].split('_LRMAX')[0],
        'test acc': test_acc, 'emp lip': emp_lip, 'robust acc': rob_acc, 'cert robust': cert_rob, 'train acc': train_acc,
        'parameters': num_params,
    }

    df.loc[len(df)] = pd.Series(dict_line)

agg = df.groupby('exp name').aggregate({
    'exp name': 'count', 'backbone': ' '.join, 'conv': ' '.join, 'linear': ' '.join, 'std': 'mean',
    'lr max': 'mean', 'test acc': 'mean', 
    'emp lip': 'mean', 'robust acc': 'mean', 'cert robust': 'mean', 'parameters': 'mean'})
agg.rename(columns={'exp name': 'count'}, inplace=True)
for key in ['backbone', 'conv', 'linear']:
    agg[key] = agg[key].apply(lambda x: x.split()[0])
agg['std'] = agg['std'].astype(bool)

agg = agg.sort_values(by = ['backbone', 'test acc'], ascending=False)
agg['exp name'] = agg.index
agg = agg.reset_index(drop=True)
print(agg.iloc[:, :-1])
agg.to_csv("results.csv")