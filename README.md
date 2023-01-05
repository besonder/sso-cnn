# Dispering CNN

## 주요 변경 사항
- 01.05
    - 실험세팅 완료(Adam, LR_max: 0.01, Standardization 항상 사용 등)
    - conv만 argument 입력시, 해당되는 linear ftn 자동 사용.
    - SESLinear bug fix.
    - PlainConv : LipConvNet 제외 모두 적용 가능
    - cifar100 사용 가능. `--dataset cifar100` 으로 이용.
    - `SES` 방법의 경우 `--lam` (default: 1.7) 추가 arg 사용 가능.
    - `LipConvNet` 의 경우 `--n_lip` (default: 1) 추가 arg 사용 가능. (1 -> LipConvNet-5에 해당.)

## 명령어
- 중요 arguments

```sh
## Baseline
python main.py --gpu 0 --exp_name Exp --conv PlainConv --backbone KWLarge --seed 1

## Cayley
python main.py --gpu 0 --exp_name Exp --conv CayleyConv --backbone ResNet9 --seed 1

## SES
python main.py --gpu 0 --exp_name Exp --conv SESConv2dFT --linear CayleyLinear --backbone ResNet9 --seed 1
## SESConvFT + SESLinear, LipConvNet-10
python main.py --gpu 0 --exp_name Exp --conv SESConv2dFT --lam 1.7 --backbone LipConvNet --n_lip 2 --seed 1

```

- `--linear` 옵션 미 설정시(default: 'none'), 해당 conv에 맞는 미리 정의된 linear가 사용됨.

'PlainConv': 'Linear', 'BCOP': 'BCOP', 'CayleyConv': 'CayleyLinear', 'SOC': 'SOC', 'ECO': 'ECO',
'CayleyConvED': 'CayleyLinear', 'CayleyConvED2': 'CayleyLinear',
'SESConv2dF': 'CayleyLinear', 'SESConv2dS': 'CayleyLinear', 
'SESConv2dFT': 'SESLinear', 'SESConv2dST1x1': 'SESLinear',

- argment 관련해서는 utils/option.py 참조

---
## Tensorboard
실험 결과는 exps/EXP_NAME 에 저장. tensorboard 결과도 같이 저장.

```sh
# tensorboard 실행
tensorboard --logdir exps --host 0.0.0.0 --port 6006
```


---

## Precaution & TODO List
- Conv들의 Kernel size는 default 1.
- SES 방법들은 LipConvNet에서 작동 안됨.
- Plain, LipConvNet안됨. (stride, padding 때문에 resolution 크기 차이.)
- SESLinearT, LipConvNet에서 안됨.

