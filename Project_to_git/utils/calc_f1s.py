import sys

import pandas as pd
import matplotlib.pyplot as plt
from torchmetrics.functional import f1_score
import torch
from helper_funcs import get_correct_list


# CNN = ['CNN_fold1', 'CNN_fold2', 'CNN_fold3', 'CNN_fold4', 'CNN_fold5']
# LRCN_1 = ['LRCN_LR01fold1', 'LRCN_LR01fold2', 'LRCN_LR01fold3', 'LRCN_LR01fold4', 'LRCN_LR01fold5']
# LRCN_2 = ['LRCN_fold1OLDlr', 'LRCN_fold2OLDlr', 'LRCN_fold3OLDlr', 'LRCN_fold4OLDlr', 'LRCN_fold5OLDlr']
# LRCN_3 = ['LRCN_fold1', 'LRCN_fold2', 'LRCN_fold3', 'LRCN_fold4', 'LRCN_fold5']
#
# model_list = [CNN, LRCN_1, LRCN_2, LRCN_3]
#
mixed = ['mixed_spkrs']
model_list = [mixed]
num_classes = 5
for experiment_list in model_list:
    f1_macro_list = []
    f1_weighted_list = []
    accuracy_list = []

    alt_f1_macro_list = []
    alt_f1_weighted_list = []
    alt_accuracy_list = []
    for experiment in experiment_list:
        experiment_classes = 0
        ignore = False
        alt_correct = 0
        alt_accuracy = 0

        csv_path = f'/exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s1509050_Jemima_Goodall/thesis/v3_code/final_exp/{experiment}/results_{experiment}.csv'
        df = pd.read_csv(csv_path, delimiter='|')
        df = df.drop(df.columns[0], axis=1)

        predictions = torch.tensor(df['predictions'])
        targets = torch.tensor(df['targets'])
        classes = set(targets.tolist())
        experiment_classes += len(classes)
        if experiment_classes < 5:
            ignore = 10 - sum(classes)

        f1_macro = f1_score(targets, predictions, num_classes=num_classes, ignore_index=ignore,
                           average='macro').item()
        f1_macro_list.append(round(f1_macro, 3))

        f1_weighted = f1_score(targets, predictions, num_classes=num_classes, ignore_index=ignore,
                           average='weighted').item()
        f1_weighted_list.append(round(f1_weighted, 3))

        correct = (predictions == targets).sum().item()
        accuracy = correct/predictions.shape[0]
        accuracy_list.append(round(accuracy, 3))


        altered_predictions = get_correct_list(targets, predictions)

        alt_f1_macro = f1_score(targets, altered_predictions, num_classes=num_classes, ignore_index=ignore,
                            average='macro').item()
        alt_f1_macro_list.append(round(alt_f1_macro, 3))

        alt_f1_weighted = f1_score(targets, altered_predictions, num_classes=num_classes, ignore_index=ignore,
                               average='weighted').item()
        alt_f1_weighted_list.append(round(alt_f1_weighted, 3))

        alt_correct += (abs(predictions - targets) <= 1).sum().item()
        alt_accuracy = alt_correct/predictions.shape[0]
        alt_accuracy_list.append(round(alt_accuracy, 3))

    print(f'MODEL {experiment_list[0]} ###########################')
    print(f'====== all results ======')
    print(f'f1_macro {f1_macro_list}')
    print(f'f1_weighted {f1_weighted_list}')
    print(f'alt_f1_macro {alt_f1_macro_list}')
    print(f'alt_f1_weighted {alt_f1_weighted_list}')
    print(f'accuracy {accuracy_list}')
    print(f'alt_accuracy {alt_accuracy_list}')
    print(f'====== average results over folds ======')
    print(f'f1_macro {round(sum(f1_macro_list)/5, 4)}')
    print(f'f1_weighted {round(sum(f1_weighted_list)/5, 4)}')
    print(f'alt_f1_macro {round(sum(alt_f1_macro_list)/5, 4)}')
    print(f'alt_f1_weighted {round(sum(alt_f1_weighted_list)/5, 4)}')
    print(f'accuracy {round(sum(accuracy_list)/5, 4)}')
    print(f'alt accuracy {round(sum(alt_accuracy_list) / 5, 4)}\n')


