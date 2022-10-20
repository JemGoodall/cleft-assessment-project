import sys
import pandas as pd
import torch
from helper_funcs import get_correct_list, plot_roc_curve, analyse_results,scatter_plot_per_patient
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from torchmetrics.functional import f1_score
import matplotlib.pyplot as plt
import os

run_title = sys.argv[1]
csv_path = f'/exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s1509050_Jemima_Goodall/thesis/v3_code/final_exp/{run_title}/results_{run_title}.csv'
model_dir = f'final_exp/{run_title}'
df = pd.read_csv(csv_path)
df = df.drop(df.columns[0], axis=1)
predictions = torch.tensor(df['predictions'])
targets = torch.tensor(df['targets'])
total_predictions = len(predictions)

scatter_plot_per_patient(df, save_path=model_dir+'/patient_scatterplot_new.png')


print('Altering predictions...')
correct_prediction = (abs(predictions - targets) <= 1).sum().item()
altered_predictions = get_correct_list(targets, predictions)
ALT_accuracy = correct_prediction / total_predictions
f1 = f1_score(targets, altered_predictions, num_classes=5,
                      average='macro').item()  # f1_score returns a tensor, use item()
print(classification_report(targets, altered_predictions, labels=[0,1,2,3,4],
                                    target_names=['class 0', 'class 1', 'class 2', 'class 3', 'class 4'], zero_division=0))
print(f'Test accuracy and macro f1 per frame: {ALT_accuracy:.2f} {f1:.2f}')

ConfusionMatrixDisplay.from_predictions(targets, altered_predictions, labels=[0, 1, 2, 3, 4], cmap=plt.cm.Blues)
plt.savefig(os.path.join(model_dir, 'confusion_matrix_altered_predictions_new.png'))

plot_roc_curve(y_test=targets, y_pred=altered_predictions, save_path=model_dir+'/ROC_curve_new.png')

mae,_,_ = analyse_results(df)
print(f'per file mae = {mae}')



print('Getting unaltered predictions...')
# unaltered predictions:
cp = (predictions == targets).sum().item()
ta = cp / total_predictions
f1_true = f1_score(targets, predictions, num_classes=5,
                   average='macro').item()
print(f'Unaltered results: test accuracy = {ta:.2f}, macro f1 = {f1_true}')

ConfusionMatrixDisplay.from_predictions(targets, predictions, labels=[0, 1, 2, 3, 4], cmap=plt.cm.Oranges)
plt.savefig(os.path.join(model_dir, 'confusion_matrix_actual_predictions_new.png'))