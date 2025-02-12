import numpy as np
import misc
import data_io as dio

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, confusion_matrix, cohen_kappa_score, jaccard_score, accuracy_score, balanced_accuracy_score, matthews_corrcoef, classification_report, multilabel_confusion_matrix, precision_score
from rich.console import Console
from rich.table import Table

def report(y_true, y_pred):
  cm = confusion_matrix(y_true, y_pred, normalize='true') * 100
  cm2 = confusion_matrix(y_true, y_pred)
  kappa_score = cohen_kappa_score(y_true, y_pred)
  rand_score = adjusted_rand_score(y_true, y_pred)
  mutual_score = adjusted_mutual_info_score(y_true, y_pred)
  jac_score = jaccard_score(y_true, y_pred, average = 'micro')
  accuracy = accuracy_score(y_true, y_pred)
  balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
  matthews_coef = matthews_corrcoef(y_true, y_pred)
  precision = precision_score(y_true, y_pred, average = 'micro')

  metrics_table = Table(title="Report")
  metrics_table.add_column("Metric", justify="left", style="cyan")
  metrics_table.add_column("Value", justify="left", style="green")

  metrics_table.add_row("Kappa", "{:.2f}".format(kappa_score))
  metrics_table.add_row("Adjusted Rand", "{:.2f}".format(rand_score))
  metrics_table.add_row("Adjusted Mutual Info", "{:.2f}".format(mutual_score))
  metrics_table.add_row("Jaccard", "{:.2f}".format(jac_score))
  metrics_table.add_row("Accuracy", "{:.2f}".format(accuracy))
  metrics_table.add_row("Balanced Accuracy", "{:.2f}".format(balanced_accuracy))
  metrics_table.add_row("Matthews Correlation", "{:.2f}".format(matthews_coef))
  metrics_table.add_row("Precision", "{:.2f}".format(precision))

  cm_table = dio.array_to_table(cm)
  cm2_table = dio.array_to_table(cm2)

  subgrid = Table.grid(expand=True)
  subgrid.add_column()
  subgrid.add_row(cm_table)
  subgrid.add_row(cm2_table)

  grid = Table.grid(expand=True)
  grid.add_column()
  grid.add_column(justify="right")
  grid.add_row(metrics_table, subgrid)

  console = Console()
  console.print(grid)

def reclass_data(y_pred, y_true):
  mapping = {}
  true_labels = np.flip(np.unique(y_true))
  for tl in true_labels:
    values, counts = np.unique(y_pred[np.nonzero(y_true == tl)], return_counts=True)
    pos = np.argmax(counts)
    old_label = values[pos]
    mapping[old_label] = tl
    y_pred[np.nonzero(y_pred == old_label)] = tl
  return mapping