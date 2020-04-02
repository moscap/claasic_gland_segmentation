import csv
import argparse
import matplotlib.pyplot as plt
import matplotlib

ACCURACY = 'acc'
LOSS = 'loss'
VAL_ACC = 'val_acc'
VAL_LOSS = 'val_loss'

def csv_plotter(file_obj, plots, title='', save_path = None):
    epoch = []  
    
    if ACCURACY in plots:
        accuracy = []
    if LOSS in plots:
        loss = []
    if VAL_ACC in plots:
        val_accuracy = []
    if VAL_LOSS in plots:
        val_loss = []
    
    reader = csv.DictReader(file_obj, delimiter=',')
    for line in reader:
        epoch.append(int(line["epoch"]))
        if ACCURACY in plots:
            accuracy.append(float(line["acc"]))
        if LOSS in plots:
            loss.append(float(line["loss"]))
        if VAL_ACC in plots:
            val_accuracy.append(float(line["val_acc"]))
        if VAL_LOSS in plots:
            val_loss.append(float(line["val_loss"]))
        
    matplotlib.rcParams.update({'font.size': 16})   
    fig, ax = plt.subplots(figsize=(20, 10))   
    ax.set_title('', fontsize=20)
    
    if ACCURACY in plots:
        ax.plot(epoch, accuracy, linewidth = 5.0)
    if LOSS in plots:
        ax.plot(epoch, loss, linewidth = 5.0)
    if VAL_ACC in plots:
        ax.plot(epoch, val_accuracy, linewidth = 5.0)
    if VAL_LOSS in plots:
        ax.plot(epoch, val_loss, linewidth = 5.0)
        
    ax.set_xlabel('Epoch', fontsize = 30)      
    fig.tight_layout() 
    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--file", required=True,
        help="path to csv file")
    ap.add_argument("-s", "--save", required=False,
        help="where to save file")
    args = vars(ap.parse_args())
    
    with open(args["file"], "r") as f_obj:
        csv_plotter(f_obj, [ACCURACY, LOSS])