import seaborn as sns
import matplotlib.pyplot as plt

def graph_model_history(history):
    """
    Takes as argument a model history: either the return of a KerasClassifier or the model.history of a Keras model.
    Plots model accuracy, validation accuracy, model loss, and validation loss training histories, if they are present in the history.history dictionary object:
    """
        
    import seaborn as sns
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1,2, figsize = (10,5))
    if 'acc' in history.history.keys():
        axes[0].plot(history.history['acc'], label = 'Train Accuracy')
    if 'val_acc' in history.history.keys():
        axes[0].plot(history.history['val_acc'], label = 'Validation Accuracy')
    axes[0].legend()
    if 'loss' in history.history.keys():
        axes[1].plot(history.history['loss'], label = 'Train Loss')
    if 'val_loss' in history.history.keys():
        axes[1].plot(history.history['val_loss'], label = 'Validation Loss')
    axes[1].legend()
    plt.show()


def plot_confusion_matrix(y_true,y_pred, save_path = None):
    """
    Uses sklearn.metrics.confusion_matrix to plot a seaborn heatmap with normalized labels for a 3x3 confusion matrix.  
    Inputs y_true and y_pred must each have 3 categories one column.
    take an optional argument: save_path = None or string-type filepath
    """
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    sns.set(context = 'notebook', style = 'whitegrid')
    from colour import Color
    blue = Color("#212D74")
    ltblue = Color('#7890CD')
    blues = list(ltblue.range_to(blue,40))


    colors = []
    for color in blues:
        colors.append(color.hex)

    sns.heatmap(confusion_matrix(y_true, y_pred, normalize = 'true'), 
            annot = True, 
            xticklabels = ['Negative','Neutral','Positive'],
            yticklabels = ['Negative','Neutral','Positive'],
            cmap = colors)
    
    if save_path:
        plt.savefig(save_path, dpi = 500, bbox_inches = 'tight', transparent = True)
    plt.xlabel = 'Predicted Sentiment'
    plt.ylabel = 'True Sentiment'
    plt.show()