def graph_model_history(history):
    import seaborn as sns
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1,2, figsize = (10,5))
    axes[0].plot(history.history['acc'], label = 'Train Accuracy')
    axes[0].plot(history.history['val_acc'], label = 'Validation Accuracy')
    axes[0].legend()
    axes[1].plot(history.history['loss'], label = 'Train Loss')
    axes[1].plot(history.history['val_loss'], label = 'Validation Loss')
    axes[1].legend()
    plt.show()

def plot_confusion_matrix(y_true,y_pred):
    import seaborn as sns

    sns.heatmap(confusion_matrix(y_true, y_pred, normalize = 'true'), 
            annot = True, 
            xticklabels = ['Negative','Neutral','Positive'],
            yticklabels = ['Negative','Neutral','Positive'])
    plt.xlabel = 'Predicted Sentiment'
    plt.ylabel = 'True Sentiment'
    plt.show()