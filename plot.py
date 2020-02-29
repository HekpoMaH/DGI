from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

def plot_tsne(data, labels, model_name):
    data = data.detach().cpu().numpy()
    labels = labels.detach().long().cpu().numpy()
    embedded = TSNE(n_components=2).fit_transform(data)
    colors = 'red', 'green', 'blue', 'cyan', 'darkmagenta', 'dimgray', 'darkorange', 'cornflowerblue'
    plt.scatter(x=embedded[:, 0], y=embedded[:, 1], c=labels, cmap=matplotlib.colors.ListedColormap(colors))
    plt.axis('off')
    #                         to remove .pkl
    plt.savefig('./figures/'+model_name[:-4]+".png")
    plt.show()

    print(embedded.shape, labels.shape)
