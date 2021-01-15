# FastText
Facebook's fasttext

# t-SNE (t-Distributed Stochastic Neighbouring Embedding)

dimensionality reduction algorithm which is used to visualize high dimensional data, such as word vectors. You can’t plot 300 dimensional data on a X-Y plane, you have to reduce the dimensions to do that. Luckily, t-SNE serves the purpose.

## Prerequisite
* Install font in your system
    ** Latha.ttf - Tamil
    ** WARNA.ttf - Sinhala

* Copy font files to `~/python/Lib/site-packages/matplotlib/mpl-data/fonts/ttf/` directory
* Edit `~/python/Lib/site-packages/matplotlib/mpl-data/matplotlibrc` and include font name in begining as shown below.

#font.sans-serif: WARNA, Latha, DejaVu Sans, Bitstream Vera Sans, Computer Modern Sans Serif, Lucida Grande, Verdana, Geneva, Lucid, Arial, Helvetica, Avant Garde, sans-serif

Following code snippet helps to check whether disered font correctly configured in the matplotlib library.
Tamil
```
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Latha"

plt.plot([1,2,3])
plt.title('பயன்படுத்தும் இலங்கையர்களுக்கு எச்சரிக்கை')
plt.show()
```

Sinhala
```
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "WARNA"
plt.rcParams['axes.unicode_minus'] = False

plt.plot([1,2,3])
plt.title('පෞද්ගලිකත්වය පිළිබඳ අපගේ ප්‍රතිපත්තිය යාවත්කාලීන කර ඇත.')
plt.show()
```

* Configure font name in script as blow.

```
plt.rcParams.update({
    'font.family':'sans-serif',
    'font.sans-serif':['WARNA'],
    })
    
plt.rcParams['axes.unicode_minus'] = False
```

* Configure fasttext pretrained models file location in plot_tsne.py

Limited the visualization to 500 words only. You may change this parameter to view more or less words.

# Tamil 
![Tamil Text](tamil.png)


# Sinhala
![Sinhala Text](sinhala.png)