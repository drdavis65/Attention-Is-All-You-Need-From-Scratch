## Lessons learned:

### Colab limitations and dataset chunking
Because Google Colab has unpredictable runtime limits and unexpected crashes can occur, I had to chunk the dataset to avoid crashes mid-training. My initial approach was to load a large portion (e.g., 4 million samples) and try to train for 100,000 steps to match the paper. But with Colab's unpredictable crashes, I kept having to restart. To work around this issue, I decided to train in chunks rather than shoot for the whole training run in one go and also learned to save checkpoints frequently which brings me to the next things I learned in this process...

---

### Track everything, Save frequently
Early on, I was only logging training and validation loss every 1000 steps. When plotting, I had to retroactively scale the x-axis to reflect actual step counts. This works, but the resolution of the plots is highly smoothed and you don't get a good idea of the actual pattern behind the learning. I found that it’s better to log the actual global step explicitly and log the losses, learning rate, etc. more frequently (every 100 steps). Also, saving checkpoints to Google Drive regularly (every 500 steps) can help recover in case anything goes wrong with your Colab environment and prevent wasted time and money on re-training.

---

### Make sure you are using the right tokenizer
I was using the "t5-small" at first and noticed a lot of overfitting after 6000 steps. I changed the validation set to check, and continued training, but the overfitting trend continued. So I read more into the tokenizer that I had chosen, only to find that it was not really a multilingual tokenizer and would not work well with the English to French translation task. I switched to the `MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")` tokenizer with much more stable results and got rid of my divergence problem.

---

### Know your dataset...
I wasn't thinking of the way that the wmt14 fr-en dataset was distributed throughout the samples and that the data may be collected from different sources. I stopped my training after 50,000 steps and switched my training set, assuming that they were all equivalent and none had repeated so far so it wouldn't be a problem. After doing this, my training loss had a significant dip before leveling out while my validation loss actually increased by 0.1 (as seen in the plot below). Questioning this, I realized that if the dataset is not randomized from the start, then the samples may be similar due to various reasons (same source, same sentence structure, etc.).

![Training Loss Plot Before Dataset Shuffle and Label Smoothing](plot_imgs/no-label-smoothing-training-progress.png)

---

### Label smoothing (read the paper thoroughly)
The original Transformer paper mentions label smoothing as a regularization technique to prevent the model from becoming overconfident during training. I missed this detail the first time around and trained for 60,000+ steps without it. Once I added label smoothing with a factor of 0.1, validation loss stabilized much more and generalization improved. So read the paper thoroughly, get the implementation right, and save yourself time. Some of the most important hyperparameters might be hidden in plain sight.
