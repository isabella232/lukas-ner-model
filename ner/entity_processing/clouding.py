import random
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator

from ..utils.file_handling import read_df_from_file


entities = read_df_from_file("data/dataframes/merged_entities_10k_df.jsonl")

entities_duplicated = []

for i in entities.index:
    for j in range(entities["no_occurrences"][i]):
        entities_duplicated += [entities["word"][i]]

entities_dict = Counter(entities_duplicated)

mask = np.array(Image.open("images/bert_mask.png"))
entity_cloud = WordCloud(
    width=804,
    height=1350,
    max_font_size=50,
    max_words=600,
    background_color="#38566F",
    mode="RGBA",
    mask=mask,
).generate_from_frequencies(entities_dict)


image_colors = ImageColorGenerator(mask)
plt.figure(figsize=[7, 7])
plt.imshow(entity_cloud.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
plt.show()

entity_cloud.to_file("images/entity_cloud.png")

