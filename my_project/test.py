import numpy as np
import random
from utils.treebank import StanfordSentiment

dataset = StanfordSentiment()
tokens = dataset.tokens()
print(tokens)

