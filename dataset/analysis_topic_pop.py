import json
from collections import Counter, defaultdict
import pandas as pd
from matplotlib.patches import Patch

import matplotlib.pyplot as plt
import seaborn as sns

# Load data
with open('final_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

with open('topics.json', 'r', encoding='utf-8') as f:
    topics_dict = json.load(f)

# Only use parent topics: map each LLM_topic to its parent if it's a child
parent_topics = set(topics_dict.keys())
child_to_parent = {}
for parent, children in topics_dict.items():
    for child in children:
        child_to_parent[child] = parent

# Collect records with only parent topics
records = []
for item in data:
    num_re = item.get('retweet_count', 0)  # Use retweet_count as date for simplicity
    topics = item.get('LLM_topic', [])
    for topic in topics:
        # Map to parent if it's a child, else keep as is
        parent_topic = child_to_parent.get(topic, topic)
        records.append({'number': num_re, 'topic': parent_topic})

# Create a DataFrame
df = pd.DataFrame(records)

topic_order = df['topic'].value_counts().index.tolist()
cmap = plt.get_cmap('tab20')
color_map = {topic: cmap(i % cmap.N) for i, topic in enumerate(topic_order)}
colors = [color_map[topic] for topic in topic_order]
# Group by topic and sum the retweet counts
topic_popularity = df.groupby('topic')['number'].sum().sort_values(ascending=False)

# Prepare data for density plot with log-scaled x axis only
plt.figure(figsize=(10, 6))
for topic in topic_popularity.index[::-1]:  # Reverse order for better visibility
    topic_data = df[df['topic'] == topic]['number']
    topic_data = topic_data[topic_data > 0]  # Only keep retweet counts > 0
    if len(topic_data) > 1:  # KDE needs at least 2 points
        sns.kdeplot(topic_data, label=topic, fill=True, alpha=0.7, clip=(0, None), log_scale=(True, False),
                    color=color_map[topic])
plt.xlim(left=1)  # Set x limit to avoid log(0)
plt.xscale('log')
plt.xlabel('Retweet Count (log scale)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Density Map', fontsize=14)

# Custom legend with alpha
labels = [f"{topic} ({topic_popularity[topic]:,})" for topic in topic_popularity.index]
handles = [Patch(color=color_map[topic], label=label, alpha=0.7) for topic, label in zip(topic_popularity.index, labels)]
plt.legend(
    handles=handles, 
    title='Topic (Total # of retweets)', 
    loc='upper left', 
    ncol=1,
    bbox_to_anchor=(1.02, 1),
    fontsize=12,
    title_fontsize='13',
)

plt.tight_layout()
plt.show()
