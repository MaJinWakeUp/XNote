import json
from collections import Counter, defaultdict
import pandas as pd
from matplotlib.patches import Patch

import matplotlib.pyplot as plt

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
    date = item.get('date')
    topics = item.get('LLM_topic', [])
    for topic in topics:
        # Map to parent if it's a child, else keep as is
        parent_topic = child_to_parent.get(topic, topic)
        records.append({'date': date, 'topic': parent_topic})

df = pd.DataFrame(records)
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')

# Pie chart of all topics with fixed color per topic, legend sorted by portion size

# Count topics and sort by count (descending)
topic_counts = Counter(df['topic'])

# Temporal chart: stacked bar chart of topics over time (monthly)
df['month'] = df['date'].dt.to_period('M')
monthly_topic_counts = df.groupby(['month', 'topic']).size().unstack(fill_value=0)

# Reorder columns (topics) by total count descending
topic_order = monthly_topic_counts.sum(axis=0).sort_values(ascending=False).index.tolist()
monthly_topic_counts = monthly_topic_counts[topic_order]

# Use the same color map as before
cmap = plt.get_cmap('tab20')
color_map = {topic: cmap(i % cmap.N) for i, topic in enumerate(topic_order)}
bar_colors = [color_map[topic] for topic in topic_order]

ax = monthly_topic_counts.plot(
    kind='bar',
    stacked=True,
    figsize=(12, 6),
    color=bar_colors,
    width=0.8
)

plt.title('Topic Trends Over Time (Stacked)')
plt.xlabel('Month')
plt.ylabel('Count')

# Build legend labels with counts
legend_labels = [f"{topic} ({monthly_topic_counts[topic].sum()})" for topic in topic_order]

plt.legend(
    # handles=legend_handles,
    labels=legend_labels,
    title='Topic (Total Count)',
    # bbox_to_anchor=(1.05, 1),
    loc='upper left',
)

plt.tight_layout()
plt.show()