import json
import re
from collections import Counter
from urllib.parse import urlparse

import matplotlib.pyplot as plt

# Load data
with open('final_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Helper to extract URLs from summary text
url_pattern = re.compile(r'https?://[^\s\]\)]+')

url_counts = []
all_domains = []

for item in data:
    summary = item.get("community_note", {}).get("summary", "")
    urls = url_pattern.findall(summary)
    url_counts.append(len(urls))
    for url in urls:
        domain = urlparse(url).netloc
        if domain.startswith('www.'):
            domain = domain[4:]
        # Get parent domain (e.g., example.com from sub.example.com)
        # parts = domain.split('.')
        # if len(parts) >= 2:
        #     parent_domain = '.'.join(parts[-2:])
        # else:
        #     parent_domain = domain
        # all_domains.append(parent_domain)
        all_domains.append(domain)

# Plot bar figure: x-axis = number of urls, y-axis = count of data
count_freq = Counter(url_counts)
x_vals = sorted(count_freq.keys())
y_vals = [count_freq[x] for x in x_vals]

plt.figure(figsize=(6,5))
plt.bar(x_vals, y_vals)
plt.xlabel('# of URLs', fontsize=18)
plt.ylabel('# of Community Notes', fontsize=18)
# plt.title('Distribution of URL Counts in Community Notes', fontsize=16)
plt.xticks(x_vals, fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()

# Top used parent websites
domain_counts = Counter(all_domains)
# merge www.youtube.com and youtu.be
domain_counts['youtube.com'] += domain_counts['youtu.be']
domain_counts['youtu.be'] = 0  # Remove youtu.be since it's merged
top_domains = domain_counts.most_common(10)

# Print top 10 parent websites
# print("Top 10 parent websites used in URLs:")
# for domain, count in top_domains:
#     print(f"{domain}: {count}")

# Plot top 10 parent websites as a bar chart (top to bottom)
domains, counts = zip(*top_domains)
plt.figure(figsize=(6, 5))
plt.barh(domains, counts)
plt.xlabel('# of Occurrences', fontsize=18)
# plt.ylabel('Website Domain', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.title('Top 10 Parent Websites Used in URLs')
plt.gca().invert_yaxis()  # Show highest at the top
plt.tight_layout()
plt.show()