# Towards Automated Community Notes-Style Fact-Checking with VLM Agents for Combating Contextual Deception

In this work, we address the gap in automated Community Notes generation for image-based contextual deception. 
1. We introduce **XCheck**, a real-world multimodal dataset of X posts paired with ground-truth Community Notes, together with retrieved external context. 
2. We further propose **ACCNote**, a retrieval-augmented multi-agent framework that enhances LVLMs to generate context-corrective notes. 
3. To evaluate whether generated notes are helpful to users, we develop a new metric Context Helpfulness Score (**CHS**) that aligns with pilot user study outcomes. 
4. Extensive experiments show that ACCNote consistently outperforms foundation LVLM baselines and naive RAG on both deception detection and note generation tasks on XCheck, and also exceeds the commercial baseline GPT5-mini.


<!-- > Try our interactive demo on Google Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/*/*/blob/main/demo.ipynb) -->

> Try our Notebook demo here: [`demo.ipynb`](demo.ipynb)


## Dataset: XCheck

**XCheck** dataset was constructed in four stages as: deceptive data curation, non-deceptive control set addition, data cleaning and annotation, and context augmentation via reverse image search.

![XCheck Dataset Construction](asset/data_collection.png)

Please check the `dataset` directory for the detailed information of **XCheck**.

## Method: ACCNote

**ACCNote** is an automated context-corrective note generation framework that combines:
* Retrieval-augmented generation (RAG) to improve credibility and veracity by grounding the model in external evidence; 
* Multi-agent collaboration to improve clarity, relevance, and neutrality by filtering noisy context, separating conflicting evidence, and refining the final note.

![ACCNote Framework](asset/Overview.png)

Please check the `src` directory for the implementation of **ACCNote**.

## Metric: CHS

CHS is designed to better reflect the five dimensions in the Community Notes guideline: credibility, clarity, relevance, veracity, and neutrality.

Please check the `src/metric` file for the implementation of **CHS**.