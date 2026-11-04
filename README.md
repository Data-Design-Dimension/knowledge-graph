# knowledge-graph

A quick exploration from February 2024: a Dash app where you drop in a handful of Word docs and see how they cluster by topic (gensim LDA, laid out in 2D with t-SNE, plotted with Plotly). I was sketching toward a document knowledge graph and didn't get further than this.

Archived as-is in its exploratory state.

```
pip install -r requirements.txt
python viz_lda_documents.py
```

Upload 5 or more `.docx` files to see the chart.
