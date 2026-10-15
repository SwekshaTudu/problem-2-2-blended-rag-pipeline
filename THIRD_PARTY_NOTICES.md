# Attribution and implementation changes

The retrieval design adapts IBM's Blended-RAG reference, specifically
`input/search_query/KNN/knn_best.txt` at revision
`07a02f24b1a0dc95c9404b39291a8b66aa287ee7`:
https://github.com/ibm-self-serve-assets/Blended-RAG

The original query combines `multi_match` best_fields over content and title
(tie_breaker 0.3) with dense KNN (k=10, boost=10). This project reimplements that
configuration locally with NumPy/SciPy and a frozen quantized MiniLM encoder,
replacing Elasticsearch and the original hosted generation service. It uses
per-field BM25, exact vector search, token-window averaging for long passages,
and one passage per document in final results. The lexical analyzer is different
from Elasticsearch; scores are not claimed to be numerically equivalent. It does
not implement the alternative ELSER retrieval configuration. Retrieval scores are
ranking signals, not probabilities or evidence that an answer is correct.

IBM reference material is licensed under CC BY 4.0:
https://creativecommons.org/licenses/by/4.0/
The full upstream license is retained in `licenses/IBM_Blended_RAG_LICENSE.txt`.

Kubernetes documentation: Kubernetes documentation contributors,
https://github.com/kubernetes/website, CC BY 4.0. The downloaded license and source
revision are retained alongside the corpus. Extraction and chunking are our
processing steps, not changes to the upstream source.

Encoder: sentence-transformers/all-MiniLM-L6-v2, Apache 2.0:
https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
