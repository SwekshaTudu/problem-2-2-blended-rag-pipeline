# Task 7: Kubernetes troubleshooting retrieval

## Problem and data

An operator investigating restarting containers or unavailable Services must find
the relevant evidence across conceptual guides, command references and debugging
instructions. Exact identifiers matter, but questions also describe symptoms in
different words. The product here is a retrieval CLI that returns evidence and
sources rather than guessing a diagnosis.

The corpus is the official English Kubernetes documentation at upstream revision
`e91eff3b59c0bf813193202a4f193d37beb7efd7`: 1,611 documents across concepts,
reference, setup, tasks and tutorials. After processing, 1,610 documents produce
17,280 passages; a template-only index page has no searchable body. Source links
are pinned to that revision, so later website changes do not alter the evidence.

## Processing and IBM integration

Markdown front matter, comments and website template wrappers are removed while
visible labels are retained where available. Headings define sections. Long
sections become passages of at most 180 words, with 30-word overlap. Code
whitespace is preserved, although long examples can cross passage boundaries.
External template includes are not expanded.

Retrieval adapts IBM's `input/search_query/KNN/knn_best.txt`, pinned to revision
`07a02f24b1a0dc95c9404b39291a8b66aa287ee7`:

1. Compute BM25 independently over title and content (heading plus passage).
2. Combine fields as the larger score plus 0.3 times the smaller score.
3. Find ten nearest semantic vectors and add a dense contribution of
   `10 * (1 + cosine) / 2` to those candidates.
4. Rank the union and return the best passage from each selected document.

The frozen, quantized all-MiniLM-L6-v2 encoder produces normalized 384-dimensional
vectors using CPU ONNX Runtime. Inputs longer than 254 content tokens are covered
by overlapping token windows; their vectors are averaged and normalized. No
classifier head, intent labels, question-stem stripping or model training is used.

Elasticsearch is replaced with local SciPy sparse matrices and an exact NumPy
vector scan. The scoring structure follows IBM's reference, but the lexical
analyzer differs and results are not numerically equivalent to Elasticsearch.
ELSER is an alternative upstream configuration and is not implemented here.
Attribution and original licenses are included in the repository.

## Deployment, validation and limits

The complete index is shipped as a checksummed snapshot to avoid an expensive
rebuild during evaluation. `run.yaml` restores it, downloads the pinned encoder,
executes real queries and runs verification without credentials or a GPU.
Source ingestion and full rebuild commands remain available in the main README.

Unit checks cover field scoring, exact error-token matches, section boundaries,
template labels and overlapping passage coverage. The full-index check verifies
dimensions, finite normalized vectors, passage checksums, unique IDs and source
links. Example searches retrieve Logging Architecture and Debug Running Pods for
container-log questions. These are functional checks; retrieval accuracy has
not been measured against independently judged relevance labels.

A broad CrashLoopBackOff query ranked a Windows-specific page first. The tool
therefore exposes document context and scores; scores are not correctness
probabilities. Other limitations are incomplete external includes, diluted
semantics in long code examples, and one passage per document potentially hiding
additional relevant sections. Exact dense search costs O(passages × dimensions)
per query; much larger deployments need an approximate vector index. Generation
is outside this submission's text-retrieval scope.
