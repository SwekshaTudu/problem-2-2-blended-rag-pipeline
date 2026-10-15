import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from retrieval import best_fields, bm25_matrix


class RetrievalTests(unittest.TestCase):
    def test_best_fields_rewards_support_without_double_counting(self):
        result = best_fields(np.array([8., 0., 2.]), np.array([5., 4., 7.]))
        np.testing.assert_allclose(result, [9.5, 4., 7.6])

    def test_bm25_exact_error_matches_and_unknown_terms(self):
        texts = ["debug CrashLoopBackOff container restarts", "debug service endpoints", "container network policy"]
        vectorizer = CountVectorizer()
        matrix = bm25_matrix(vectorizer.fit_transform(texts))
        error = (matrix @ vectorizer.transform(["CrashLoopBackOff"]).T).toarray().ravel()
        self.assertEqual(int(error.argmax()), 0)
        self.assertGreater(error[0], 0)
        self.assertEqual(int(np.count_nonzero(error)), 1)
        unknown = matrix @ vectorizer.transform(["unseenword"]).T
        self.assertEqual(unknown.nnz, 0)

    def test_empty_field_does_not_create_invalid_scores(self):
        vectorizer = CountVectorizer()
        result = bm25_matrix(vectorizer.fit_transform(["", "healthy pod"]))
        self.assertTrue(np.isfinite(result.data).all())
        self.assertEqual(result.getrow(0).nnz, 0)


if __name__ == "__main__":
    unittest.main()
