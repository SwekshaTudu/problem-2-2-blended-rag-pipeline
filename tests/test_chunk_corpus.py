import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chunk_corpus import chunk_documents, passages, sections


class ChunkingTests(unittest.TestCase):
    def test_headings_and_fenced_shell_comments(self):
        result = list(sections('## Debug\nCheck pods.\n```sh\n# shell comment\nkubectl get pods\n```\n### Logs\nRead logs.'))
        self.assertEqual([heading for heading, _ in result], ["Debug", "Debug > Logs"])
        self.assertIn("# shell comment", result[0][1])

    def test_visible_template_labels_and_commands_survive(self):
        result = list(sections('A {{< glossary_tooltip text="cluster" term_id="cluster" >}}.\n<!-- hidden -->\n```sh\necho "<pod-name>"\n```'))
        self.assertIn("A cluster.", result[0][1])
        self.assertNotIn("hidden", result[0][1])
        self.assertIn('echo "<pod-name>"', result[0][1])

    def test_overlap_keeps_boundary_context_and_all_words(self):
        words = [str(i) for i in range(47)]
        result = list(passages(" ".join(words), 20, 5))
        self.assertTrue(all(len(part.split()) <= 20 for part in result))
        self.assertEqual(result[0].split()[-5:], result[1].split()[:5])
        self.assertEqual(set(" ".join(result).split()), set(words))

    def test_multiline_helpers_preserve_visible_labels(self):
        result = list(sections('Read {{<\nglossary_tooltip text="control plane"\nterm_id="control-plane" >}} docs.'))
        self.assertEqual(result, [("", "Read control plane docs.")])

    def test_sources_and_distinct_passage_ids(self):
        document = dict(document_id="a.md", title="Debug", body_markdown="## Logs\n" + "word " * 50,
                        section="tasks", source_url="https://example.org/source")
        result = list(chunk_documents([document], 20, 5))
        self.assertEqual(len(result), len({r["chunk_id"] for r in result}))
        self.assertTrue(all(r["source_url"] == document["source_url"] for r in result))
        self.assertTrue(all(r["heading"] == "Logs" for r in result))

    def test_invalid_overlap_rejected(self):
        with self.assertRaises(ValueError):
            list(chunk_documents([], 20, 20))


if __name__ == "__main__":
    unittest.main()
