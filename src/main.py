"""Search Kubernetes documentation with IBM-style blended retrieval."""
import argparse
import json
import sys

from encoder import download
from retrieval import Retriever
from snapshot import restore


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("setup", help="Restore the full index and download the pinned CPU encoder")
    search = commands.add_parser("search", help="Return relevant passages, sources, and ranking scores")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--json", action="store_true", help="Print the complete machine-readable result")
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "setup":
            restore()
            download()
            print("Ready. Run: python src/main.py search \"How do I view previous container logs?\"")
            return 0
        if not args.query.strip() or len(args.query) > 4000:
            parser.error("Query must contain 1 to 4000 characters")
        if not 1 <= args.top_k <= 100:
            parser.error("--top-k must be between 1 and 100")
        result = Retriever().search(args.query, args.top_k)
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"Question: {result['query']}\n")
            for rank, row in enumerate(result["results"], 1):
                scores = row["scores"]
                print(f"{rank}. {row['title']}")
                if row["heading"]:
                    print(f"   Section: {row['heading']}")
                print(f"   Source: {row['source_url']}")
                print(f"   Score: {scores['combined']:.3f} | title: {scores['title_bm25']:.3f} | text: {scores['content_bm25']:.3f} | cosine: {scores['cosine']:.3f}")
                print(f"\n{row['text']}\n")
            print(f"Search time: {result['retrieval_ms']:.2f} ms (excludes loading). Scores are ranking signals, not confidence probabilities.")
        return 0
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}\nRun 'python src/main.py setup' to restore the supplied index and encoder.", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"Setup or file access failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
