import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    pages = dict()
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )
    return pages


def transition_model(corpus, page, damping_factor):
    distribution = {}
    num_pages = len(corpus)

    if corpus[page]:
        for p in corpus:
            distribution[p] = (1 - damping_factor) / num_pages
        for link in corpus[page]:
            distribution[link] += damping_factor / len(corpus[page])
    else:
        for p in corpus:
            distribution[p] = 1 / num_pages

    return distribution


def sample_pagerank(corpus, damping_factor, n):
    ranks = {page: 0 for page in corpus}
    page = random.choice(list(corpus.keys()))

    for _ in range(n):
        ranks[page] += 1
        model = transition_model(corpus, page, damping_factor)
        pages = list(model.keys())
        weights = list(model.values())
        page = random.choices(pages, weights=weights)[0]

    ranks = {page: count / n for page, count in ranks.items()}
    return ranks


def iterate_pagerank(corpus, damping_factor):
    num_pages = len(corpus)
    ranks = {page: 1 / num_pages for page in corpus}

    while True:
        new_ranks = {}
        for page in corpus:
            rank = (1 - damping_factor) / num_pages
            for other_page in corpus:
                if page in corpus[other_page]:
                    rank += damping_factor * ranks[other_page] / len(corpus[other_page])
                if not corpus[other_page]:
                    rank += damping_factor * ranks[other_page] / num_pages
            new_ranks[page] = rank

        converged = all(
            abs(new_ranks[page] - ranks[page]) < 0.001
            for page in ranks
        )
        ranks = new_ranks
        if converged:
            break

    return ranks


if __name__ == "__main__":
    main()
