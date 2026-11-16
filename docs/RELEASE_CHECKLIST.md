# Public release checklist

Before pushing this directory to GitHub:

- [ ] Choose a repository name and code license.
- [ ] Confirm that all third-party license files under `upstream/` are retained.
- [ ] Re-run `python scripts/smoke_test.py`.
- [ ] Build `manuscript/article.pdf` on a clean TeX installation, if a source-only release is desired.
- [ ] Check the GitHub repository's large-file limits; this bundle is intentionally below them and contains no weights.
- [ ] Replace any remaining local-only data paths in run commands with paths on the user's machine.
- [ ] Add the final paper DOI, authors, and citation metadata once available.
