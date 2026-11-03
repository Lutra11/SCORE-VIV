# GitHub Actions use

## CI

`ci.yml` installs Python 3.10 on Ubuntu, caches pip dependencies, installs the
package in editable mode, and runs the lightweight test suite.

## Manual experiments

Open **Actions → Run experiment → Run workflow**, then select E1--E9, provide a
configuration path relative to that experiment directory, set the seed, and
choose whether the job is a dry run.

Dry run is the default while the repository contains only experiment scaffolds.
When an experiment is implemented, its runner must write all outputs under the
provided `--output-dir`; the workflow uploads that directory as an artifact.

Raw data are cached using the contents of `dataset/manifests/` as the cache key.
Generated results should never be put into the dependency/data cache; they are
workflow artifacts.

The workflow choices and input types follow GitHub's documented
`workflow_dispatch` syntax. Python is provided through `actions/setup-python`,
and outputs are retained with `actions/upload-artifact`.

