# Contributing

Thanks for helping. A few ground rules keep the numbers trustworthy.

1. **Every pricing change needs a test against an independent result:** a textbook value, a closed form, parity, or a finite-difference check.
2. **New instruments only need `pv()`** plus `to_dict` round-tripping. Add them to `instruments.py` with the `@_register` decorator so they can be logged and replayed.
3. **Never change what a stored run means.** If a fix changes results, say so in the changelog so users know old logs will replay with differences.
4. Run `ruff check src tests`, `ruff format src tests` and `pytest` before opening a pull request.

Good first issues: yield curves, a CSV position loader, American options via a binomial tree, and more stress scenarios.
