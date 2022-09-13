import sys
import os




def _run_guidance():
    sys.path.insert(0, os.path.abspath("./examples/guidance"))
    import elqr

    elqr.run()

    sys.path.pop(0)


def run_examples():
    _run_guidance()
