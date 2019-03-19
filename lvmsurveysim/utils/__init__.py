
def add_doc(value):
    """Wrap method to programatically add docstring."""

    def _doc(func):
        func.__doc__ = value.__doc__
        return func

    return _doc
