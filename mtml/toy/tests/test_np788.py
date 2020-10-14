__doc__ = "Test of ``mtml.toy.np788.add_user_tag``."

from .. import echo_args
from ..np788 import add_user_tag

def test_np788():
    "Test for mtml.toy.np788.add_user_tag by wrapping ``echo_args``."
    wrapped_echo_args = add_user_tag(echo_args)
    print(wrapped_echo_args("Neelang", "Parghi", id = "np788"))
    assert (hasattr(wrapped_echo_args, "__user_tag__") and
            (wrapped_echo_args.__user_tag__ == "np788"))