__doc__ = "Test of ``mtml.toy.phetdam.add_user_tag``."

from .. import echo_args
from ..phetdam import add_user_tag


def test_add_user_tag():
    "Test ``mtml.toy.phetdam.add_user_tag by wrapping ``echo_args``."
    wrapped = add_user_tag(echo_args)
    # print only shows on failure
    print(wrapped(1, 2, "foo", bar = "bar", baz = "baz"))
    # test fails if assert fails
    assert (hasattr(wrapped, "__user_tag__") and
            (wrapped.__user_tag__ == "phetdam"))