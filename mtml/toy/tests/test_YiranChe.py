__doc__ = "Test mtml.toy.YiranChe.add_user_tag"

from .. import echo_args
from ..YiranChe import add_user_tag

def test_YiranChe():
   decorated_f = add_user_tag(echo_args)
   print(decorated_f("Hello", " world, ", 2020))

   assert (hasattr(decorated_f, "__user_tag__") and (decorated_f.__user_tag__ == "YiranChe"))
