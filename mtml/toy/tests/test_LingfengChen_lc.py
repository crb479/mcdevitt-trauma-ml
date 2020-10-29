from .. import echo_args
from ..LingfengChen_lc import add_user_tag

def test_LingfengChen_lc():
    f = add_user_tag(echo_args)
    print(f('A', 'B'))
    assert f.__user_tag__ == 'LingfengChen_lc'
