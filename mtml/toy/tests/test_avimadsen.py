from ..avimadsen import add_user_tag
from ..  import echo_args

def test_avimadsen():

	wrapped_echo_args = add_user_tag(echo_args)

	print(wrapped_echo_args('test1', 'test2'))

	assert(hasattr(wrapped_echo_args, '__user_tag__') and
		(wrapped_echo_args.__user_tag__ == 'avimadsen'))