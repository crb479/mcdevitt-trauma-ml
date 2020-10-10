# Testing for Demo Task

__doc__ = "Test of mtml.toy.crb479.add_user_tag."

from .. import echo_args
from ..crb479 import add_user_tag

def test_crb479():
	'''Test 'mtml.toy.crb479.add_user_tag'fxn by wrapping 'echo_args'
	'''
	wrapped_echo_args = add_user_tag(echo_args)
	# Evaluate with arbitrary arguments
	print(wrapped_echo_args('apple', 'orange', data = {'num': 1, 'letter': 'a'}))
	# Assert that 'wrapped_echo_args' has desired attribute
	assert(hasattr(wrapped_echo_args, '__user_tag__') and 
		(wrapped_echo_args.__user_tag__ == 'crb479'))
