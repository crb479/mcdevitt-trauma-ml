from .. import echo_args
from ..wh7182 import add_user_tag


def test_wh7182():
	obj = add_user_tag(echo_args)
	obj("randomTest")
	assert obj.__user_tag__ =="wh7182"
		
