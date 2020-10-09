# For Demo Task

def add_user_tag(__user_tag__):
	def decorator_set_user_tag(func):
		func.__user_tag__ = __user_tag__
		return func
	return decorator_set_user_tag
