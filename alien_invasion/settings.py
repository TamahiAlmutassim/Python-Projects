class Settings :
	"""Aclass to store all settings for aliean invasion"""
	def __init__(self):
		"""intialize the game settings"""
		#screen settings:
		self.screen_width = 1200
		self.screen_height = 650
		self.bg_color = (200,162,200)

		# Ship settings
		self.ship_speed = .2