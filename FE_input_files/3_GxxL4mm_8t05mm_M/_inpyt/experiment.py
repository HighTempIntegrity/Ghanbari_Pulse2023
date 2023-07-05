class TextFile:
	def __init__(self, template_name, exp_name):
		self.exp_name = exp_name
		self.format = template_name.split('.')[-1]
		self.function = '_'+'_'.join(template_name.split('_')[1:])
		self.filename = self.exp_name + self.function
		
		with open(template_name, 'r') as file:
			self.contents = file.readlines()
	
	def update_file_name(self, new_name):
		self.filename = new_name
	
	def swap_line(self, old_line, new_line):
		search_id = self.contents.index(old_line)
		self.contents[search_id] = new_line
	
	def write_file(self):
		with open(self.filename,'w+') as file:
			for line in self.contents:
				file.write(line)
				
	def swap_string(self, old_string, new_string):
		for ii in range(len(self.contents)):
			self.contents[ii] = self.contents[ii].replace(old_string,new_string)

class InputFile:
	def __init__(self, template_name, exp_name):
		self.exp_name = exp_name
		self.function = '_'+'_'.join(template_name.split('_')[1:])
		
		f=open(template_name, "r")
		self.contents = f.readlines()
		f.close()
	
	def swap_string(self, old_string, new_string):
		for ii in range(len(self.contents)):
			self.contents[ii] = self.contents[ii].replace(old_string,new_string)
	
	def swap_line(self, old_line, new_line):
		search_id = self.contents.index(old_line)
		self.contents[search_id] = new_line
	
	def set_laser(self, laser_file):
		# search_id = self.contents.index('	INPUT = "1_AM_laser.inp"\n')
		search_id = 19
		new_line = 'INPUT = \"'+laser_file+'\"\n'
		self.contents[search_id] = new_line
	
	def set_material(self, mat_file):
		search_id = self.contents.index('*Include,input=1_material.inp\n')
		new_line = '*Include,input='+mat_file+'\n'
		self.contents[search_id] = new_line
		
	def set_step(self, step_file):
		search_id = self.contents.index('*Include,input=1_step.inp\n')
		new_line = '*Include,input='+step_file+'\n'
		self.contents[search_id] = new_line
		
	def set_tableCollection(self, tab_file):
		search_id = self.contents.index('*Include,input=1_AM_tableCollections.inp\n')
		new_line = '*Include,input='+tab_file+'\n'
		self.contents[search_id] = new_line
	
	def write_file(self, exp_id):
		self.filename = self.exp_name + self.function
		with open(self.filename,'w+') as file:
			for line in self.contents:
				file.write(line)
