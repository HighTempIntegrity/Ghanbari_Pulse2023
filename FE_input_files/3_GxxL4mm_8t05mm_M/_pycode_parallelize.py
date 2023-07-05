# Use modern python to run this. Notepadd++ F5 command:
# cmd /k cd /d $(CURRENT_DIRECTORY) && python $(FULL_CURRENT_PATH)

import pandas as pd

STG = {	# Settings
	'batches':2,	# Number of parallel batches
	'zeros':3,
}
TPL = {	# Templates
	'experiment':'2_exp.csv',
	'script':'_pycode_experiment.py',
}

class ExperimentScript:
	def __init__(self, template_name):
		
		with open(template_name, 'r') as file:
			self.contents = file.readlines()
		
	def set_exp(self, exp_file):
		search_id = self.contents.index('	\'experiment\':\'2_exp.csv\',\n')
		new_line = '	\'experiment\':\'%s\',\n'%(exp_file)
		self.contents[search_id] = new_line
		
	def set_log(self, batch_label):
		search_id = self.contents.index('	\'time_file\':\'9_time.log\',\n')
		new_line = '	\'time_file\':\'5_batch%s_time.log\',\n'%(batch_label)
		self.contents[search_id] = new_line
		
	def write_file(self, batch_label):
		filename = '3_pycode_exp%s.py'%(batch_label)
		with open(filename,'w+') as file:
			for line in self.contents:
				file.write(line)

## Script execution
if __name__ == '__main__':

	# Read the experimental design vectors
	exp_df = pd.read_csv(TPL['experiment'], index_col='id')

	for bid in range(STG['batches']):
		batch_label = str(bid+1).zfill(STG['zeros'])
		batch_name = '3_batch%s.csv'%(batch_label)
		batch_df = exp_df.iloc[bid::STG['batches'], :]
		batch_df.to_csv(batch_name)
		
		pycode = ExperimentScript(TPL['script'])
		pycode.set_exp(batch_name)
		pycode.set_log(batch_label)
		pycode.write_file(bid+1)
	
	print('Done!')