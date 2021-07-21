import pandas as pd


if __name__ == '__main__':
	file_name = 'lambada_generated_result.csv'
	data = pd.read_csv(file_name)
	
	data = data[data['label'] == data['pred']]
	data['excerpt'] = data['text'].apply(lambda x: x[:x.find('</s>')].strip())
	data = data[['excerpt']]
	data = data.reset_index()
	data['url_legal'] = ''
	data['license'] = ''
	data.rename(columns={'index':'id'}, inplace=True)
	data = data[['id', 'url_legal', 'license', 'excerpt']]
	data.to_csv('fixed.csv', index=False)