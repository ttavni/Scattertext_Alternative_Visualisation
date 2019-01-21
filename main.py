import scattertext as st
from scattertext import ScatterChartExplorer
import nltk
import pandas as pd
import json
import numpy as np

if __name__ == "__main__":

	# Dependencies
	df = pd.read_csv('data/political_data.csv', encoding='latin1')
	df.dropna(inplace=True)
	df['text'] = df['text'].apply(lambda x: nltk.sent_tokenize(x))
	df = pd.DataFrame({col:np.repeat(df[col].values, df['text'].str.len()) for col in df.columns.difference(['text'])}).assign(**{'text': np.concatenate(df['text'].values)})[df.columns.tolist()]

	threshold = 0.2 * len(df)
	corpus = st.CorpusFromPandas(df, category_col='party', text_col='text',nlp=st.WhitespaceNLP.whitespace_nlp).build()
	corpus = corpus.remove_terms_used_in_less_than_num_docs(threshold=threshold)
	sce = ScatterChartExplorer(corpus)
	data = pd.DataFrame(sce.to_dict(category='democrat')['data'])

	final = pd.DataFrame({'word': data['term'],'x': data['x'],'y': data['y']})

	text_list = df['text'].tolist()
	final['example_numbers'] = final['word'].apply(lambda BxH: [text_list.index(s) for s in text_list if BxH.lower() in s.lower()])

	df['meta'] = df['party'].astype(str) + '  (' + df['speaker'].astype(str) + ')  '
	final['cats'] = final['example_numbers'].apply(lambda x: [df['meta'][i] for i in x])

	final['examples'] = final['word'].apply(
		lambda x: ("<br> <br> <br>".join(s.lower() for s in df['text'].tolist() if x.lower() in s.lower())).replace(
			x.lower(), '<font color="#F39C12"><strong >{}</strong></font>'.format(x.lower())))


	final['cats_html'] = final['cats'].apply(lambda x: ['<strong>{}</strong>: '.format(i) for i in x])
	final['example'] = ([("<br> <br>".join([str(a) + b for a, b in zip(final['cats_html'][i], final['examples'][i].split('<br> <br>'))])) for i in range(len(final))])
	final['example'] = final['example'].apply(lambda x: x.replace('democrat  ','<font color="#34AAE0">DEMOCRAT </font>'))
	final['example'] = final['example'].apply(lambda x: x.replace('republican  ', '<font color="#E91D0E">REPUBLICAN </font>'))
	final = final[['word','x','y','example']]

	final_json = final.head(50).to_dict(orient='records')
	with open('visualisation/data.json', 'w') as fout:
		json.dump(final_json, fout)