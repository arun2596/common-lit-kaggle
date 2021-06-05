import argparse
import json
import pandas as pd
import textstat
from tqdm import tqdm
import nlpaug.augmenter.word as naw

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        

class Augmenter:
    def __init__(self, config_path):
        """
            augs: list of (category, model) e.g. [('back_translation', None), ('context_word_embs', 'bert-base-cased')]
        """
        
        with open(config_path) as f:
            self.config = json.load(f)
        
        self.augs = []
        for cat, model_type in self.config['augs']:
            if cat == 'back_translation':
                self.augs.append(naw.BackTranslationAug(device=self.config['device']))
            elif cat == 'context_word_embs':
                self.augs.append(naw.ContextualWordEmbsAug(model_path=model_type, device=self.config['device']))
    
    def filter(self, aug_data):
        def cal_change_score(x, threshold, voters):
                score = 0
                for v in voters:
                    if x['{}_score_change'.format(v)] <= threshold and x['{}_score_change'.format(v)] >= -threshold:
                        score += 1

                return score

        aug_data['change_score'] = aug_data.apply(lambda x: cal_change_score(
            x, self.config['change_threshold'], self.config['validation_methods']), axis=1)
        
        return aug_data[aug_data['change_score'] >= self.config['min_voter']]
        
    def generate(self, data):
        aug_data = self._generate(data)
        aug_data = self.filter(aug_data)
        aug_data = aug_data[['id', 'url_legal', 'license', 'excerpt', 'target', 'standard_error']]
        return aug_data
        
    def _generate(self, data):
        """
            validation_methods: list of ['fre', 'smog', 'gp', 'cf', 'cli']. e.g. ['fre', 'smog', 'gp', 'cf', 'cli']
            change_threshold: acceptable change in percentage
        """
        
        list_of_augmented_data = []
        
        data['fre_score'] = data['excerpt'].apply(lambda x: textstat.flesch_reading_ease((x)))
        data['smog_score'] = data['excerpt'].apply(lambda x: textstat.smog_index((x)))
        data['gp_score'] = data['excerpt'].apply(lambda x: textstat.gutierrez_polini((x)))
        data['cf_score'] = data['excerpt'].apply(lambda x: textstat.crawford((x)))
        data['cli_score'] = data['excerpt'].apply(lambda x: textstat.coleman_liau_index((x)))
        
        for i, aug in enumerate(self.augs):
            augmented_data = []
            for d in tqdm(chunks(data['excerpt'].tolist(), self.config['batch_size']), desc='{}/{}:{}'.format(i+1, len(self.augs), aug.name)):
                augmented_data.extend(aug.augment(d))
            augmented_data = pd.DataFrame(augmented_data, columns=['excerpt'])

            for c in ['url_legal', 'license']:
                augmented_data[c] = ''

            augmented_data['id'] = data['id'].tolist()
            augmented_data['target'] = data['target'].tolist()
            augmented_data['standard_error'] = -1

            augmented_data['fre_score'] = augmented_data['excerpt'].apply(lambda x: textstat.flesch_reading_ease((x)))
            augmented_data['smog_score'] = augmented_data['excerpt'].apply(lambda x: textstat.smog_index((x)))
            augmented_data['gp_score'] = augmented_data['excerpt'].apply(lambda x: textstat.gutierrez_polini((x)))
            augmented_data['cf_score'] = augmented_data['excerpt'].apply(lambda x: textstat.crawford((x)))
            augmented_data['cli_score'] = augmented_data['excerpt'].apply(lambda x: textstat.coleman_liau_index((x)))
            
            augmented_data = augmented_data.reset_index()

            def cal_percentage_change(x, col, orig_scores):
                idx = x['index']
                orig_score = orig_scores[idx]

                change = (x['{}_score'.format(col)] - orig_score) / orig_score    
                return change

            for c in self.config['validation_methods']:
                orig_scores = data['{}_score'.format(c)].tolist()
                augmented_data['{}_orig_score'.format(c)] = orig_scores
                augmented_data['{}_score_change'.format(c)] = augmented_data.apply(
                    lambda x: cal_percentage_change(x, c, orig_scores), axis=1)

            list_of_augmented_data.append(augmented_data)
    
        return pd.concat(list_of_augmented_data)