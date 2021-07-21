import argparse, os, gc
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel
import pandas as pd
import numpy as np
from torch.utils.data import (
	Dataset, DataLoader
)


class LitDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, inference_only=False):
        super().__init__()

        self.df = df        
        self.inference_only = inference_only
        self.text = df.excerpt.tolist()
        #self.text = [text.replace("\n", " ") for text in self.text]
        
        if not self.inference_only:
            self.target = torch.tensor(df.target.values, dtype=torch.float32)        
    
        self.encoded = tokenizer.batch_encode_plus(
            self.text,
            padding = 'max_length',            
            max_length = max_length,
            truncation = True,
            return_attention_mask=True
        )        
 

    def __len__(self):
        return len(self.df)

    
    def __getitem__(self, index):        
        input_ids = torch.tensor(self.encoded['input_ids'][index])
        attention_mask = torch.tensor(self.encoded['attention_mask'][index])
        
        if self.inference_only:
            return (input_ids, attention_mask)            
        else:
            target = self.target[index]
            return (input_ids, attention_mask, target)
			
class LitModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()

        config = AutoConfig.from_pretrained(model_path)
        config.update({"output_hidden_states":True, 
                       "hidden_dropout_prob": 0.0,
                       "layer_norm_eps": 1e-7})                       
        
        self.roberta = AutoModel.from_pretrained(model_path, config=config)  
            
        self.attention = nn.Sequential(            
            nn.Linear(768, 512),            
            nn.Tanh(),                       
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )        

        self.regressor = nn.Sequential(                        
            nn.Linear(768, 1)                        
        )
        

    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(input_ids=input_ids,
                                      attention_mask=attention_mask)        

        # There are a total of 13 layers of hidden states.
        # 1 for the embedding layer, and 12 for the 12 Roberta layers.
        # We take the hidden states from the last Roberta layer.
        last_layer_hidden_states = roberta_output.hidden_states[-1]

        # The number of cells is MAX_LEN.
        # The size of the hidden state of each cell is 768 (for roberta-base).
        # In order to condense hidden states of all cells to a context vector,
        # we compute a weighted average of the hidden states of all cells.
        # We compute the weight of each cell, using the attention neural network.
        weights = self.attention(last_layer_hidden_states)
                
        # weights.shape is BATCH_SIZE x MAX_LEN x 1
        # last_layer_hidden_states.shape is BATCH_SIZE x MAX_LEN x 768        
        # Now we compute context_vector as the weighted average.
        # context_vector.shape is BATCH_SIZE x 768
        context_vector = torch.sum(weights * last_layer_hidden_states, dim=1)        
        
        # Now we reduce the context vector to the prediction score.
        return self.regressor(context_vector)
		
		
def predict(model, data_loader, device):
    """Returns an np.array with predictions of the |model| on |data_loader|"""
    model.eval()

    result = np.zeros(len(data_loader.dataset))    
    index = 0
    
    with torch.no_grad():
        for batch_num, (input_ids, attention_mask) in enumerate(data_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
                        
            pred = model(input_ids, attention_mask)                        

            result[index : index + pred.shape[0]] = pred.flatten().to("cpu")
            index += pred.shape[0]

    return result



def main(args):
	NUM_MODELS = 2
	DEVICE = 'cuda'
	MAX_LEN = 248


	tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
	
	inf_df = pd.read_csv(args.inference_data)
	inf_dataset = LitDataset(inf_df, tokenizer, MAX_LEN, inference_only=True)
	
	
	#all_predictions = np.zeros((NUM_MODELS, len(inf_df)))
	inf_loader = DataLoader(inf_dataset, batch_size=args.batch_size,
		drop_last=False, shuffle=False, num_workers=2)

	for model_index in range(NUM_MODELS):            
		model_path = os.path.join(args.model_dir, 'model_{}.pth'.format(model_index+1))
		print(f"\nUsing {model_path}")
							
		model = LitModel(args.model_type)
		model.load_state_dict(torch.load(model_path, map_location=DEVICE))    
		model.to(DEVICE)
			
		# all_predictions[model_index] = predict(model, test_loader)
		inf_df['model_{}'.format(model_index+1)] = predict(model, inf_loader, DEVICE)
				
		del model
		gc.collect()
		
	def avg_pred(x, num_model):
		preds = []
		for i in range(num_model):
			preds.append(x['model_{}'.format(i+1)])
		
		return round(sum(preds)/len(preds), 9)

	inf_df['target'] = inf_df.apply(lambda x: avg_pred(x, NUM_MODELS), axis=1)
	inf_df.to_csv(args.output_path, index=False)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='parameters for data and weights location',
								 prefix_chars='-',
								 )

	parser.add_argument('-inference_data', default='data/raw/train.csv', help='Train Data Location')
	parser.add_argument('-output_path', default='data/inference/inf.csv')
	parser.add_argument('-model_dir', default='model/baseline')
	parser.add_argument('-model_type', default='roberta-base')
	parser.add_argument('-tokenizer', default='roberta-base')
	parser.add_argument('-batch_size', default=16)
	
	
	args = parser.parse_args()
	main(args)

			
			
			