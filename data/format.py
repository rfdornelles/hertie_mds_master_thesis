import pandas as pd
from datasets import Dataset

# function to save the dataset
def save_dataset(path, input_col = 'julgado', split = None):
    # recover the name of dataset from the file
    name = path.split('.parquet')[0]
    
    # load
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f'Error: {e}')
        return
      
    # transform the columns into json output
    input = df[input_col]
    output = df.drop(columns=[input_col, 'id', 'processo']).apply(lambda x: x.to_dict(), axis=1)
    
    # build dictionary
    base_output = Dataset.from_dict({'input': input, 'output': output}, split = split)
    
    # save
    try: 
      base_output.save_to_disk(name)
    except Exception as e:
        print(f'Error: {e}')
        return
    print(f'Successfully saved dataset {name}!')
    

save_dataset('validation.parquet')
save_dataset('train.parquet')
save_dataset('test.parquet')
save_dataset('lacerda_clean.parquet')