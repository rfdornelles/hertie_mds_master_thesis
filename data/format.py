import pandas as pd
from datasets import Dataset

# function to save the dataset
def save_dataset(path, input_col = 'julgado', split = None):
    # recover the name of dataset from the file
    name = path.split('.parquet')[0]
    
    print(f"Processing {name} dataset...")
    
    # load
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f'Error: {e}')
        return
      
    # transform the columns into json output
    input_data = df[["processo", input_col]]
    output_data = df.drop(columns=[input_col, 'id', 'processo']).apply(lambda x: x.to_dict(), axis=1)
    
    # build dictionary
    base_output = Dataset.from_pandas(pd.DataFrame({'processo': input_data['processo'],'input': input_data[input_col], 'output': output_data}), split=split)
    
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