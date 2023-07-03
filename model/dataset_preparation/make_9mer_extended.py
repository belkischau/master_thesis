import pandas as pd 

negative_with_label = pd.read_csv('../../dataset/model/negative_extended_with_label.csv')

fixed_negative_label = negative_with_label[(negative_with_label['peptide'].str.len() == 9 ) & (negative_with_label['extended'].str.len() == 21)]
fixed_negative_label = fixed_negative_label[~fixed_negative_label['extended'].str.contains('U|J|X')]

fixed_negative_label.to_csv('../../dataset/model/negative_extended_9mer_with_label.csv', index=False)

positive_with_label = pd.read_csv('../../dataset/model/positive_extended_with_label.csv')

fixed_positive_label = positive_with_label[(positive_with_label['peptide'].str.len() == 9) & (positive_with_label['extended'].str.len() == 21)]
fixed_positive_label = fixed_positive_label[~fixed_positive_label['extended'].str.contains('U|J|X')]

fixed_positive_label.to_csv('../../dataset/model/positive_extended_9mer_with_label.csv', index=False)

