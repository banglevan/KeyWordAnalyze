import pandas as pd
import numpy as np
import warnings
import torch
import mobileclip
from tqdm import tqdm

warnings.filterwarnings('ignore')

class AdsDataDetail:
    def __init__(self):
        self.encoder_version = 's0'
        self.model, _, self.preprocess = mobileclip.create_model_and_transforms(f'mobileclip_{self.encoder_version}',
                                                                      pretrained=f'checkpoints/mobileclip_{self.encoder_version}.pt')
        self.tokenizer = mobileclip.get_tokenizer(f'mobileclip_{self.encoder_version}')

    def read_csv(self, path):
        df = pd.read_csv(path)
        # nan_percent = df.isna().mean() * 100
        df = df[['keyword', 'cpc', 'country']].dropna()
        # df = pd.get_dummies(df, columns=['country'], drop_first=True)
        # countries = df['country'].unique()
        # 1. Datetime Feature Engineering
        # df['time'] = pd.to_datetime(df['time'], errors='coerce')
        # df['year'] = df['time'].dt.year
        # df['month'] = df['time'].dt.month
        # df['day'] = df['time'].dt.day
        # df['hour'] = df['time'].dt.hour
        # df['day_of_week'] = df['time'].dt.weekday
        # df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        # df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        # df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        return df

    @torch.no_grad()
    def runner_mobile_clip(self, text):
        text = self.tokenizer([text])
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            vector = text_features.detach().cpu().numpy()
        return vector
    def country_one_hot(self, df):
        df = df[['country']]
        df_encoded = pd.get_dummies(df, columns=['country'], drop_first=True)
        df_encoded.to_parquet('features/encoded_dataframe.parquet', index=False)

    def keywords_encoder(self, df):
        tqdm.pandas()
        df['mobile_clip_vector'] = df['keyword'].apply(lambda x: self.runner_mobile_clip(x))
        vectors = df['mobile_clip_vector']
        vectors.to_parquet('features/mobile_clip/vectors.parquet', index=False)

    def parquet_loader(self, path):
        df = pd.read_parquet(path)

if __name__ == '__main__':
    detailer = AdsDataDetail()
    df = detailer.read_csv('data.csv')
    # detailer.country_one_hot(df)
    detailer.keywords_encoder(df)
