import os.path

import pandas as pd
import numpy as np
import warnings
import torch
import mobileclip
import open_clip
from mobileclip.modules.common.mobileone import reparameterize_model
from tqdm import tqdm

warnings.filterwarnings('ignore')

class AdsDataDetail:
    def __init__(self, model_type='mobile_clip'):
        self.model_type = model_type
        if model_type == 'mobile_clip':
            self.encoder_version = 's1'
            self.model_mclip, _, self.preprocess_mclip = mobileclip.create_model_and_transforms(f'mobileclip_{self.encoder_version}',
                                                                          pretrained=f'checkpoints/mobileclip_{self.encoder_version}.pt')
            self.tokenizer_mclip = mobileclip.get_tokenizer(f'mobileclip_{self.encoder_version}')
        else:
            self.encoder_version = 's2'
            model_oclip, _, self.preprocess_oclip = open_clip.create_model_and_transforms('MobileCLIP-S2', pretrained='datacompdr')
            self.tokenizer_oclip = open_clip.get_tokenizer('MobileCLIP-S2')
            # For inference/model exporting purposes, please reparameterize first
            model_oclip.eval()
            self.model_oclip = reparameterize_model(model_oclip)

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
    def runner_open_clip(self, text):
        if self.model_type == 'mobile_clip':
            raise ValueError(">> not init open clip yet")
        text = self.tokenizer_oclip([text])
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.model_oclip.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            vector = text_features.detach().cpu().numpy()
        return vector

    @torch.no_grad()
    def runner_mobile_clip(self, text):
        if self.model_type != 'mobile_clip':
            raise ValueError(">> not init mobile clip yet")
        text = self.tokenizer_mclip([text])
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.model_mclip.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            vector = text_features.detach().cpu().numpy()[0]
        return vector
    def country_one_hot(self, df):
        df = df[['country']]
        df_encoded = pd.get_dummies(df, columns=['country'], drop_first=True)
        df_encoded.to_parquet('features/encoded_dataframe.parquet', index=False)

    def keywords_encoder(self, df):
        tqdm.pandas()
        # df = df.iloc[:10]
        df['mobile_clip_vector'] = df['keyword'].progress_apply(lambda x: self.runner_mobile_clip(x))
        vectors = df[['mobile_clip_vector']]
        print(vectors)
        if self.model_type == 'mobile_clip':
            middle_path = 'mobile_clip'
        else:
            middle_path = 'open_clip'
        save_path = os.path.join('features', middle_path)
        os.makedirs(save_path, exist_ok=True)
        to_save = os.path.join(save_path, f'vectors_{self.encoder_version}.parquet')
        vectors.to_parquet(to_save, index=False)

    def parquet_loader(self, path):
        # Load the Parquet file
        df = pd.read_parquet(path)
        matrix = df.to_numpy()
        sqzed = np.apply_along_axis(squeeze, 1, matrix)
        print(sqzed.shape)

def squeeze(row):
    return row[0]

if __name__ == '__main__':
    detailer = AdsDataDetail()
    df = detailer.read_csv('data.csv')
    # detailer.country_one_hot(df)
    # detailer.keywords_encoder(df)
    detailer.parquet_loader('features/mobile_clip/vectors_s0.parquet')