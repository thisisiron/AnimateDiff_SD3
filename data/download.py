import os
import requests
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm 
from concurrent.futures import ThreadPoolExecutor, as_completed

dataset = load_dataset("ilovehentai9000/ilove-anime-sakuga-1TiB")

def download_video(url, output_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        return output_path
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return None

def preprocess_dataset(dataset, dataset_type):
    df = dataset.to_pandas()

    before_count = len(df)
    df.drop_duplicates(subset=['identifier', 'file_ext'], inplace=True)
    after_count = len(df)
    removed_count = before_count - after_count

    if removed_count > 0:
        print(f"{dataset_type} 데이터셋에서 {removed_count}개의 중복된 행이 제거되었습니다.")

    output_csv_path = f"{dataset_type}_dataset.csv"
    df.to_csv(output_csv_path, index=False)
    print(f"{dataset_type} 데이터셋이 {output_csv_path}에 저장되었습니다.")

    return df

def download_videos_from_dataset(df, dataset_type, output_dir, max_count=None, max_workers=8):

    os.makedirs(output_dir, exist_ok=True)

    successful_downloads = []

    def download_task(row):
        identifier = row['identifier']
        url_link = row['url_link']
        file_ext = row['file_ext']

        file_name = f"{identifier}.{file_ext}"
        output_path = os.path.join(output_dir, file_name)

        if not os.path.exists(output_path):
            result = download_video(url_link, output_path)
            if result:
                return row
        else:
            return row
        return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_task, row): row for _, row in df.iterrows()}

        progress_bar = tqdm(total=min(len(futures), max_count or len(futures)), desc=f"Downloading {dataset_type} videos")

        for future in as_completed(futures):
            if max_count and len(successful_downloads) >= max_count:
                executor.shutdown(wait=False, cancel_futures=True)
                break

            try:
                result = future.result()
                if result is not None:
                    successful_downloads.append(result)
            except Exception as e:
                print(f"Download failed: {e}")

            progress_bar.update(1)

        progress_bar.close()

    success_df = pd.DataFrame(successful_downloads)

    output_csv_path = f"{dataset_type}.csv"
    success_df.to_csv(output_csv_path, index=False)
    print(f"{dataset_type} 데이터셋의 다운로드 성공 데이터가 {output_csv_path}에 저장되었습니다.")

    return success_df

train_df = preprocess_dataset(dataset["train"], "train")
validation_df = preprocess_dataset(dataset["validation"], "validation")
# test_df = preprocess_dataset(dataset["test"], "test")

train_output_dir = "dataset/train"
validation_output_dir = "dataset/validation"
# test_output_dir = "dataset/test"

train_success_df = download_videos_from_dataset(train_df, "train", train_output_dir, max_count=1000, max_workers=4)
validation_success_df = download_videos_from_dataset(validation_df, "validation", validation_output_dir, max_count=500, max_workers=4)
# test_success_df = download_videos_from_dataset(test_df, "test", test_output_dir, max_count=100, max_workers=4)

