from pathlib import Path
import pandas as pd
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import TestData


def get_train_data():
    label_data = {}
    for i in range(525):
        img_paths = Path(f'dataset/train/{i}').glob('*.jpg')
        img_paths = sorted(list(img_paths))
        imgs = [cv2.imread(str(img_paths[j])) for j in range(4)]
        imgs = [cv2.resize(img, (112, 112)) for img in imgs]
        label_data[i] = cv2.hconcat([cv2.vconcat(imgs[:2]), cv2.vconcat(imgs[2:])])
    return label_data

def test(model, log_dir, device='cuda'):
    log_dir_test = log_dir / Path('testing_result')
    if not log_dir_test.exists():
        log_dir_test.mkdir(parents=True)

    test_set = TestData('dataset/test')
    test_loader = DataLoader(test_set, 10, shuffle=False, num_workers=2)

    anns = []
    label_data = get_train_data()
    for img_b, img_id_b in tqdm(iter(test_loader)):
        pred_b = model(img_b.to(device)).cpu()
        _, cls_b = pred_b.max(1)

        for img_id, cls in zip(img_id_b, cls_b):
            anns.append([img_id, cls.item()])
            
            img = cv2.imread(f'dataset/test/{img_id}.jpg')
            img = cv2.resize(img, (224, 224))

            # concatenate the testing image with the training image of the predicted class
            # if they are in the same class -> correct prediction
            result_img = cv2.hconcat([img, label_data[cls.item()]])
            
            # write the predicted class on the image
            cv2.putText(result_img, str(cls.item()), (10, 224-10), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255, 255, 255), thickness=2)
            cv2.imwrite(str(log_dir_test / f'{img_id}.jpg'), result_img)

    df_pred = pd.DataFrame(anns)
    df_pred.columns = ['id', 'predict']
    df_pred.to_csv(str(log_dir_test / 'test_pred.csv'), index=False)
    print(f"save at {str(log_dir_test / 'test_pred.csv')}")