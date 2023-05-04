# 上海工程技术大学
# 崔嘉亮
# 开发时间：2022/6/12 11:06
from torch.utils.data import Dataset
import pickle
from torchvision.transforms import transforms

RAIL = 'RAIL'
GAF = 'GAF'
LABEL = 'LABEL'
TRAIN = 'Train'
VALID = 'Valid'
TEST = 'Test'


def total(params):
    """
    count the total number of hyperparameter settings
    """
    settings = 1
    for k, v in params.items():
        settings *= len(v)
    return settings


def load_Rail(data_path):
    # parse the input args

    class IMG(Dataset):

        def __init__(self, rail, gaf, labels, transform):
            self.rail = rail
            self.gaf = gaf
            self.labels = labels
            self.transform = transform

        def __getitem__(self, idx):
            rail_img = self.rail[idx]
            gaf_img = self.gaf[idx]

            label = self.labels[idx]

            if self.transform:
                rail_img = self.transform(rail_img)
                gaf_img = self.transform(gaf_img)

            return rail_img, gaf_img, label

        def __len__(self):
            return len(self.rail)

    Rail_GAF_data = pickle.load(open(data_path + "Rail_GAF_PIL.pkl", 'rb'))

    Rail_GAF_train, Rail_GAF_valid, Rail_GAF_test = Rail_GAF_data[TRAIN], Rail_GAF_data[VALID], Rail_GAF_data[TEST]

    train_rail, train_gaf, train_labels = Rail_GAF_train[RAIL], Rail_GAF_train[GAF], Rail_GAF_train[LABEL]
    valid_rail, valid_gaf, valid_labels = Rail_GAF_valid[RAIL], Rail_GAF_valid[GAF], Rail_GAF_valid[LABEL]
    test_rail, test_gaf, test_labels = Rail_GAF_test[RAIL], Rail_GAF_test[GAF], Rail_GAF_test[LABEL]

    Image_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomResizedCrop((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # code that instantiates the Dataset objects
    train_set = IMG(train_rail, train_gaf, train_labels, Image_transform)
    valid_set = IMG(valid_rail, valid_gaf, valid_labels, Image_transform)
    test_set = IMG(test_rail, test_gaf, test_labels, Image_transform)

    rail_dim = train_set[0][0].shape[0]
    print("Rail image  feature dimension is: {}".format(rail_dim))
    gaf_dim = train_set[0][1].shape[0]
    print("GAF image feature dimension is: {}".format(gaf_dim))

    input_dims = (rail_dim, gaf_dim)

    return train_set, valid_set, test_set, input_dims
