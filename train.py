from monai.losses import DiceLoss, DiceCELoss, GeneralizedWassersteinDiceLoss
from torch.nn.functional import one_hot
import torch.optim as optim
from monai.metrics import compute_dice
from tqdm import tqdm
import csv
import torch
from Network.unetr import UNETR
from Dataloader.dataloader import data_loaders


device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")

model = UNETR(in_channels=1, out_channels=5, img_size=(256, 256, 128), hidden_size=384, mlp_dim=1024, num_heads=6,
              feature_size=8)
model.to(device)

training_loss = DiceCELoss(sigmoid=True, include_background=False)
loader_train, loader_valid = data_loaders()
loaders = {"train": loader_train, "valid": loader_valid}
optimizer = optim.Adam(model.parameters(), lr=0.001)

epoch_num = 50
class_num = 5

for epoch in tqdm(range(epoch_num)):
    print("{epc} is running".format(epc=epoch))
    loss_train = []
    loss_valid = []
    Dice_valid = []
    Dice_per_organ = {}
    step = 0
    img_print = 0
    # log_itter = random.randint(150, 200)

    for phase in ["train", 'valid']:
        if phase == "train":
            all_dice_per_organ = torch.zeros(1, 5)
            all_dice_per_organ = all_dice_per_organ.to(device)
            model.train()
        else:
            model.eval()

        for i, data in enumerate(loaders[phase]):
            x, y_true, patient_id = data
            patient_id = int(patient_id[0])
            x, y_true = x.to(device), y_true.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):
                pred = model(x)

                if phase == "valid":
                    pred = pred.argmax(dim=1)
                    pred = one_hot(pred, num_classes=class_num).permute(0, 4, 1, 2, 3)
                    y_true = torch.squeeze(y_true, dim=1)
                    y_true = one_hot(y_true, num_classes=class_num).permute(0, 4, 1, 2, 3)
                    accuracy = compute_dice(pred, y_true)

                    print(f"Validation Patient {patient_id}: {accuracy}")

                    all_dice_per_organ = torch.cat((all_dice_per_organ, accuracy), 0)

                if phase == "train":
                    y_true = torch.squeeze(y_true, dim=1)
                    y_true = one_hot(y_true, num_classes=class_num).permute(0, 4, 1, 2, 3)
                    loss = training_loss(pred, y_true)
                    if (i % 5 == 0):
                        print(loss)
                    loss.backward()
                    optimizer.step()

    avg_all_dice_per_organ = torch.mean(all_dice_per_organ[1:], 0)

    with open("/content/drive/MyDrive/UNETR/Result_UNETR_SegThor.csv", 'a') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerow([epoch, avg_all_dice_per_organ[0].item(), avg_all_dice_per_organ[1].item(),
                             avg_all_dice_per_organ[2].item(), avg_all_dice_per_organ[3].item(),
                             avg_all_dice_per_organ[4].item()])

#   # loss_train_final.append(t_loss)
#   # writer.add_scalar("train Loss", t_loss, epoch)
#   # dice_valid_final.append(V_dice)
#   # writer.add_scalar("validation IoU", V_dice, epoch)
#   # Valid_loss_final.append(v_loss)
#   # writer.add_scalar("validation loss", v_loss, epoch)

#   # for i in range(5):
#   #   writer.add_scalar("Organ IoU/{organ}".format(organ=i), float(avg_all_dice_per_organ[i].item()), epoch)

#   if (Max_avg_all_dice_per_organ < overall_dice_per_organ_mean) or (epoch % 10 == 0):
#     Max_avg_all_dice_per_organ = overall_dice_per_organ_mean
#     check_file = "/content/drive/MyDrive/Research_output/LCTSC/unet/dice_seg_{ep}_{dic}.pt".format(ep=epoch, dic = overall_dice_per_organ_mean)
#     # check_file = "/content/drive/MyDrive/dataset/Output/checkpoint_IoU/seg_{ep}_{dic}.pt".format(ep=epoch, dic = V_dice)
#     torch.save({
#               'epoch': epoch,
#               'model_state_dict': unet.state_dict(),
#               'optimizer_state_dict': optimizer.state_dict(),
#               'loss': loss,
#               't_loss': t_loss,
#               'v_loss': v_loss,
#               'V_dice': V_dice,
#               'dice_01': avg_all_dice_per_organ
#               }, check_file)

# # writer.flush()

# # trl = [i for i in range(1,len(loss_train_final)+1)]
# # plt.figure("test loss")
# # plt.plot(trl, loss_train_final)
# # plt.show()

# # vpl = [i for i in range(1,len(dice_valid_final)+1)]
# # plt.figure("Valid IoU")
# # plt.plot(vpl, dice_valid_final)
# # plt.show()
