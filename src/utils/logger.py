import os
import matplotlib.pyplot as plt

def export_train_plot(x, train_y, val_y, label, title, save_path):
    plt.figure(figsize=(10,4))
    
    plt.plot(x, train_y, label="train", color="blue")
    plt.plot(x, val_y, label="val", color="red")
    plt.xlabel("epoch")
    plt.ylabel(label)
    plt.title(title)
    
    plt.legend()
    plt.savefig(save_path)

def export_train_logs(save_dir, dataset, val_ratio, total_size, train_size, val_size, model_info, batch_size, classes, epochs, train_loss_list, train_acc_list, val_loss_list, val_acc_list, train_time_list, val_time_list, best_epoch):
    with open(os.path.join(save_dir, "log.txt"), "w", encoding="utf-8") as f:
        f.write("dataset: ")
        f.write(f"{dataset}\n")
        
        f.write("val_ratio: ")
        f.write(f"{val_ratio}\n")
        f.write("total_size: ")
        f.write(f"{total_size}\n")
        f.write("train_size: ")
        f.write(f"{train_size}\n")
        f.write("val_size: ")
        f.write(f"{val_size}\n")
        f.write("batch_size: ")
        f.write(f"{batch_size}\n")
        
        f.write("classes: [\n")
        for klass in classes:
            f.write(f"  '{klass}',\n")
        f.write("]\n")
        
        f.write(f"epochs: {epochs}\n")
        f.write(f"Best epoch: {best_epoch}\n")
        
        f.write("train loss list: [\n")
        for train_loss in train_loss_list:
            f.write(f"  {train_loss}\n")
        f.write("]\n")
        f.write("train acc list: [\n")
        for train_acc in train_acc_list:
            f.write(f"  {train_acc}\n")
        f.write("]\n")
        f.write("val loss list: [\n")
        for val_loss in val_loss_list:
            f.write(f"  {val_loss}\n")
        f.write("]\n")
        f.write("val acc list: [\n")
        for val_acc in val_acc_list:
            f.write(f"  {val_acc}\n")
        f.write("]\n")
        
        f.write("train time list: [\n")
        for train_time in train_time_list:
            f.write(f"  {train_time} min\n")
        f.write("]\n")
        f.write("val time list: [\n")
        for val_time in val_time_list:
            f.write(f"  {val_time} min\n")
        f.write("]\n")
        
        f.write("\nmodel: ")
        f.write(str(model_info))