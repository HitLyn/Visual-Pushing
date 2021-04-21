import tkinter as tk
from PIL import Image, ImageTk
import os
import numpy as np
from VisualRL.vae.model import VAE
import torch
from torchvision import transforms
from IPython import embed

IMAGE_PATH = "/homeL/cong/HitLyn/Visual-Pushing/images/masks"
SCALE_RANGE = 10

device = torch.device('cuda:1')
# global model
model = VAE(device = device, image_channels = 3, h_dim = 1024, z_dim = 6)
model.load("/homeL/cong/HitLyn/Visual-Pushing/results/vae/04_21-15_34/vae_model", 99, map_location='cuda:1')

window = tk.Tk()
window.title("VAE Latent Space")

frame = tk.Frame(master = window, width = 900, height = 750)
frame.pack()

l1 = tk.Label(master = frame, text = '')
l1.place(x = 150, y = 700)

# image = Image.open("/home/lyn/0001.png").resize([100, 100])
# image = ImageTk.PhotoImage(image)
label_input = tk.Label(master = frame)
label_input.place(x = 50, y = 50)

label_output = tk.Label(master = frame)
label_output.place(x = 550, y = 50)

def set_s1_value(val):
    # l1.configure(text = val)
    # print("latent_value", latent_value)
    mu = latent_value.copy()
    mu[0] = val
    show_out_put(mu)
    # print(latent_value)

def set_s2_value(val):
    # l1.configure(text = val)
    mu = latent_value.copy()
    mu[1] = val
    show_out_put(mu)

def set_s3_value(val):
    # l1.configure(text = val)
    mu = latent_value.copy()
    mu[2] = val
    show_out_put(mu)

def set_s4_value(val):
    # l1.configure(text = val)
    mu = latent_value.copy()
    mu[3] = val
    show_out_put(mu)

def set_s5_value(val):
    # l1.configure(text = val)
    mu = latent_value.copy()
    # embed()
    mu[4] = val
    show_out_put(mu)

def set_s6_value(val):
    # l1.configure(text = val)
    mu = latent_value.copy()
    mu[5] = val
    show_out_put(mu)

def show_out_put(mu):
    # print(mu)
    with torch.no_grad():
        image_recon_tensor = model.decode(torch.tensor(mu).to(dtype = torch.float32).to(device).unsqueeze(0))
    image_recon_pil = transforms.ToPILImage()(image_recon_tensor.squeeze().cpu())

    # image_recon_tk = ImageTk.PhotoImage(image_recon_pil)
    image_recon_tk = ImageTk.PhotoImage(image_recon_pil.resize([256, 256]))
    label_output.configure(image = image_recon_tk)
    label_output.photo = image_recon_tk
    # print('update output image...')


def select_image():
    # print('selecting new')
    global latent_value
    n = np.random.randint(1,10000)
    file_name = os.path.join(IMAGE_PATH, "{:0>5d}.png".format(n))
    l1.configure(text = "{:0>5d}.png".format(n))
    image = Image.open(file_name)
    image_tk = ImageTk.PhotoImage(image.resize([256,256]))
    # image_tk = ImageTk.PhotoImage(image)
    label_input.configure(image = image_tk)
    label_input.photo = image_tk
    # reset slider
    tensor_image = transforms.ToTensor()(transforms.Resize(64)(image.convert("RGB"))).unsqueeze(0).to(device)
    # embed()
    with torch.no_grad():
        image_recon, z, mu, logvar = model(tensor_image)
        image_recon = model.predict(tensor_image)
    latent_value = mu[0].cpu().numpy()
    print(latent_value)
    show_out_put(latent_value)

    # reset values
    # embed();exit()
    s1.set(latent_value[0])
    show_out_put(latent_value)
    s2.set(latent_value[1])
    show_out_put(latent_value)
    s3.set(latent_value[2])
    show_out_put(latent_value)
    s4.set(latent_value[3])
    show_out_put(latent_value)
    s5.set(latent_value[4])
    show_out_put(latent_value)
    s6.set(latent_value[5])
    show_out_put(latent_value)
    # print(latent_value)

    # show_out_put(latent_value)

    # print(s1.get(), s2.get(), s3.get(), s4.get(), s5.get(), s6.get())


    # image_recon_pil = transforms.ToPILImage()(image_recon.squeeze().cpu())
    # image_recon_tk = ImageTk.PhotoImage(image_recon_pil.resize([256,256]))
    # # image_recon_tk = ImageTk.PhotoImage(image_recon_pil)
    # label_output.configure(image = image_recon_tk)
    # label_output.photo = image_recon_tk




# slider
s1 = tk.Scale(window, from_ = -SCALE_RANGE, to= SCALE_RANGE, orient=tk.HORIZONTAL, length = 150, resolution = 0.01, command = set_s1_value)
s1.place(x = 350, y = 50)
s2 = tk.Scale(window, from_ = -SCALE_RANGE, to= SCALE_RANGE, orient=tk.HORIZONTAL, length = 150, resolution = 0.01, command = set_s2_value)
s2.place(x = 350, y = 150)
s3 = tk.Scale(window, from_ = -SCALE_RANGE, to= SCALE_RANGE, orient=tk.HORIZONTAL, length = 150, resolution = 0.01, command = set_s3_value)
s3.place(x = 350, y = 250)
s4 = tk.Scale(window, from_ = -SCALE_RANGE, to= SCALE_RANGE, orient=tk.HORIZONTAL, length = 150, resolution = 0.01, command = set_s4_value)
s4.place(x = 350, y = 350)
s5 = tk.Scale(window, from_ = -SCALE_RANGE, to= SCALE_RANGE, orient=tk.HORIZONTAL, length = 150, resolution = 0.01, command = set_s5_value)
s5.place(x = 350, y = 450)
s6 = tk.Scale(window, from_ = -SCALE_RANGE, to= SCALE_RANGE, orient=tk.HORIZONTAL, length = 150, resolution = 0.01, command = set_s6_value)
s6.place(x = 350, y = 550)
# button
button = tk.Button(frame, text = "random image", command = select_image)
button.place(x = 350, y = 650)












window.mainloop()
