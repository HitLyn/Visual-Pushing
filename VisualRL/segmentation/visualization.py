import tkinter as tk
from PIL import Image, ImageTk
import os
import numpy as np
from VisualRL.vae.model import VAE
import torch
from torchvision import transforms
from IPython import embed

IMAGE_PATH = "/home/lyn"

device = torch.device('cuda')
# global model
model = VAE(device = device, image_channels = 3, h_dim = 1024, z_dim = 4)

window = tk.Tk()
window.title("VAE Latent Space")

frame = tk.Frame(master = window, width = 900, height = 550)
frame.pack()

l1 = tk.Label(master = frame, text = '')
l1.place(x = 100, y = 350)

# image = Image.open("/home/lyn/0001.png").resize([100, 100])
# image = ImageTk.PhotoImage(image)
label_input = tk.Label(master = frame)
label_input.place(x = 50, y = 50)

label_output = tk.Label(master = frame)
label_output.place(x = 550, y = 50)

def set_s1_value(val):
    # l1.configure(text = val)
    mu = np.zeros(4)
    mu[0] = val
    mu[1] = s2.get()
    mu[2] = s3.get()
    mu[3] = s4.get()
    show_out_put(mu)

def set_s2_value(val):
    # l1.configure(text = val)
    mu = np.zeros(4)
    mu[0] = s1.get()
    mu[1] = val
    mu[2] = s3.get()
    mu[3] = s4.get()
    show_out_put(mu)

def set_s3_value(val):
    # l1.configure(text = val)
    mu = np.zeros(4)
    mu[0] = s1.get()
    mu[1] = s2.get()
    mu[2] = val
    mu[3] = s4.get()
    show_out_put(mu)

def set_s4_value(val):
    # l1.configure(text = val)
    mu = np.zeros(4)
    mu[0] = s1.get()
    mu[1] = s2.get()
    mu[2] = s3.get()
    mu[3] = val
    show_out_put(mu)

def show_out_put(mu):
    with torch.no_grad():
        image_recon_tensor = model.decode(torch.tensor(mu).to(dtype = torch.float32).to(device).unsqueeze(0))
    image_recon_pil = transforms.ToPILImage()(image_recon_tensor.squeeze().cpu())

    image_recon_tk = ImageTk.PhotoImage(image_recon_pil)
    label_output.configure(image = image_recon_tk)
    label_output.photo = image_recon_tk
    print('update output image...')


def select_image():
    # print('selecting new')
    n = np.random.randint(1,4)
    file_name = os.path.join(IMAGE_PATH, "{:0>4d}.png".format(n))
    image = Image.open(file_name).resize([64, 64])
    image_tk = ImageTk.PhotoImage(image)
    label_input.configure(image = image_tk)
    label_input.photo = image_tk
    # reset slider
    tensor_image = transforms.ToTensor()(image.convert("RGB")).unsqueeze(0).to(device)
    # embed()
    with torch.no_grad():
        image_recon, z, mu, logvar = model(tensor_image)
    latent_value = mu[0].cpu().numpy()
    # print(latent_value[0])
    s1.set(latent_value[0]*100)
    s2.set(latent_value[1]*100)
    s3.set(latent_value[2]*100)
    s4.set(latent_value[3]*100)

    image_recon_pil = transforms.ToPILImage()(image_recon.squeeze().cpu())
    image_recon_tk = ImageTk.PhotoImage(image_recon_pil)
    label_output.configure(image = image_recon_tk)
    label_output.photo = image_recon_tk




# slider
s1 = tk.Scale(window, from_ = -5, to= 5, orient=tk.HORIZONTAL, length = 100, resolution = 0.01, command = set_s1_value)
s1.place(x = 350, y = 50)
s2 = tk.Scale(window, from_ = -5, to= 5, orient=tk.HORIZONTAL, length = 100, resolution = 0.01, command = set_s2_value)
s2.place(x = 350, y = 150)
s3 = tk.Scale(window, from_ = -5, to= 5, orient=tk.HORIZONTAL, length = 100, resolution = 0.01, command = set_s3_value)
s3.place(x = 350, y = 250)
s4 = tk.Scale(window, from_ = -5, to= 5, orient=tk.HORIZONTAL, length = 100, resolution = 0.01, command = set_s4_value)
s4.place(x = 350, y = 350)

# button
button = tk.Button(frame, text = "?", command = select_image)
button.place(x = 350, y = 450)












window.mainloop()
