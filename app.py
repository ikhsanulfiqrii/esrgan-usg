import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
from torchvision.transforms.functional import to_tensor, to_pil_image

# --- 1. DEFINISI ARSITEKTUR (Wajib ada agar model bisa load state_dict) ---
class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels + growth_rate, growth_rate, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels + 2 * growth_rate, growth_rate, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels + 3 * growth_rate, growth_rate, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels + 4 * growth_rate, in_channels, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.relu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.relu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.dropout(self.conv5(torch.cat([x, x1, x2, x3, x4], 1)))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, in_channels):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock(in_channels)
        self.RDB2 = ResidualDenseBlock(in_channels)
        self.RDB3 = ResidualDenseBlock(in_channels)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_features=64, num_blocks=16):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, num_features, kernel_size=3, stride=1, padding=1)
        self.RRDB_blocks = nn.Sequential(*[RRDB(num_features) for _ in range(num_blocks)])
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(num_features, num_features * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
        )
        self.conv3 = nn.Conv2d(num_features, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out1 = self.relu(self.conv1(x))
        out = self.RRDB_blocks(out1)
        out = self.conv2(out)
        out = out1 + out
        out = self.upsample(out)
        out = self.conv3(out)
        return out

# --- 2. FUNGSI LOAD MODEL ---
@st.cache_resource
def load_esrgan():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Sesuaikan path model .pth kamu di sini
    model_path = 'models/generator_epoch_156.pth' 
    
    model = Generator(in_channels=3, out_channels=3, num_features=64, num_blocks=16)
    
    # Memuat bobot (weights) ke dalam arsitektur
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval().to(device)
    return model, device

# --- 3. UI STREAMLIT ---
st.title("🖼️ ESRGAN Image Upscaler")
st.write(" Web untuk Super Resolution (USG)")

model, device = load_esrgan()

uploaded_file = st.file_uploader("Upload gambar resolusi rendah kamu...", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    input_img = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(input_img, caption="Gambar Asli", use_container_width=True)
    
    if st.button("✨ Lakukan Super Resolution"):
        with st.spinner("Sedang memproses..."):
            # Pre-processing
            img_tensor = to_tensor(input_img).unsqueeze(0).to(device)
            
            # Inference
            with torch.no_grad():
                output_tensor = model(img_tensor).clamp(0, 1)
            
            # Post-processing
            output_img = to_pil_image(output_tensor.squeeze().cpu())
            
            with col2:
                st.image(output_img, caption="Hasil ESRGAN (4x)", use_container_width=True)
                
                # Tombol Download
                buf = io.BytesIO()
                output_img.save(buf, format="PNG")
                st.download_button(
                    label="💾 Download Hasil",
                    data=buf.getvalue(),
                    file_name=f"SR_{uploaded_file.name}",
                    mime="image/png"
                )