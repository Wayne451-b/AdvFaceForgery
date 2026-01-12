import torch
import torch.nn as nn

class PhaseMix(nn.Module):
    def __init__(self, swap_strength=1.0, channels=3):
        super(PhaseMix, self).__init__()
        self.swap_strength = nn.Parameter(torch.tensor(swap_strength))
        self.amp_conv_1x1 = nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=False)
        self.phase_conv_3x3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, original_img, reference_img):
        original_fft = torch.fft.fft2(original_img, dim=(-2, -1))
        reference_fft = torch.fft.fft2(reference_img, dim=(-2, -1))
        original_amp = torch.abs(original_fft)
        original_phase = torch.angle(original_fft)
        reference_amp = torch.abs(reference_fft)
        reference_phase = torch.angle(reference_fft)
        # mixed_phase_ori = self.swap_strength * original_phase + (1 - self.swap_strength) * reference_phase
        # mixed_phase_ref = (1 - self.swap_strength) * original_phase + self.swap_strength * reference_phase
        original_mix_real = original_amp * torch.cos(reference_phase)
        original_mix_imag = original_amp * torch.sin(reference_phase)
        reference_mix_real = reference_amp * torch.cos(original_phase)
        reference_mix_imag = reference_amp * torch.sin(original_phase)
        original_amp_processed = self.amp_conv_1x1(original_mix_real)
        original_phase_processed = self.phase_conv_3x3(original_mix_imag)
        reference_amp_processed = self.amp_conv_1x1(reference_mix_real)
        reference_phase_processed = self.phase_conv_3x3(reference_mix_imag)
        original_mixed_fft = torch.complex(original_amp_processed, reference_phase_processed)
        reference_mixed_fft = torch.complex(reference_amp_processed, original_phase_processed)
        phase_swapped_ori = self.phase_conv_3x3(torch.fft.ifft2(original_mixed_fft, dim=(-2, -1)).real + original_img)
        phase_swapped_ref = self.phase_conv_3x3(torch.fft.ifft2(reference_mixed_fft, dim=(-2, -1)).real + original_img)

        return phase_swapped_ori, phase_swapped_ref
