import torch

from neural_waveshaping_synthesis.models.modules.dynamic import DynamicFFTConv1d, DynamicSincConv1d


# def test_dynamic_fft():
#     dynamic_fft = DynamicFFTConv1d(
#         in_channels=2, out_channels=5, kernel_size=4, hop_length=1, conditioning_size=4
#     )

#     fake_input = torch.ones(1, 2, 12)
#     fake_conditioning = torch.ones(1, 4, 12)
#     print(dynamic_fft(fake_input, fake_conditioning))


def test_dynamic_sinc():
    dynamic_sinc = DynamicSincConv1d(
        in_channels=2,
        out_channels=5,
        kernel_size=128,
        hop_length=128 // 2,
        n_sincs=3,
        conditioning_size=4,
    )
    fake_input = torch.ones(1, 2, 64000, requires_grad=True)
    fake_conditioning = torch.ones(1, 4, 64000)
    output = dynamic_sinc(fake_input, fake_conditioning)
    assert output.shape[-1] == fake_input.shape[-1]