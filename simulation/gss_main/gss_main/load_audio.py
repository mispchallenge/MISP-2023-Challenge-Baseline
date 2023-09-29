# import  audioread
import soundfile as sf


wav_path = '/disk4/hblan/graduate/gss_main/train_wave/far/wpe/gss_new/enhanced/R13_S215216217218_C04_I1_Far/S217-R13_S215216217218_C04_I1_Far-030327_030392.wav'
data, samplerate = sf.read('/disk4/hblan/graduate/gss_main/train_wave/far/wpe/gss_new/enhanced/R13_S215216217218_C04_I1_Far/S217-R13_S215216217218_C04_I1_Far-030327_030392.wav')
# with wave.open(wav_path, "rb") as f:
#     f = wave.open(wav_path)
#     print(f.getparams())
print(data,samplerate)