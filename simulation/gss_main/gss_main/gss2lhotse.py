import argparse
import glob
import os
import warnings
from copy import deepcopy
from pathlib import Path
import tqdm
import lhotse
import soundfile as sf


def get_new_manifests(input_dir, output_filename,mode="near"):
    assert os.path.exists(os.path.join(input_dir, "enhanced")), (
        f"{input_dir} does not contain a sub-folder "
        f"named enhanced, is it the correct path ?"
    )

    wavs = glob.glob(os.path.join(input_dir, "enhanced/**/*.flac"), recursive=True)
    if wavs == []:
        wavs = glob.glob(os.path.join(input_dir, "enhanced/**/*.wav"), recursive=True)
    if wavs == []:
        raise ValueError("Pay attention to the suffix to the gss audio output")
    segcuts = lhotse.CutSet.from_jsonl(
        os.path.join(input_dir, "cuts_per_segment.jsonl.gz")
    )
    id2wav = {Path(w).stem: w for w in wavs}
    recordings = []
    supervisions = []
    # import pdb;pdb.set_trace()
    if mode == "mid":
        for c_k in tqdm.tqdm(segcuts.data.keys()):
            c_cut = segcuts.data[c_k]
            recording_id_org = "_".join(c_cut.id.split("_")[1:-1])
            recording_id = recording_id_org + "_Middle"
            speaker = c_cut.supervisions[0].speaker
            gss_id = (
                f"{recording_id}-{speaker}-"
                f"{int(100*c_cut.start):06d}_{int(100*c_cut.end):06d}"
            )
            utt_id = (
                f"Middle_{speaker}_{recording_id_org}-"
                f"{int(100*c_cut.start):06d}_{int(100*c_cut.end):06d}"
            )
            try:
                enhanced_audio = id2wav[gss_id]
            except KeyError:
                warnings.warn(
                    "Skipped example {}, of length {:.2f}, it was not found in GSS input. "
                    "This may lead to significant errors for ASR if this is inference."
                    "It could be that it was discarded by GSS "
                    "because the segment was too long,"
                    "check GSS max-segment-length argument.".format( gss_id, c_cut.end - c_cut.start
                    )
                )
                continue

            # recording id is unique for this example we add a gss postfix
            sources = [lhotse.AudioSource(type="file", channels=[0], source=enhanced_audio)]
            audio_sf = sf.SoundFile(enhanced_audio)
            duration = audio_sf.frames / audio_sf.samplerate

            if not abs(duration - (c_cut.end - c_cut.start)) < 0.1:
                continue
            recordings.append(
                lhotse.Recording(
                    id=utt_id,
                    sources=sources,
                    sampling_rate=audio_sf.samplerate,
                    num_samples=audio_sf.frames,
                    duration=duration,
                )
            )

            new_sup = deepcopy(c_cut.supervisions[0])
            new_sup.id = utt_id
            new_sup.recording_id = utt_id
            new_sup.start = 0
            new_sup.duration = duration
            new_sup.channel = [0]
    elif mode =="Far":# 
        for c_k in tqdm.tqdm(segcuts.data.keys()):
            # import pdb;pdb.set_trace()
            c_cut = segcuts.data[c_k]
            recording_id_org = "_".join(c_cut.id.split("_")[1:-1])
            recording_id = recording_id_org + "_Far" # 修改了+ "_Far"
            speaker = c_cut.supervisions[0].speaker
            gss_id = (
                f"{recording_id}-{speaker}-" # 修改了{speaker}{recording_id}
                f"{int(round(100*(c_cut.start))):06d}_{int(round(100*(c_cut.end))):06d}"
            )
            utt_id = (
                f"{speaker}_{recording_id_org}_"
                f"{int(round(100*(c_cut.start))):06d}_{int(round(100*(c_cut.end))):06d}"
            )
            import pdb;pdb.set_trace()
            try:
                enhanced_audio = id2wav[gss_id]
            except KeyError:
                warnings.warn(
                    "Skipped example {}, of length {:.2f}, it was not found in GSS input. "
                    "This may lead to significant errors for ASR if this is inference."
                    "It could be that it was discarded by GSS "
                    "because the segment was too long,"
                    "check GSS max-segment-length argument.".format(
                        gss_id, c_cut.end - c_cut.start
                    )
                )
                continue

            # recording id is unique for this example we add a gss postfix
            sources = [lhotse.AudioSource(type="file", channels=[0], source=enhanced_audio)]
            audio_sf = sf.SoundFile(enhanced_audio)
            duration = audio_sf.frames / audio_sf.samplerate

            if not abs(duration - (c_cut.end - c_cut.start)) < 0.1:
                continue
            recordings.append(
                lhotse.Recording(
                    id=utt_id,
                    sources=sources,
                    sampling_rate=audio_sf.samplerate,
                    num_samples=audio_sf.frames,
                    duration=duration,
                )
            )

            new_sup = deepcopy(c_cut.supervisions[0])
            new_sup.id = utt_id
            new_sup.recording_id = utt_id
            new_sup.start = 0
            new_sup.duration = duration
            new_sup.channel = [0]
            supervisions.append(new_sup)
    else:# 下面部分需要修改因为命名格式不同的问题
        for c_k in tqdm.tqdm(segcuts.data.keys()):
            # import pdb;pdb.set_trace()
            c_cut = segcuts.data[c_k]
            speaker = c_cut.supervisions[0].speaker
            # import pdb;pdb.set_trace()
            recording_id_org = "_".join(c_cut.id.split("_")[1:-1])
            recording_id = recording_id_org + "_Far" #  "_Far"远场—通用 line 140-150
            gss_id = (
                f"{speaker}-{recording_id}-" # 
                f"{int(round(100*(c_cut.start))):06d}_{int(round(100*(c_cut.end))):06d}"
            )
            utt_id = (
                f"{speaker}_{recording_id}_"
                f"{int(round(100*(c_cut.start))):06d}_{int(round(100*(c_cut.end))):06d}"
            )
            # gss_id = (
            #     f"{recording_id}-{speaker}-" 
            #     f"{int(100*c_cut.start):06d}_{int(100*c_cut.end):06d}"
            # )
            # utt_id = (
            #     f"{speaker}_{recording_id}_"
            #     f"{int(100*c_cut.start):06d}-{int(100*c_cut.end):06d}"
            # )
            # import pdb;pdb.set_trace()
            try:
                enhanced_audio = id2wav[gss_id]
            except KeyError:
                warnings.warn(
                    "Skipped example {}, of length {:.2f}, it was not found in GSS input. "
                    "This may lead to significant errors for ASR if this is inference."
                    "It could be that it was discarded by GSS "
                    "because the segment was too long,"
                    "check GSS max-segment-length argument.".format(
                        gss_id, c_cut.end - c_cut.start
                    )
                )
                continue

            # recording id is unique for this example we add a gss postfix
            sources = [lhotse.AudioSource(type="file", channels=[0], source=enhanced_audio)]
            audio_sf = sf.SoundFile(enhanced_audio)
            duration = audio_sf.frames / audio_sf.samplerate

            if not abs(duration - (c_cut.end - c_cut.start)) < 0.1:
                continue
            recordings.append(
                lhotse.Recording(
                    id=utt_id,
                    sources=sources,
                    sampling_rate=audio_sf.samplerate,
                    num_samples=audio_sf.frames,
                    duration=duration,
                )
            )

            new_sup = deepcopy(c_cut.supervisions[0])
            new_sup.id = utt_id
            new_sup.recording_id = utt_id
            new_sup.start = 0
            new_sup.duration = duration
            new_sup.channel = [0]
            supervisions.append(new_sup)

    # import pdb;pdb.set_trace()
    recording_set, supervision_set = lhotse.fix_manifests(
        lhotse.RecordingSet.from_recordings(recordings),
        lhotse.SupervisionSet.from_segments(supervisions),
    )
    # Fix manifests
    lhotse.validate_recordings_and_supervisions(recording_set, supervision_set)

    Path(output_filename).parent.mkdir(exist_ok=True, parents=True)
    filename = Path(output_filename).stem
    supervision_set.to_file(
        os.path.join(Path(output_filename).parent, f"{filename}_supervisions.jsonl.gz")
    )
    recording_set.to_file(
        os.path.join(Path(output_filename).parent, f"{filename}_recordings.jsonl.gz")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Lhotse manifests generation scripts for CHiME-7 Task 1 data.",
        add_help=True,
        usage="%(prog)s [options]",
    )
    parser.add_argument(
        "-i,--in_gss_dir",
        type=str,
        metavar="STR",
        dest="gss_dir",
        help="Path to the GPU-GSS dir "
        "containing ./enhanced/*.flac enhanced files and "
        "a cuts_per_segment.json.gz manifest.",
    )
    parser.add_argument(
        "-o,--output_name",
        type=str,
        metavar="STR",
        dest="output_name",
        help="Path and filename for the new manifest, "
        "e.g. /tmp/chime6_gss will create in /tmp "
        "/tmp/chime6_gss-recordings.jsonl.gz "
        "and /tmp/chime6_gss-supervisions.jsonl.gz.",
    )

    args = parser.parse_args()
    get_new_manifests(args.gss_dir, args.output_name)

                       
