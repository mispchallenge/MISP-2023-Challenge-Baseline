import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import tqdm
from lhotse.audio import AudioSource, Recording, RecordingSet, audioread_info
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import (
    Pathlike,
    add_durations,
    compute_num_samples,
    is_module_available,
)
import argparse
import os
from lhotse import CutSet


def get_duration(
    path: Pathlike,
) -> float:
    """
    Read a audio file, it supports pipeline style wave path and real waveform.

    :param path: Path to an audio file or a Kaldi-style pipe.
    :return: float duration of the recording, in seconds.
    """
    path = str(path)
    if path.strip().endswith("|"):
        if not is_module_available("kaldi_native_io"):
            raise ValueError(
                "To read Kaldi's data dir where wav.scp has 'pipe' inputs, "
                "please 'pip install kaldi_native_io' first."
            )
        import kaldi_native_io

        wave = kaldi_native_io.read_wave(path)
        # assert wave.data.shape[0] == 1, f"Expect 1 channel. Given {wave.data.shape[0]}"

        return wave.duration
    try:
        # Try to parse the file using pysoundfile first.
        import soundfile

        info = soundfile.info(path)
    except Exception:
        # import pdb;pdb.set_trace()
        # Try to parse the file using audioread as a fallback.
        info = audioread_info(path)
        
    return info.duration


def load_kaldi_data_dir(
    path: Pathlike,
    sampling_rate: int,
    map_string_to_underscores: Optional[str] = None,
    use_reco2dur: bool = True,
    num_jobs: int = 1,
    channel_num: int = 1,
    sids:list = None,
    savedir: str = None, 
    wav_type: str = None,
) -> Tuple[RecordingSet, Optional[SupervisionSet]]:
    """
    Load a Kaldi data directory and convert it to a Lhotse RecordingSet and
    SupervisionSet manifests. For this to work, at least the wav.scp file must exist.
    SupervisionSet is created only when a segments file exists. reco2dur is used by
    default when exists (to enforce reading the duration from the audio files
    themselves, please set use_reco2dur = False.
    All the other files (text, utt2spk, etc.) are optional, and some of them might
    not be handled yet. In particular, feats.scp files are ignored.

    :param map_string_to_underscores: optional string, when specified, we will replace
        all instances of this string in SupervisonSegment IDs to underscores.
        This is to help with handling underscores in Kaldi
        (see :func:`.export_to_kaldi`). This is also done for speaker IDs.
    """
    # import pdb;pdb.set_trace()
    path = Path(path)
    assert path.is_dir()

    def fix_id(t: str) -> str:
        if map_string_to_underscores is None:
            return t
        return t.replace(map_string_to_underscores, "_")
    # import pdb;pdb.set_trace()
    # must exist for RecordingSet
    recordings = load_kaldi_text_mapping(path / "wav.scp", must_exist=True)
    reco2dur = path / "reco2dur"
    if use_reco2dur and reco2dur.is_file():
        durations = load_kaldi_text_mapping(reco2dur, float_vals=True)
        assert len(durations) == len(recordings), (
            "The duration file reco2dur does not "
            "have the same length as the  wav.scp file"
        )
    else:
        with ProcessPoolExecutor(num_jobs) as ex:
            if  channel_num!=1:
                dur_vals = []
                for value in  recordings.values():
                    tmptimes = []
                    for i in range(channel_num):
                        # import pdb;pdb.set_trace()
                        tmptimes.append(get_duration(value+f"_{i}.wav"))
                    dur_vals.append(min(tmptimes))                
            else:
                dur_vals = []
                for value in  recordings.values():
                    tmptimes = []
                    for i in range(channel_num):
                        tmptimes.append(get_duration(value+f".wav"))
                    dur_vals.append(min(tmptimes))  
                # dur_vals = ex.map(get_duration, recordings.values()+f".wav") # ex.map(get_duration, recordings.values())
                # dur_vals = ex.map(lambda x: get_duration(x + ".wav"), [value for value in recordings.values()])
        # import pdb;pdb.set_trace()
        durations = dict(zip(recordings.keys(), dur_vals))

    sesssion_recordings = []

    for recording_id, path_or_cmd in tqdm.tqdm(recordings.items()):
        if savedir:
            path_or_cmd = os.path.join(savedir,path_or_cmd.split("/")[-1])
        if sids:
            if recording_id not in sids:
                continue   
        # import pdb;pdb.set_trace()
        if wav_type == "merge":
            sesssion_recordings.append(Recording.from_file(path_or_cmd+".wav", recording_id=recording_id))
        else:
                sesssion_recordings.append(
                    Recording(
                        id=recording_id,
                        sources= [
                            AudioSource(
                                type="command" if path_or_cmd.endswith("|") else "file",
                                channels=[0],
                                source=path_or_cmd[:-1] # 
                                if path_or_cmd.endswith("|")
                                else path_or_cmd+".wav",# 
                            )
                        ] if channel_num==1 else
                        [
                            AudioSource(
                                type="file",
                                channels=[channel_id],
                                source=path_or_cmd+f"_{str(channel_id)}.wav"
                            ) for channel_id in range(channel_num)
                        ],
                        sampling_rate=sampling_rate,
                        num_samples=compute_num_samples(durations[recording_id], sampling_rate),
                        duration=durations[recording_id],
                    )
                )
    recording_set = RecordingSet.from_recordings(sesssion_recordings)
                    
    
    supervision_set = None
    segments = path / "segments"
    
    with segments.open() as f:
        supervision_segments = [sup_string.strip().split() for sup_string in f]

    texts = load_kaldi_text_mapping(path / "text")
    speakers = load_kaldi_text_mapping(path / "utt2spk")
    genders = load_kaldi_text_mapping(path / "spk2gender")
    languages = load_kaldi_text_mapping(path / "utt2lang")
    sups = []
    for segment_id, recording_id, start, end in tqdm.tqdm(supervision_segments):
        if sids:
            if recording_id not in sids: 
                continue
        sups.append(
            SupervisionSegment(
                id=fix_id(segment_id),
                recording_id=recording_id,
                start=float(start),
                duration=add_durations(
                    float(end), -float(start), sampling_rate=sampling_rate
                ),
                channel=0 if channel_num==1 else list(range(channel_num)),
                text=texts[segment_id],
                language=languages[segment_id],
                speaker=fix_id(speakers[segment_id]),
                gender=genders[speakers[segment_id]],
            )
        )
        # import pdb;pdb.set_trace() 
        supervision_set = SupervisionSet.from_segments(sups)

    return recording_set, supervision_set


def load_kaldi_text_mapping(
    path: Path, must_exist: bool = False, float_vals: bool = False
) -> Dict[str, Optional[str]]:
    """Load Kaldi files such as utt2spk, spk2gender, text, etc. as a dict."""
    mapping = defaultdict(lambda: None)
    if path.is_file():
        with path.open() as f:
            mapping = dict(line.strip().split(maxsplit=1) for line in f)
        if float_vals:
            mapping = {key: float(val) for key, val in mapping.items()}
    elif must_exist:
        raise ValueError(f"No such file: {path}")
    return mapping



def import_(
    data_dir: Pathlike,
    sampling_rate: int,
    manifest_dir: Pathlike,
    map_string_to_underscores: Optional[str],
    num_jobs: int,
    compute_durations: bool,
    channel_num: int = 1,
    sids: list = None,
    savedir:str = None,
    wav_type:str = None,
):
    """
    Convert a Kaldi data dir DATA_DIR into a directory MANIFEST_DIR of lhotse manifests. Ignores feats.scp.
    The SAMPLING_RATE has to be explicitly specified as it is not available to read from DATA_DIR.
    """

    recording_set, maybe_supervision_set = load_kaldi_data_dir(
        path=data_dir,
        sampling_rate=sampling_rate,
        map_string_to_underscores=map_string_to_underscores,
        num_jobs=num_jobs,
        use_reco2dur=not compute_durations,
        channel_num=channel_num,
        sids = sids,
        savedir=savedir,
        wav_type = wav_type,
    )
    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    recording_set.to_file(manifest_dir / "recordings.jsonl.gz")
    if maybe_supervision_set is not None:
        maybe_supervision_set.to_file(manifest_dir / "supervisions.jsonl.gz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("lhoste")
    parser.add_argument("--data_dir", type=Path)
    parser.add_argument("--sampling_rate", type=int)
    parser.add_argument("--manifest_dir", type=Path)
    parser.add_argument(
        "-u",
        "--map-string-to-underscores",
        default=None,
        type=str,
        help="""When specified, we will replace all instances of this string
        in SupervisonSegment IDs to underscores. This is to help with handling
        underscores in Kaldi (see 'export_to_kaldi').""",
    )
    parser.add_argument(
        "-j",
        "--num-jobs",
        default=1,
        type=int,
        help="Number of jobs for computing recording durations.",
    )
    parser.add_argument(
        "-d",
        "--compute-durations",
        default=True,
        type=bool,
        help="Compute durations by reading the whole file instead of using reco2dur file",
    )
    parser.add_argument(
        "-c",
        "--channels",
        default=1,
        type=int,
        help="in wav.scp, wav_path.wav to wav_path_channelid.wav",
    )
    parser.add_argument(
        "-s",
        "--sids",
        default=None,
        nargs = "+",
        type=str,
        help="",
    )
    parser.add_argument(
        "--savedir",
        default=None,
        type=str,
        help="",
    )
    parser.add_argument(
        "--wav_type",
        default=None,
        type=str,
        help="",
    )
    parser.add_argument(
        "--mode",
        default=None,
        type=str,
        help="",
    )
    parser.add_argument(
        "--cutpath",
        default=None,
        type=str,
        help="",
    )

    args = parser.parse_args()
    if args.mode == "filter_cut":
        cuts = CutSet.from_file(args.cutpath)
        clean_cuts = cuts.filter(lambda c: len(c.supervisions)!=0).to_eager()
        clean_cuts.to_file(args.cutpath)
    else:
        import_(args.data_dir,args.sampling_rate,args.manifest_dir,args.map_string_to_underscores,\
            args.num_jobs,args.compute_durations,args.channels,args.sids,args.savedir,args.wav_type)

    
