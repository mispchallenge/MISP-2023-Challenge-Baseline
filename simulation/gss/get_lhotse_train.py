import glob
import os.path
from collections import defaultdict
from pathlib import Path
import soundfile as sf
from lhotse import fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet

import sys
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description='Prepare clean information json.', 
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--misp_se_root_path', metavar='<dir>', required=True,
                    help='simulation_data directory name.')
parser.add_argument('--input_rttm', metavar='<file>', required=True,
                    help='input rttm.')
parser.add_argument('--outdir', metavar='<file>', required=True,
                    help='Output directory name.')
args = parser.parse_args()

ignore_shorter = 0.5

misp_se_root_path = args.misp_se_root_path
input_rttm = args.input_rttm
outdir=args.outdir

os.system(f'mkdir -p {outdir}')

manifests = defaultdict(dict)
recordings = []
supervisions = []

sess_list = []

with open(input_rttm) as f:
    for line in f.readlines():
        if line.split(' ')[1] not in sess_list:
            sess_list.append(line.split(' ')[1])

for session in tqdm(sess_list):
    ids,noisy=session.split("_")
    audio_paths = [
        Path(x) for x in glob.glob(
            os.path.join(misp_se_root_path+f"/{noisy}", f'{ids}_ch*_{noisy}.wav')
        )
    ]
    sources = []
    for idx, audio_path in enumerate(sorted(audio_paths)):
        sources.append(
            AudioSource(type='file', channels=[idx], source=str(audio_path))
        )
    audio_sf = sf.SoundFile(str(audio_paths[0]))
    recordings.append(
        Recording(
            id=f'{session}',
            sources=sources,
            sampling_rate=int(audio_sf.samplerate),
            num_samples=audio_sf.frames,
            duration=audio_sf.frames/audio_sf.samplerate,
        )
    )

recordings = RecordingSet.from_recordings(recordings)

with open(input_rttm) as f:
    for idx,line in enumerate(tqdm(f.readlines())):
        line=line.strip()
        session=line.split(' ')[1]
        spk_id=line.split(' ')[7]
        channel=[0,1,2,3,4,5]
        start=float(line.split(' ')[3])
        dur=float(line.split(' ')[4])
        if ignore_shorter is not None and dur < ignore_shorter:
            print(f"ignore session:{session} spkid:{spk_id} start:{start}")
            continue
        if dur<=0:
            continue
        supervisions.append(
            SupervisionSegment(
                id=f"{session}-{idx}",
                recording_id=session,
                start=start,
                duration=dur,
                channel=channel,
                text=None,
                speaker=spk_id
            )
        )

supervisions=SupervisionSet.from_segments(supervisions)

recording_set,supervision_set=fix_manifests(
    recordings=recordings,supervisions=supervisions
)

validate_recordings_and_supervisions(recording_set,supervision_set)
supervision_set.to_file(
    os.path.join(outdir,"supervisions.jsonl.gz")
)
recording_set.to_file(
    os.path.join(outdir,"recordings.jsonl.gz")
)






