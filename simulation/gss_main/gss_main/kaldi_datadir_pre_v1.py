from espnet2.fileio.read_text import read_2column_text
from pathlib import Path
from espnet2.fileio.datadir_writer import DatadirWriter
import argparse
import json
import codecs
import os 
def json2dic(jsonpath, dic=None, format=False):
    """
    read dic from json or write dic to json
    :param jsonpath: filepath of json
    :param dic: content dic or None, None means read
    :param format: if True, write formated json, it needs more time
    :return: content dic for read while None for write
    """
    if dic is None:
        with codecs.open(jsonpath, 'r') as handle:
            output = json.load(handle)
        return output
    else:
        assert isinstance(dic, dict)
        with codecs.open(jsonpath, 'w') as handle:
            if format:
                json.dump(dic, handle, indent=4, ensure_ascii=False)
            else:
                json.dump(dic, handle)
        return None

# input wangze json, create datadir in kaldi format: utt2uiq.utt2spk,pathfile_names
def json2datadir(index_file,pathfile_names,items,tgtdir):
    index = json2dic(os.path.join(index_file,"eval_with_gss.json"))
    with DatadirWriter(tgtdir) as writer:
        utt2uniq = writer["utt2uniq"]
        utt2spk = writer["utt2spk"]
        utt2paths = [ writer[pathfile_names[i]] for i in range(len(pathfile_names)) ]
        for key in index["keys"]:
            for i in range(len(utt2paths)):
                utt2paths[i][key] =  index["key2path"][key][items[i]].replace("/raw7/cv1/hangchen2/misp2022/Released","/yrfs1/intern/zhewang18/zhewang/misp2021_avsr/data/misp2022/Released").replace("far_gss_segment","far_gss_oracle_segment")
            utt2uniq[key] = key
            utt2spk[key] = key.split("_")[0]

# input ptdir, create datadir in kaldi format:utt2uiq.utt2spk,filename
def ptdir2datadir(srcptdir,filename,tgtdir,filter_file=None):
    if filter_file:
        fileter_keys = list(read_2column_text(filter_file).keys())
    else:
        fileter_keys = []
    # filter_file
    tgtdir = Path(tgtdir)
    if not tgtdir.exists():
        tgtdir.mkdir(parents=True, exist_ok=True) 
    files = Path(srcptdir).glob("*.pt")
    with DatadirWriter(tgtdir) as writer:
        utt2path = writer[filename]
        utt2uniq = writer["utt2uniq"]
        utt2spk = writer["utt2spk"]
        for file in files:
            key = file.stem
            if key not in fileter_keys:
                utt2path[key] = str(file)
                utt2uniq[key] = key
                utt2spk[key] = key.split("_")[0]

def gen_roiscp(pt_dir,roiscpdir,filename,splitnum=None,suffix=".pt"):
    if splitnum:
        if len(splitnum.split("-"))==1:
            use_splitnums = [int(splitnum)]
        else:
            splitmin,splitmax = splitnum.split("-")
            use_splitnums = list(range(int(splitmin),int(splitmax+1)))
        print(f"selecting splitnum: {use_splitnums}")
    
    print(pt_dir)
    print(os.path.join(roiscpdir,filename))
    files = pt_dir.rglob(f"*{suffix}")
    with DatadirWriter(roiscpdir) as writer:
        subwriter = writer[filename]
        for file in files:
            if splitnum:
                if len(file.stem.split("@"))-1 in use_splitnums:
                    subwriter[file.stem] = str(file)
            else:
                subwriter[file.stem] = str(file)

def json2scp(srcpath,shapename,tgtdir):
    wavfile = Path(tgtdir) / "wav.scp"
    assert wavfile.exists(),f"there is no {wavfile}"
    wav2path = read_2column_text(wavfile)
    tgtpath = Path(tgtdir) / shapename
    with open(srcpath,"r") as f:
        dic = json.load(f)
    tgtdir = tgtpath.parent
    filename = tgtpath.name
    with DatadirWriter(tgtdir) as writer:
        subwriter = writer[filename]
        for key in wav2path.keys():
            subwriter[key] = str(int(dic[key][0]))

def speech2reco2dur(shape_file):
    with DatadirWriter(str(shape_file.parent)) as writer:
        subwriter = writer["reco2dur"]
        for key,value in read_2column_text(shape_file).items():
            subwriter[key] = str("{:.2f}".format(float(int(value)/16000)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser("run_wpe")
    parser.add_argument( "--pt_dir",type=str,default="/raw7/cv1/hangchen2/misp2021_avsr/feature/misp2021_avsr/eval_far_video_lip_segment/pt",
    help="the path where roi.pt files stored",)
    parser.add_argument( "--tgtdir", type=str, default="/yrfs2/cv1/hangchen2/espnet/misp2021/asr1/dump/raw/org/eval_mid_lip", help="the path where roi.scp storened")
    parser.add_argument( "--filename", type=str, default="roi.scp", help="the path where roi.scp storened")
    
    parser.add_argument( "--mode", type=str, default=None, help="ptdir2scp,shapejson2shapescp,ptdir2datadir,json2datadir")
    
    # ptdir2scp
    parser.add_argument( "--suffix", type=str, default=".pt", help="file suffix")
    # ptdir2datadir
    parser.add_argument( "--splitnum", type=str, default=None, help="2-3 mean use spliting 2-3")
    #json2datadir
    parser.add_argument( "--pathfile_names", nargs="+",type=str, default=None, help="filenames of outputfiles")
    parser.add_argument( "--items", nargs="+",type=str, default=None, help="items to get in json")
    #filter_file 
    parser.add_argument( "--filter_file",type=str, default=None, help="use the uids in this file to cut the files")

    #speech2reco2dur
    parser.add_argument( "--shape_file",type=Path, default=None, help="speech_shape2reco2dur")

    args = parser.parse_args()
    
    if args.mode == "ptdir2scp":
        gen_roiscp(Path(args.pt_dir),args.tgtdir,args.filename,args.splitnum,args.suffix)

    elif args.mode == "shapejson2shapescp":
        json2scp(Path(args.pt_dir).parent / "key2shape.json",args.filename,Path(args.tgtdir))
    
    elif args.mode == "ptdir2datadir":
        ptdir2datadir(args.pt_dir,args.filename,args.tgtdir,args.filter_file)
    
    elif args.mode == "json2datadir":
        json2datadir(args.pt_dir,args.pathfile_names,args.items,args.tgtdir)
        # index_file,pathfile_names,items,tgtdir
    elif args.mode == "speech2reco2dur":
        speech2reco2dur(args.shape_file)