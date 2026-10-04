"""Build EchoTrace CSV using ResNet-50 ImageNet V2 and cosine nearest neighbours."""
import csv, json, time, hashlib
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

ROOT = Path(__file__).resolve().parent

def main():
    torch.set_num_threads(6)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    paths = sorted((ROOT/'data').glob('*.png'))
    names = [p.name for p in paths]
    assert len(names) == len(set(names)) and len(names) > 6
    weights_path = ROOT/'weights/resnet50-11ad3fa6.pth'
    assert hashlib.sha256(weights_path.read_bytes()).hexdigest().startswith('11ad3fa6')
    cache = ROOT/'features.npz'
    if cache.exists():
        saved = np.load(cache)
        assert list(saved['names']) == names
        features = torch.from_numpy(saved['features'])
    else:
        model = resnet50(weights=None)
        model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
        model.fc = nn.Identity()
        model.eval().to(device)
        transform = ResNet50_Weights.IMAGENET1K_V2.transforms()
        chunks = []
        started = time.time()
        with torch.inference_mode():
            for start in range(0, len(paths), 32):
                batch = []
                for p in paths[start:start+32]:
                    with Image.open(p) as im:
                        batch.append(transform(im.convert('RGB')))
                result = model(torch.stack(batch).to(device)).cpu()
                chunks.append(nn.functional.normalize(result, dim=1))
                if start % 320 == 0:
                    print(f'{min(start+32,len(paths))}/{len(paths)} images, {time.time()-started:.0f}s', flush=True)
        features = torch.cat(chunks)
        np.savez(cache, names=np.array(names), features=features.numpy())
    assert torch.isfinite(features).all()
    rankings = []
    with torch.inference_mode():
        for start in range(0,len(paths),256):
            sims = features[start:start+256] @ features.T
            rows = torch.arange(len(sims))
            sims[rows, start+rows] = -float('inf')
            rankings.extend(sims.topk(6,dim=1).indices.tolist())
    output = ROOT/'submission.csv'
    with output.open('w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename','ranking'])
        for i, neighbours in enumerate(rankings):
            assert len(set(neighbours)) == 6 and i not in neighbours
            writer.writerow([names[i], ' '.join(names[j] for j in neighbours)])
    with output.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows)==len(names) and {r['filename'] for r in rows}==set(names)
    for r in rows:
        rec = r['ranking'].split()
        assert len(rec)==len(set(rec))==6 and r['filename'] not in rec and set(rec)<=set(names)
    selected = [0,1,8,10,19,25,34,100]
    sheet = Image.new('RGB',(7*150,len(selected)*165),'white')
    draw = ImageDraw.Draw(sheet)
    for row,i in enumerate(selected):
        for col,j in enumerate([i]+rankings[i]):
            with Image.open(paths[j]) as im:
                im = im.convert('RGB'); im.thumbnail((144,140))
                sheet.paste(im,(col*150,row*165))
            draw.text((col*150,row*165+142),('QUERY ' if col==0 else '')+names[j],fill='black')
    sheet.save(ROOT/'neighbours.jpg')
    report = dict(images=len(names), recommendations_per_image=6,model='torchvision ResNet50 IMAGENET1K_V2',
                  features=2048,metric='cosine similarity',format_checks='passed',contest_score=None)
    (ROOT/'validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    main()
