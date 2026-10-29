"""E-044: unchanged, flat local review copies of MRSim image files."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import uuid
import xml.etree.ElementTree as ET
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=ROOT.parents[1]
EXTENSIONS={'.png','.jpg','.jpeg','.gif','.tif','.tiff','.bmp','.webp','.svg','.avif','.eps'}
EXCLUDED_PARTS={'.git','.venv','.venv-local','.uv-cache','node_modules','__pycache__','.pytest_cache','.mplconfig','codex_history','reference_inputs','external_inputs','icons'}


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def excluded(path,base):
    rel=path.relative_to(base)
    parts=[p.lower() for p in rel.parts]
    return any(p in EXCLUDED_PARTS or p.startswith('profile-') for p in parts) or path.name.lower() in {'favicon.png','favicon.svg','favicon.ico'}


def discover(workspace=WORKSPACE):
    roots=[workspace/'MRSIM/MRSimTracks',workspace/'MRSIM/MRSim_Analysis',workspace/'Charles_email_figures_2026-08-31']
    found=set();excluded_files=[]
    for root in roots:
        if not root.is_dir():continue
        command=['rg','--files','--hidden','--no-ignore']
        for ext in sorted(EXTENSIONS):command.extend(['--iglob','*'+ext])
        for part in sorted(EXCLUDED_PARTS):command.extend(['--glob','!**/'+part+'/**'])
        command.extend(['--glob','!**/profile-*/**',str(root)])
        result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8')
        if result.returncode not in (0,1):raise RuntimeError(result.stderr)
        for line in result.stdout.splitlines():
            p=Path(line).resolve()
            if not p.is_relative_to(root.resolve()):raise ValueError('Source escaped scope')
            if excluded(p,workspace):excluded_files.append(str(p.relative_to(workspace)));continue
            found.add(p)
    # Only explicitly MRSim-named loose images, not unrelated CFD projects.
    for p in workspace.iterdir():
        if p.is_file() and p.suffix.lower() in EXTENSIONS and any(s in p.stem.lower() for s in ['mrsim','aorta','phantom','koma','tracks']):found.add(p.resolve())
    return sorted(found,key=lambda p:sort_key(p,workspace)),dict(roots=[str(p) for p in roots],excluded_named_files=sorted(excluded_files),excluded_directory_names=sorted(EXCLUDED_PARTS),excluded_patterns=['profile-*'],loose_workspace_file_rule='MRSim/aorta/phantom/Koma/tracks name only',archives_extracted=False,embedded_html_images_extracted=False,pdfs_videos_included=False)


def sort_key(path,workspace):
    rel=path.relative_to(workspace).as_posix().lower()
    priorities=['e043_aorta_branch_views_v3/','e042_publication_slices_v4/','e040_concise_png_v3/']
    rank=next((i for i,s in enumerate(priorities) if s in rel),3)
    return rank,rel


def copy_name(index,path,workspace):
    rel=path.relative_to(workspace).as_posix()
    token=hashlib.sha256(rel.encode('utf-8')).hexdigest()[:10]
    stem=''.join(c if c not in '<>:"/\\|?*' else '_' for c in path.stem)[:100].rstrip(' .')
    return f'{index:06d}__{stem}__{token}{path.suffix.lower()}'


def image_metadata(path):
    try:
        if path.suffix.lower()=='.svg':
            root=ET.parse(path).getroot()
            return dict(format='SVG',width=root.get('width'),height=root.get('height'),frames=1,animated=False,metadata_status='XML_PARSED',metadata_error='')
        with Image.open(path) as im:
            width,height=im.size;frames=getattr(im,'n_frames',1);fmt=im.format
        # Reopen for the format's structural check; not an exhaustive visual review.
        with Image.open(path) as im:im.verify()
        return dict(format=fmt,width=width,height=height,frames=frames,animated=frames>1,metadata_status='HEADER_FRAME_STRUCTURE_READ',metadata_error='')
    except Exception as error:
        return dict(format=path.suffix[1:].upper(),width=None,height=None,frames=None,animated=None,metadata_status='METADATA_UNAVAILABLE_COPY_PRESERVED',metadata_error=str(error))


def context_label(rel):
    text=rel.lower()
    if 'charles_email_figures' in text:return 'Earlier email-selection copy'
    if 'mrsim_analysis/' in text:return 'Legacy analysis'
    if '/.partial-' in text:return 'Partial-run image; not a completed result'
    if 'visual_review' in text or '/screenshots/' in text or 'browser' in text:return 'Review/screenshot derivative'
    if 'archive' in text or '/historical/' in text:return 'Historical/archive image'
    if '/ignore/' in text:return 'Historical rendering test'
    if '/docs/' in text or '/site/' in text:return 'Documentation/example image'
    return 'Project output; consult original run context'


def copy_verified(source,dest):
    before=source.stat()
    if dest.exists():raise FileExistsError(dest)
    shutil.copy2(source,dest)
    source_hash=digest(source);copy_hash=digest(dest);after=source.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('Source changed during copy: '+str(source))
    if source_hash!=copy_hash or before.st_size!=dest.stat().st_size:raise RuntimeError('Byte-copy verification failed: '+str(source))
    return source_hash,before


def collect(target,workspace=WORKSPACE):
    workspace=workspace.resolve();target=target.resolve()
    if target.exists():raise FileExistsError(target)
    if not target.is_relative_to(workspace) or target==workspace:raise ValueError('Review target must be a new subfolder in the MRSim workspace')
    sources,scope=discover(workspace)
    if not sources:raise ValueError('No scoped images found')
    total=sum(p.stat().st_size for p in sources)
    free=shutil.disk_usage(workspace).free
    if free<total+512*1024**2:raise OSError('Insufficient space with512MiB reserve')
    # Normal inherited Windows ACLs: tempfile.mkdtemp uses a private DACL that
    # can prevent the workspace owner's normal account from reading the folder.
    out=target.with_name(target.name+'.partial-'+uuid.uuid4().hex)
    out.mkdir(mode=0o777,exist_ok=False);rows=[];first_hash={}
    for i,source in enumerate(sources,1):
        name=copy_name(i,source,workspace);dst=out/name;sha,stat=copy_verified(source,dst)
        meta=image_metadata(dst);relative=source.relative_to(workspace).as_posix()
        first=first_hash.setdefault(sha,i)
        rows.append(dict(id=i,copy=name,source_relative=relative,source_absolute=str(source),bytes=stat.st_size,source_modified_utc=datetime.fromtimestamp(stat.st_mtime,timezone.utc).isoformat(),sha256=sha,duplicate_of=first if first!=i else None,context=context_label(relative),**meta))
        if i%100==0:print(f'Copied and hash-checked {i}/{len(sources)}',flush=True)
    summary=dict(experiment='E-044',created_utc=datetime.now(timezone.utc).isoformat(),destination=str(target),count=len(rows),bytes=total,unique_content=len(first_hash),exact_duplicate_copies=len(rows)-len(first_hash),animated_images=sum(r['animated'] is True for r in rows),gif_files=sum(Path(r['copy']).suffix=='.gif' for r in rows),by_extension=dict(Counter(Path(r['copy']).suffix for r in rows)),metadata_issues=[r['id'] for r in rows if r['metadata_error']],copy_hashes_all_match=True,originals_modified=False,scope=scope,visual_review='Inventory/metadata and byte integrity only; not a scientific or visual judgment of every image')
    manifest=dict(summary=summary,images=rows)
    (out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    with (out/'IMAGE_INDEX.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    template=(ROOT/'scripts/project_image_viewer.html').read_text(encoding='utf-8')
    # Self-contained metadata, no fetch: works directly from a local file URL.
    raw=json.dumps(manifest,ensure_ascii=False).replace('<','\\u003c')
    (out/'OPEN_IMAGES.html').write_text(template.replace('__IMAGE_MANIFEST__',raw),encoding='utf-8')
    note=f'''# MRSim: all saved research images for local review

{len(rows)} byte-identical image copies in ONE flat folder; {len(first_hash)} unique
contents, {len(rows)-len(first_hash)} additional exact duplicate copies. Nothing was
deduplicated, resaved, converted, resized or moved. {summary['gif_files']} GIFs
retain their original bytes; {summary['animated_images']} images are multi-frame.

Open OPEN_IMAGES.html to view one image at a time. Use Previous/Next or left/right
arrow keys. Search original paths, filter by format/context, jump to an image ID,
or optionally hide exact duplicates. Default includes every copy. OPEN_IMAGES.html
needs no internet or local server. Numbered image copies can also be opened in
Windows Photos/File Explorer. IMAGE_INDEX.csv and MANIFEST.json map every copy
to its original location and record SHA256, size, dimensions and frame count.

Scope: main MRSimTracks project, legacy MRSim_Analysis, earlier Charles-email
figure folder and explicitly MRSim-named loose workspace images. Generated
outputs, historical revisions, animation frames, GIF/WebP animations, review
screenshots and documentation/example images are retained. Source paths and
context labels are NOT a claim that every image is current, correct or suitable
for publication. Latest E-043/E-042/E-040 figure sets are ordered first.

Excluded: unrelated surrounding CFD projects, private scanner/reference inputs,
external source-input packages, chat-history/system snapshots, dependency caches,
browser profiles, app UI icons and favicons. No archive was extracted, no embedded
HTML image decoded, no PDF/report or video converted. All eligible saved loose
image files in the recorded scope are copied, including exact duplicates.

This is a LOCAL review collection, not a publicly de-identified release or email
attachment package. The metadata includes local source paths. Image metadata and
copy hashes were checked; every image has NOT been scientifically or visually
assessed. Metadata-unreadable IDs: {summary['metadata_issues']}.

Total original-image bytes copied: {total:,} ({total/1024**2:.1f} MiB).
Original scientific data and figures remain untouched.
'''
    (out/'README.md').write_text(note,encoding='utf-8')
    provenance=dict(script_sha256=digest(Path(__file__)),viewer_template_sha256=digest(ROOT/'scripts/project_image_viewer.html'),files={name:dict(bytes=(out/name).stat().st_size,sha256=digest(out/name)) for name in ['MANIFEST.json','IMAGE_INDEX.csv','OPEN_IMAGES.html','README.md']})
    (out/'COLLECTION_PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    if target.exists():raise FileExistsError(target)
    out.rename(target);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);collect(p.parse_args().output)
