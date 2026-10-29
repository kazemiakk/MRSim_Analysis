from pathlib import Path
import pytest
from PIL import Image
from scripts.collect_project_images import copy_name,copy_verified,image_metadata,excluded,collect,sort_key


def test_flat_collision_safe_names(tmp_path):
    a=tmp_path/'a'/'same.png';b=tmp_path/'b'/'same.png'
    x=copy_name(1,a,tmp_path);y=copy_name(2,b,tmp_path)
    assert x!=y and '/' not in x and '\\' not in x
    assert len(copy_name(1,tmp_path/('x'*200+'.png'),tmp_path))<140


def test_copy_preserves_animation_bytes_and_metadata(tmp_path):
    a=tmp_path/'source.gif';b=tmp_path/'copy.gif'
    frames=[Image.new('RGB',(8,8),c) for c in ['red','blue']]
    frames[0].save(a,save_all=True,append_images=frames[1:],duration=[80,120],loop=0)
    copy_verified(a,b)
    assert a.read_bytes()==b.read_bytes()
    meta=image_metadata(b);assert meta['frames']==2 and meta['animated'] is True
    assert not meta['metadata_error']


def test_copy_does_not_overwrite(tmp_path):
    a=tmp_path/'a.png';b=tmp_path/'b.png';a.write_bytes(b'a');b.write_bytes(b'b')
    with pytest.raises(FileExistsError):copy_verified(a,b)
    assert b.read_bytes()==b'b'


def test_metadata_failure_does_not_discard_file(tmp_path):
    a=tmp_path/'bad.png';a.write_bytes(b'not an image')
    meta=image_metadata(a);assert meta['metadata_error'] and a.read_bytes()==b'not an image'


def test_scope_skips_private_cache_icons_not_real_figures(tmp_path):
    assert excluded(tmp_path/'reference_inputs'/'scanner.png',tmp_path)
    assert excluded(tmp_path/'outputs'/'profile-a'/'cache.png',tmp_path)
    assert excluded(tmp_path/'report_app'/'src'/'icons'/'a.svg',tmp_path)
    assert not excluded(tmp_path/'outputs'/'historical'/'figure.png',tmp_path)


def test_existing_target_preserved(tmp_path):
    with pytest.raises(FileExistsError):collect(tmp_path,tmp_path)


def test_recent_sets_first(tmp_path):
    a=tmp_path/'outputs/e043_aorta_branch_views_v3/a.png';b=tmp_path/'outputs/aorta_old/b.png'
    assert sort_key(a,tmp_path)<sort_key(b,tmp_path)
