"""Notice repacking must preserve payload bytes and refuse malformed archives."""
import importlib.util
from pathlib import Path
import zipfile
import pytest

path=Path(__file__).resolve().parents[1]/'tools/package_exact_provider_notices.py'
spec=importlib.util.spec_from_file_location('exact_notice_packager',path)
package=importlib.util.module_from_spec(spec);spec.loader.exec_module(package)


def original(path):
    with zipfile.ZipFile(path,'x') as wheel:
        wheel.writestr('provider/native.pyd',b'\x00native\xff')
        wheel.writestr('provider/__init__.py',b'unchanged bootstrap\n')
        wheel.writestr('provider.libs/exact-name.dll',b'unchanged DLL')
        wheel.writestr('provider-1.dist-info/METADATA',b'Name: provider\nVersion: 1\n\ndescription\n')
        wheel.writestr('provider-1.dist-info/WHEEL',b'Tag: cp311-cp311-win_amd64\n')
        wheel.writestr('provider-1.dist-info/RECORD',b'old record')


def test_additive_repack_preserves_runtime_and_validates_every_record_row(tmp_path):
    before=tmp_path/'before.whl';after=tmp_path/'after.whl';original(before)
    result=package.repack(before,after,{'NOTICE.test.txt':b'license notice'},'metadata note')
    assert result['runtime_bytes_identical'] and result['record_verified']
    assert set(result['preserved_runtime_members'])=={
        'provider/native.pyd','provider/__init__.py','provider.libs/exact-name.dll','provider-1.dist-info/WHEEL'}
    assert result['original_sha256']!=result['refreshed_sha256']
    with pytest.raises(FileExistsError):package.repack(before,after,{},'retry forbidden')


@pytest.mark.parametrize('member',['provider/native.pyd','provider/__init__.py','provider.libs/exact-name.dll','provider-1.dist-info/WHEEL'])
def test_changed_runtime_or_tag_never_passes_byte_preservation(tmp_path,member):
    before=tmp_path/'before.whl';after=tmp_path/'after.whl';original(before)
    with zipfile.ZipFile(before) as src,zipfile.ZipFile(after,'x') as dst:
        for entry in src.infolist():dst.writestr(entry,b'tampered' if entry.filename==member else src.read(entry))
    with pytest.raises(ValueError,match='runtime member changed'):package.verify_wheel(before,after)


@pytest.mark.parametrize('name',['../native.pyd','/absolute','C:/absolute','../../escape'])
def test_path_traversal_refused(name):
    with pytest.raises(ValueError):package.safe_name(name)


def test_windows_relative_indexes_are_portable_on_linux():
    assert package.safe_name(r'wheels-final\linux311\provider.whl')=='wheels-final/linux311/provider.whl'


def test_notice_cannot_add_runtime_member(tmp_path):
    before=tmp_path/'before.whl';original(before)
    with pytest.raises(ValueError,match='permitted dist-info'):
        package.repack(before,tmp_path/'after.whl',{'new-runtime.py':b'bad'},'note')
