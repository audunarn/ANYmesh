param(
    [Parameter(Mandatory = $true)][string]$GateName,
    [Parameter(Mandatory = $true)][string]$Repository,
    [Parameter(Mandatory = $true)][string]$ExpectedHead,
    [Parameter(Mandatory = $true)][string]$ExpectedTree,
    [Parameter(Mandatory = $true)][string]$ExpectedPathManifestSha256,
    [Parameter(Mandatory = $true)][int]$ExpectedPathCount,
    [Parameter(Mandatory = $true)][string]$TempRoot,
    [Parameter(Mandatory = $true)][string]$ResultPath,
    [Parameter(Mandatory = $true)][string]$LogPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$gate = [Threading.EventWaitHandle]::OpenExisting($GateName)
try {
    if (-not $gate.WaitOne([TimeSpan]::FromSeconds(30))) {
        throw 'The coordinator did not release the startup gate within 30 seconds'
    }
}
finally {
    $gate.Dispose()
}

$utf8 = [Text.UTF8Encoding]::new($false)
$startedUtc = [DateTimeOffset]::UtcNow

function Assert-PlainDirectory {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $false)][string]$ExpectedParent,
        [Parameter(Mandatory = $false)][string]$ExpectedLeaf
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "Required directory is absent or not a directory: $Path"
    }
    $item = Get-Item -LiteralPath $Path -Force
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Reparse-point directory is forbidden: $($item.FullName)"
    }
    $full = [IO.Path]::GetFullPath($item.FullName).TrimEnd([IO.Path]::DirectorySeparatorChar)
    if ($PSBoundParameters.ContainsKey('ExpectedParent')) {
        $parentFull = [IO.Path]::GetFullPath($ExpectedParent).TrimEnd([IO.Path]::DirectorySeparatorChar)
        if (-not [string]::Equals([IO.Path]::GetDirectoryName($full), $parentFull, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Directory is not an exact child of $parentFull : $full"
        }
    }
    if ($PSBoundParameters.ContainsKey('ExpectedLeaf') -and [IO.Path]::GetFileName($full) -cne $ExpectedLeaf) {
        throw "Directory leaf mismatch: expected $ExpectedLeaf, got $([IO.Path]::GetFileName($full))"
    }
    return $full
}

function New-PlainChildDirectory {
    param(
        [Parameter(Mandatory = $true)][string]$Parent,
        [Parameter(Mandatory = $true)][string]$Leaf
    )
    $parentFull = Assert-PlainDirectory -Path $Parent
    $path = [IO.Path]::GetFullPath((Join-Path $parentFull $Leaf))
    if (-not [string]::Equals([IO.Path]::GetDirectoryName($path), $parentFull, [StringComparison]::OrdinalIgnoreCase) -or
        [IO.Path]::GetFileName($path) -cne $Leaf) {
        throw "Unsafe child directory path: $path"
    }
    if (Test-Path -LiteralPath $path) {
        throw "Refusing pre-existing worker directory: $path"
    }
    [IO.Directory]::CreateDirectory($path) | Out-Null
    return (Assert-PlainDirectory -Path $path -ExpectedParent $parentFull -ExpectedLeaf $Leaf)
}

function Get-FileIdentity {
    param(
        [Parameter(Mandatory = $true)][string]$Path
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required file is absent: $Path"
    }
    $item = Get-Item -LiteralPath $Path -Force
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Reparse-point file is forbidden: $($item.FullName)"
    }
    return [ordered]@{
        path = $item.FullName
        bytes = [int64]$item.Length
        sha256 = (Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}

function Get-Utf8LfSha256 {
    param(
        [Parameter(Mandatory = $true)][string[]]$Lines
    )
    $payload = [string]::Join([char]10, $Lines) + [char]10
    $bytes = [Text.Encoding]::UTF8.GetBytes($payload)
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        return [ordered]@{
            bytes = [int64]$bytes.Length
            sha256 = ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace('-', '').ToLowerInvariant()
        }
    }
    finally {
        $sha.Dispose()
    }
}

function Write-WorkerLog {
    param(
        [Parameter(Mandatory = $true)][string]$Message
    )
    $line = '{0} {1}{2}' -f [DateTimeOffset]::UtcNow.ToString('o'), $Message, [char]10
    [IO.File]::AppendAllText($LogPath, $line, $utf8)
    [Console]::Out.Write($line)
}

function Invoke-LoggedNative {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory
    )
    Write-WorkerLog -Message ('COMMAND cwd={0} exe={1} args={2}' -f $WorkingDirectory, $FilePath, ($Arguments -join ' '))
    Push-Location -LiteralPath $WorkingDirectory
    try {
        & $FilePath @Arguments 2>&1 | ForEach-Object {
            Write-WorkerLog -Message ('OUTPUT ' + $_.ToString())
        }
        $exitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
    Write-WorkerLog -Message ("EXIT $exitCode")
    if ($exitCode -ne 0) {
        throw "Native command failed with exit code ${exitCode}: $FilePath"
    }
}

function Invoke-CapturedJson {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory
    )
    Write-WorkerLog -Message ('JSON_COMMAND cwd={0} exe={1} args={2}' -f $WorkingDirectory, $FilePath, ($Arguments -join ' '))
    Push-Location -LiteralPath $WorkingDirectory
    try {
        $output = @(& $FilePath @Arguments 2>&1)
        $exitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
    foreach ($line in $output) {
        Write-WorkerLog -Message ('JSON_OUTPUT ' + $line.ToString())
    }
    if ($exitCode -ne 0) {
        throw "JSON command failed with exit code ${exitCode}: $FilePath"
    }
    $nonEmpty = @($output | ForEach-Object { $_.ToString() } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    if ($nonEmpty.Count -ne 1) {
        throw "Expected one JSON output line, found $($nonEmpty.Count)"
    }
    return ($nonEmpty[0] | ConvertFrom-Json)
}

function Assert-SafeZipName {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][bool]$Directory
    )
    if ([string]::IsNullOrWhiteSpace($Name) -or $Name.Contains('\') -or $Name.StartsWith('/') -or $Name -match '^[A-Za-z]:') {
        throw "Unsafe ZIP member name: $Name"
    }
    $trimmed = if ($Directory) { $Name.TrimEnd('/') } else { $Name }
    if ([string]::IsNullOrWhiteSpace($trimmed) -or $trimmed.Contains('//')) {
        throw "Unsafe ZIP member name: $Name"
    }
    foreach ($part in $trimmed.Split('/')) {
        if ($part -ceq '.' -or $part -ceq '..' -or [string]::IsNullOrEmpty($part)) {
            throw "Unsafe ZIP path component in: $Name"
        }
    }
}

$tempRootFull = Assert-PlainDirectory -Path $TempRoot
$repositoryFull = Assert-PlainDirectory -Path $Repository
$resultFull = [IO.Path]::GetFullPath($ResultPath)
$logFull = [IO.Path]::GetFullPath($LogPath)
if (-not [string]::Equals([IO.Path]::GetDirectoryName($resultFull), $tempRootFull, [StringComparison]::OrdinalIgnoreCase) -or
    [IO.Path]::GetFileName($resultFull) -cne 'worker_result.json') {
    throw "Unsafe worker result path: $resultFull"
}
if (-not [string]::Equals([IO.Path]::GetDirectoryName($logFull), $tempRootFull, [StringComparison]::OrdinalIgnoreCase) -or
    [IO.Path]::GetFileName($logFull) -cne 'worker_build.log') {
    throw "Unsafe worker log path: $logFull"
}
if ((Test-Path -LiteralPath $resultFull) -or (Test-Path -LiteralPath ($resultFull + '.partial')) -or (Test-Path -LiteralPath $logFull)) {
    throw 'Refusing a pre-existing worker output path'
}
$logStream = [IO.FileStream]::new($logFull, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::Read)
$logStream.Dispose()

try {
    Write-WorkerLog -Message 'STAGE_A_WORKER_START'

    $git = 'C:\Program Files\Git\cmd\git.exe'
    $basePython = 'C:\Python\Python313\python.exe'
    foreach ($requiredExecutable in @($git, $basePython)) {
        if (-not (Test-Path -LiteralPath $requiredExecutable -PathType Leaf)) {
            throw "Required executable is absent: $requiredExecutable"
        }
    }

    $actualHead = (& $git -C $repositoryFull rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $actualHead -cne $ExpectedHead) {
        throw "Worker HEAD mismatch: $actualHead"
    }
    $actualTree = (& $git -C $repositoryFull rev-parse ($ExpectedHead + '^{tree}')).Trim()
    if ($LASTEXITCODE -ne 0 -or $actualTree -cne $ExpectedTree) {
        throw "Worker tree mismatch: $actualTree"
    }

    $buildInputs = @(
        [ordered]@{ name = 'setuptools'; version = '83.0.0'; path = 'C:\Users\AudunArnesenNyhus\AppData\Local\ANYmesh\qualification-artifacts\pypi\build-stack\setuptools\83.0.0\setuptools-83.0.0-py3-none-any.whl'; bytes = 1008090L; sha256 = '29b23c360f22f414dc7336bb39178cc7bcbf6021ed2733cde173f09dba19abb3' },
        [ordered]@{ name = 'wheel'; version = '0.47.0'; path = 'C:\Users\AudunArnesenNyhus\AppData\Local\ANYmesh\qualification-artifacts\pypi\build-stack\wheel\0.47.0\wheel-0.47.0-py3-none-any.whl'; bytes = 32218L; sha256 = '212281cab4dff978f6cedd499cd893e1f620791ca6ff7107cf270781e587eced' },
        [ordered]@{ name = 'packaging'; version = '25.0'; path = 'C:\Users\AudunArnesenNyhus\AppData\Local\ANYmesh\qualification-artifacts\pypi\build-stack\packaging\25.0\packaging-25.0-py3-none-any.whl'; bytes = 66469L; sha256 = '29572ef2b1f17581046b3a2227d5c611fb25ec70ca1ba8554b24b0e69331a484' },
        [ordered]@{ name = 'pyproject-hooks'; version = '1.2.0'; path = 'C:\Users\AudunArnesenNyhus\AppData\Local\ANYmesh\qualification-artifacts\pypi\build-stack\pyproject-hooks\1.2.0\pyproject_hooks-1.2.0-py3-none-any.whl'; bytes = 10216L; sha256 = '9e5c6bfa8dcc30091c74b0cf803c81fdd29d94f01992a7707bc97babb1141913' },
        [ordered]@{ name = 'colorama'; version = '0.4.6'; path = 'C:\Users\AudunArnesenNyhus\AppData\Local\ANYmesh\qualification-artifacts\pypi\build-stack\colorama\0.4.6\colorama-0.4.6-py2.py3-none-any.whl'; bytes = 25335L; sha256 = '4f1d9991f5acc0ca119f9d443620b77f9d6b33703e51011c16baf57afb285fc6' },
        [ordered]@{ name = 'build'; version = '1.5.0'; path = 'C:\Users\AudunArnesenNyhus\AppData\Local\ANYmesh\qualification-artifacts\pypi\build-stack\build\1.5.0\build-1.5.0-py3-none-any.whl'; bytes = 26018L; sha256 = '13f3eecb844759ab66efec90ca17639bbf14dc06cb2fdf37a9010322d9c50a6f' }
    )
    $inputPre = [Collections.Generic.List[object]]::new()
    foreach ($input in $buildInputs) {
        $identity = Get-FileIdentity -Path $input.path
        if ($identity.bytes -ne $input.bytes -or $identity.sha256 -cne $input.sha256) {
            throw "Offline build input mismatch: $($input.name)"
        }
        [void]$inputPre.Add([ordered]@{ name = $input.name; version = $input.version; path = $identity.path; bytes = $identity.bytes; sha256 = $identity.sha256 })
    }

    $archive = Join-Path $tempRootFull 'source.zip'
    $source = New-PlainChildDirectory -Parent $tempRootFull -Leaf 'source'
    $dist = New-PlainChildDirectory -Parent $tempRootFull -Leaf 'dist'
    if (Test-Path -LiteralPath $archive) {
        throw "Refusing pre-existing source archive: $archive"
    }
    Invoke-LoggedNative -FilePath $git -Arguments @('-C', $repositoryFull, 'archive', '--format=zip', ('--output=' + $archive), $ExpectedHead) -WorkingDirectory $tempRootFull

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archiveZip = [IO.Compression.ZipFile]::OpenRead($archive)
    $archiveFiles = [Collections.Generic.List[string]]::new()
    $archiveSeen = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    try {
        foreach ($entry in $archiveZip.Entries) {
            $isDirectory = $entry.FullName.EndsWith('/')
            Assert-SafeZipName -Name $entry.FullName -Directory $isDirectory
            if (-not $archiveSeen.Add($entry.FullName)) {
                throw "Duplicate source archive member: $($entry.FullName)"
            }
            $unixType = (($entry.ExternalAttributes -shr 16) -band 0xF000)
            if ($unixType -eq 0xA000) {
                throw "Symbolic links are forbidden in the source archive: $($entry.FullName)"
            }
            if (-not $isDirectory) {
                [void]$archiveFiles.Add($entry.FullName)
            }
        }
    }
    finally {
        $archiveZip.Dispose()
    }
    $archiveFileArray = $archiveFiles.ToArray()
    [Array]::Sort($archiveFileArray, [StringComparer]::Ordinal)
    if ($archiveFileArray.Count -ne $ExpectedPathCount) {
        throw "Source archive file-count mismatch: $($archiveFileArray.Count)"
    }
    $archiveManifest = Get-Utf8LfSha256 -Lines $archiveFileArray
    if ($archiveManifest.sha256 -cne $ExpectedPathManifestSha256.ToLowerInvariant()) {
        throw "Source archive path-manifest mismatch: $($archiveManifest.sha256)"
    }
    [IO.Compression.ZipFile]::ExtractToDirectory($archive, $source)
    $sourcePrefix = $source.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    $extractedFiles = @(Get-ChildItem -LiteralPath $source -Recurse -File -Force | ForEach-Object {
        if (-not $_.FullName.StartsWith($sourcePrefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Extracted path escaped source root: $($_.FullName)"
        }
        $_.FullName.Substring($sourcePrefix.Length).Replace('\', '/')
    })
    [Array]::Sort($extractedFiles, [StringComparer]::Ordinal)
    if (($extractedFiles -join [char]10) -cne ($archiveFileArray -join [char]10)) {
        throw 'Extracted source paths do not exactly match the ZIP member paths'
    }
    $archiveIdentity = Get-FileIdentity -Path $archive

    $venv = Join-Path $tempRootFull 'build-venv'
    if (Test-Path -LiteralPath $venv) {
        throw "Refusing pre-existing build venv: $venv"
    }
    Invoke-LoggedNative -FilePath $basePython -Arguments @('-m', 'venv', '--without-pip', $venv) -WorkingDirectory $tempRootFull
    $venv = Assert-PlainDirectory -Path $venv -ExpectedParent $tempRootFull -ExpectedLeaf 'build-venv'
    $venvPython = Join-Path $venv 'Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
        throw 'Fresh build venv has no Python executable'
    }
    $venvConfig = Get-Content -LiteralPath (Join-Path $venv 'pyvenv.cfg')
    if (@($venvConfig | Where-Object { $_ -match '^\s*include-system-site-packages\s*=\s*false\s*$' }).Count -ne 1) {
        throw 'Fresh build venv does not disable system site packages'
    }
    Invoke-LoggedNative -FilePath $venvPython -Arguments @('-m', 'ensurepip', '--default-pip') -WorkingDirectory $tempRootFull

    $environmentProbe = @'
import importlib.metadata as md
import json
import pathlib
import pip
import sys

venv = pathlib.Path(sys.prefix).resolve()

def under(path):
    return pathlib.Path(path).resolve().is_relative_to(venv)

result = {
    "prefix": str(venv),
    "executable": str(pathlib.Path(sys.executable).resolve()),
    "pip_version": md.version("pip"),
    "pip_origin": str(pathlib.Path(pip.__file__).resolve()),
    "pip_under_venv": under(pip.__file__),
    "packages": {},
}
for name in ("setuptools", "wheel", "packaging", "pyproject-hooks", "colorama", "build"):
    try:
        dist = md.distribution(name)
    except md.PackageNotFoundError:
        continue
    result["packages"][name] = {
        "version": dist.version,
        "root": str(pathlib.Path(dist.locate_file("")).resolve()),
        "under_venv": under(dist.locate_file("")),
    }
print(json.dumps(result, sort_keys=True, separators=(",", ":")))
'@
    $environmentProbePath = Join-Path $tempRootFull 'build_environment_probe.py'
    [IO.File]::WriteAllText($environmentProbePath, $environmentProbe, $utf8)
    $venvPre = Invoke-CapturedJson -FilePath $venvPython -Arguments @($environmentProbePath) -WorkingDirectory $tempRootFull
    if ($venvPre.pip_version -cne '25.2' -or -not $venvPre.pip_under_venv -or @($venvPre.packages.psobject.Properties).Count -ne 0) {
        throw 'Fresh build venv pre-install identity is not isolated pip 25.2'
    }

    foreach ($input in $buildInputs) {
        Invoke-LoggedNative -FilePath $venvPython -Arguments @('-m', 'pip', 'install', '--no-index', '--no-deps', '--no-cache-dir', $input.path) -WorkingDirectory $tempRootFull
    }
    $venvPostInstall = Invoke-CapturedJson -FilePath $venvPython -Arguments @($environmentProbePath) -WorkingDirectory $tempRootFull
    if ($venvPostInstall.pip_version -cne '25.2' -or -not $venvPostInstall.pip_under_venv) {
        throw 'Build venv pip identity drifted after offline installs'
    }
    foreach ($input in $buildInputs) {
        $property = $venvPostInstall.packages.psobject.Properties[$input.name]
        if ($null -eq $property -or $property.Value.version -cne $input.version -or -not $property.Value.under_venv) {
            throw "Build venv package identity mismatch: $($input.name)"
        }
    }

    Invoke-LoggedNative -FilePath $venvPython -Arguments @('-m', 'build', '--wheel', '--no-isolation', '--outdir', $dist) -WorkingDirectory $source
    $wheels = @(Get-ChildItem -LiteralPath $dist -Filter '*.whl' -File -Force)
    if ($wheels.Count -ne 1 -or $wheels[0].Name -cne 'anymesher-0.2.1-cp313-cp313-win_amd64.whl') {
        throw "Expected exactly anymesher-0.2.1-cp313-cp313-win_amd64.whl, found: $($wheels.Name -join ', ')"
    }
    $wheelIdentity = Get-FileIdentity -Path $wheels[0].FullName

    $wheelValidator = @'
import base64
import csv
import hashlib
import io
import json
import pathlib
import re
import sys
import zipfile
from email import policy
from email.parser import BytesParser

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

wheel_path = pathlib.Path(sys.argv[1]).resolve()
expected_name = "anymesher-0.2.1-cp313-cp313-win_amd64.whl"
if wheel_path.name != expected_name:
    raise SystemExit(f"unexpected wheel name: {wheel_path.name}")

def unsafe(name):
    if not name or "\\" in name or name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        return True
    if name.endswith("/") or "//" in name:
        return True
    return any(part in ("", ".", "..") for part in name.split("/"))

with zipfile.ZipFile(wheel_path, "r") as archive:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if any(info.is_dir() for info in infos):
        raise SystemExit("directory entries are forbidden in the wheel")
    if len(names) != len(set(names)):
        raise SystemExit("duplicate wheel member")
    if any(unsafe(name) for name in names):
        raise SystemExit("unsafe wheel member")
    if any(name.startswith(("src/", "tests/", "C:/", "C:\\")) for name in names):
        raise SystemExit("source-tree path present in wheel")

    metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
    wheel_names = [name for name in names if name.endswith(".dist-info/WHEEL")]
    record_names = [name for name in names if name.endswith(".dist-info/RECORD")]
    native_names = [name for name in names if re.fullmatch(r"anymesher/_native[^/]*\.pyd", name, re.IGNORECASE)]
    if len(metadata_names) != 1 or len(wheel_names) != 1 or len(record_names) != 1 or len(native_names) != 1:
        raise SystemExit("wheel singleton member contract failed")

    metadata = BytesParser(policy=policy.default).parsebytes(archive.read(metadata_names[0]))
    if canonicalize_name(metadata["Name"] or "") != "anymesher" or metadata["Version"] != "0.2.1":
        raise SystemExit("METADATA name/version mismatch")
    raw_requirements = metadata.get_all("Requires-Dist", [])
    if len(raw_requirements) != 2:
        raise SystemExit(f"expected two Requires-Dist values, found {len(raw_requirements)}")
    expected = {
        "numpy": frozenset({(">=", "1.26")}),
        "anygeometry": frozenset({(">=", "0.2"), ("<", "0.3")}),
    }
    normalized = {}
    for raw in raw_requirements:
        requirement = Requirement(raw)
        name = canonicalize_name(requirement.name)
        if name in normalized:
            raise SystemExit(f"duplicate requirement: {name}")
        if requirement.extras or requirement.marker is not None or requirement.url is not None:
            raise SystemExit(f"unapproved requirement qualifier: {raw}")
        specifiers = frozenset((item.operator, item.version) for item in requirement.specifier)
        if name not in expected or specifiers != expected[name]:
            raise SystemExit(f"unapproved runtime requirement: {raw}")
        normalized[name] = {
            "raw": raw,
            "specifiers": sorted([list(item) for item in specifiers]),
        }
    if set(normalized) != set(expected):
        raise SystemExit("normalized runtime requirement set mismatch")

    wheel_headers = BytesParser(policy=policy.default).parsebytes(archive.read(wheel_names[0]))
    tags = wheel_headers.get_all("Tag", [])
    if tags != ["cp313-cp313-win_amd64"]:
        raise SystemExit(f"wheel tag mismatch: {tags}")

    rows = list(csv.reader(io.TextIOWrapper(io.BytesIO(archive.read(record_names[0])), encoding="utf-8", newline="")))
    if any(len(row) != 3 for row in rows):
        raise SystemExit("RECORD row width mismatch")
    if len(rows) != len(names):
        raise SystemExit("RECORD/member count mismatch")
    record_paths = [row[0] for row in rows]
    if len(record_paths) != len(set(record_paths)) or set(record_paths) != set(names):
        raise SystemExit("RECORD path coverage mismatch")
    record_by_path = {row[0]: row[1:] for row in rows}
    member_records = []
    for info in sorted(infos, key=lambda value: value.filename):
        data = archive.read(info.filename)
        digest_field, size_field = record_by_path[info.filename]
        digest_hex = hashlib.sha256(data).hexdigest()
        if info.filename == record_names[0]:
            if digest_field or size_field:
                raise SystemExit("RECORD self-row must have blank digest and size")
            record_verified = True
        else:
            expected_digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
            if digest_field != "sha256=" + expected_digest:
                raise SystemExit(f"RECORD digest mismatch: {info.filename}")
            if size_field != str(len(data)):
                raise SystemExit(f"RECORD size mismatch: {info.filename}")
            record_verified = True
        member_records.append({
            "path": info.filename,
            "bytes": len(data),
            "sha256": digest_hex,
            "record_verified": record_verified,
        })

result = {
    "wheel_name": wheel_path.name,
    "metadata_path": metadata_names[0],
    "wheel_metadata_path": wheel_names[0],
    "record_path": record_names[0],
    "compiled_members": native_names,
    "normalized_requirements": normalized,
    "tags": tags,
    "member_count": len(member_records),
    "members": member_records,
    "record_complete": True,
}
print(json.dumps(result, sort_keys=True, separators=(",", ":")))
'@
    $wheelValidatorPath = Join-Path $tempRootFull 'validate_wheel.py'
    [IO.File]::WriteAllText($wheelValidatorPath, $wheelValidator, $utf8)
    $wheelValidation = Invoke-CapturedJson -FilePath $venvPython -Arguments @($wheelValidatorPath, $wheelIdentity.path) -WorkingDirectory $tempRootFull

    $venvPostBuild = Invoke-CapturedJson -FilePath $venvPython -Arguments @($environmentProbePath) -WorkingDirectory $tempRootFull
    if ($venvPostBuild.pip_version -cne '25.2' -or -not $venvPostBuild.pip_under_venv) {
        throw 'Build venv pip identity drifted after wheel build'
    }
    foreach ($input in $buildInputs) {
        $property = $venvPostBuild.packages.psobject.Properties[$input.name]
        if ($null -eq $property -or $property.Value.version -cne $input.version -or -not $property.Value.under_venv) {
            throw "Post-build package identity mismatch: $($input.name)"
        }
    }

    $inputPost = [Collections.Generic.List[object]]::new()
    foreach ($input in $buildInputs) {
        $identity = Get-FileIdentity -Path $input.path
        if ($identity.bytes -ne $input.bytes -or $identity.sha256 -cne $input.sha256) {
            throw "Offline build input changed during build: $($input.name)"
        }
        [void]$inputPost.Add([ordered]@{ name = $input.name; version = $input.version; path = $identity.path; bytes = $identity.bytes; sha256 = $identity.sha256 })
    }

    $completedUtc = [DateTimeOffset]::UtcNow
    $result = [ordered]@{
        schema_version = 1
        success = $true
        started_utc = $startedUtc.ToString('o')
        completed_utc = $completedUtc.ToString('o')
        elapsed_seconds = ($completedUtc - $startedUtc).TotalSeconds
        expected_head = $ExpectedHead
        actual_head = $actualHead
        expected_tree = $ExpectedTree
        actual_tree = $actualTree
        source_archive = [ordered]@{
            path = $archiveIdentity.path
            bytes = $archiveIdentity.bytes
            sha256 = $archiveIdentity.sha256
            file_count = $archiveFileArray.Count
            path_manifest_bytes = $archiveManifest.bytes
            path_manifest_sha256 = $archiveManifest.sha256
            paths = $archiveFileArray
            extracted_paths_equal = $true
            byte_reproducibility_claim = $false
        }
        build_venv = [ordered]@{
            path = $venv
            system_site_packages = $false
            pre_install = $venvPre
            post_install = $venvPostInstall
            post_build = $venvPostBuild
            install_order = @($buildInputs | ForEach-Object { $_.name + '==' + $_.version })
            install_flags = @('--no-index', '--no-deps', '--no-cache-dir')
        }
        build_inputs_pre = @($inputPre)
        build_inputs_post = @($inputPost)
        build = [ordered]@{
            frontend = 'build==1.5.0'
            backend = 'setuptools.build_meta'
            isolation = $false
            command = 'python -m build --wheel --no-isolation --outdir <fresh-dist>'
        }
        wheel = [ordered]@{
            path = $wheelIdentity.path
            name = [IO.Path]::GetFileName($wheelIdentity.path)
            bytes = $wheelIdentity.bytes
            sha256 = $wheelIdentity.sha256
            validation = $wheelValidation
        }
    }
    $resultJson = ($result | ConvertTo-Json -Depth 30) + [char]10
    $resultPartial = $resultFull + '.partial'
    if ((Test-Path -LiteralPath $resultFull) -or (Test-Path -LiteralPath $resultPartial)) {
        throw 'Worker result target became non-fresh'
    }
    $resultBytes = $utf8.GetBytes($resultJson)
    $resultStream = [IO.FileStream]::new($resultPartial, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try {
        $resultStream.Write($resultBytes, 0, $resultBytes.Length)
        $resultStream.Flush($true)
    }
    finally {
        $resultStream.Dispose()
    }
    [IO.File]::Move($resultPartial, $resultFull)
    Write-WorkerLog -Message 'STAGE_A_WORKER_SUCCESS'
}
catch {
    try {
        Write-WorkerLog -Message ('STAGE_A_WORKER_FAILURE ' + $_.Exception.ToString())
    }
    catch {
        [Console]::Error.WriteLine($_.Exception.ToString())
    }
    throw
}
