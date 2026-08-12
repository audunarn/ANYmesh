param(
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-fA-F]{64}$')][string]$ExpectedCoordinatorSha256,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-fA-F]{64}$')][string]$ExpectedWorkerSha256
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$expectedCoordinatorHash = $ExpectedCoordinatorSha256.ToLowerInvariant()
$expectedWorkerHash = $ExpectedWorkerSha256.ToLowerInvariant()
$expectedHead = '574fac99db064cc447bdb3e91ff029047a3c2248'
$expectedTree = 'c3d6a66aaeeab4cc1c7770f2d3290112f0c55a33'
$expectedPathManifestSha256 = 'cb3aa53b849774b234aff5342a224e16cdc4ae4fe4eeb8af19b9fa743164e9c3'
$expectedPathCount = 106
$repository = 'C:\Github\ANYmesh'
$planPath = 'C:\Github\ANYmesh\reports\native_hybrid\windows_wheel_attempt_4_plan.md'
$planSha256 = '50f30c5f3619cd2fe4b2dfc7680668b3ccb4af9f4a1c1388c3461df3e26268ca'
$compiledPlanPath = 'C:\Github\ANYmesh\reports\native_hybrid\compiled_triangulation_addendum.md'
$compiledPlanSha256 = 'e368269c7b9ec2c3e9912ea9baccb78b3d004330a8f5297082aeb198d19f4a92'
$historicalCompiledPlanSha256 = 'a64e3dc1dc7733a6ed065e85c7475b49a1071840be7a075d976f13935cffbd95'
$workerPath = 'C:\Github\ANYmesh\reports\native_hybrid\attempt4\stage_a_worker.ps1'
$powershellPath = 'C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe'
$gitPath = 'C:\Program Files\Git\cmd\git.exe'
$basePython = 'C:\Python\Python313\python.exe'
$attempt3Report = 'C:\Github\ANYmesh\reports\native_hybrid\wheel_smoke_windows_attempt_3.json'
$jobMemoryLimit = 1610612736L
$combinedMemoryLimit = 1879048192L
$workerDeadlineSeconds = 210
$cleanupMarginSeconds = 60
$utf8 = [Text.UTF8Encoding]::new($false)

$priorReports = @(
    [ordered]@{ label = 'attempt_1'; path = 'C:\Github\ANYmesh\reports\native_hybrid\wheel_smoke_windows.json'; expected_sha256 = 'fec903e15c24a8a028c7a3b084f52a3a32192650e3e7637480e855ea0171e813' },
    [ordered]@{ label = 'attempt_2'; path = 'C:\Github\ANYmesh\reports\native_hybrid\wheel_smoke_windows_attempt_2.json'; expected_sha256 = 'a6e3ba9e570986109fded3bdd9c13f508dc5fdd4ae4549c0fc029daeed2512f6' }
)

$buildInputs = @(
    [ordered]@{ name = 'setuptools'; version = '83.0.0'; path = 'C:\Users\AudunArnesenNyhus\AppData\Local\ANYmesh\qualification-artifacts\pypi\build-stack\setuptools\83.0.0\setuptools-83.0.0-py3-none-any.whl'; bytes = 1008090L; sha256 = '29b23c360f22f414dc7336bb39178cc7bcbf6021ed2733cde173f09dba19abb3' },
    [ordered]@{ name = 'wheel'; version = '0.47.0'; path = 'C:\Users\AudunArnesenNyhus\AppData\Local\ANYmesh\qualification-artifacts\pypi\build-stack\wheel\0.47.0\wheel-0.47.0-py3-none-any.whl'; bytes = 32218L; sha256 = '212281cab4dff978f6cedd499cd893e1f620791ca6ff7107cf270781e587eced' },
    [ordered]@{ name = 'packaging'; version = '25.0'; path = 'C:\Users\AudunArnesenNyhus\AppData\Local\ANYmesh\qualification-artifacts\pypi\build-stack\packaging\25.0\packaging-25.0-py3-none-any.whl'; bytes = 66469L; sha256 = '29572ef2b1f17581046b3a2227d5c611fb25ec70ca1ba8554b24b0e69331a484' },
    [ordered]@{ name = 'pyproject-hooks'; version = '1.2.0'; path = 'C:\Users\AudunArnesenNyhus\AppData\Local\ANYmesh\qualification-artifacts\pypi\build-stack\pyproject-hooks\1.2.0\pyproject_hooks-1.2.0-py3-none-any.whl'; bytes = 10216L; sha256 = '9e5c6bfa8dcc30091c74b0cf803c81fdd29d94f01992a7707bc97babb1141913' },
    [ordered]@{ name = 'colorama'; version = '0.4.6'; path = 'C:\Users\AudunArnesenNyhus\AppData\Local\ANYmesh\qualification-artifacts\pypi\build-stack\colorama\0.4.6\colorama-0.4.6-py2.py3-none-any.whl'; bytes = 25335L; sha256 = '4f1d9991f5acc0ca119f9d443620b77f9d6b33703e51011c16baf57afb285fc6' },
    [ordered]@{ name = 'build'; version = '1.5.0'; path = 'C:\Users\AudunArnesenNyhus\AppData\Local\ANYmesh\qualification-artifacts\pypi\build-stack\build\1.5.0\build-1.5.0-py3-none-any.whl'; bytes = 26018L; sha256 = '13f3eecb844759ab66efec90ca17639bbf14dc06cb2fdf37a9010322d9c50a6f' }
)

$toolAnchors = @(
    [ordered]@{ label = 'powershell'; path = $powershellPath; bytes = 454656L; sha256 = '7600ffe12da441fe89d035b13801e8e91d064bc544a27b19a5cf49f6ab8b18f5'; file_version = '10.0.26100.8875 (WinBuild.160101.0800)' },
    [ordered]@{ label = 'git'; path = $gitPath; bytes = 46920L; sha256 = '7b7971dd13f0c3a284e538601f2f9770b3a87dfaccb5fb52d68141c67ed22364'; file_version = '2.55.0.windows.3' },
    [ordered]@{ label = 'python'; path = $basePython; bytes = 105816L; sha256 = '08a64dc73ac3e3776b49f0097c6306bdb9c8f7990a037065213324d328467bf5'; file_version = '3.13.9' },
    [ordered]@{ label = 'python_dll'; path = 'C:\Python\Python313\python313.dll'; bytes = 6125912L; sha256 = 'c9f98606d0d06f4e8ae75ae385021e58b57c90d4fd325c0313c8c42abe1ebf63'; file_version = '3.13.9' },
    [ordered]@{ label = 'python_lib'; path = 'C:\Python\Python313\libs\python313.lib'; bytes = 368882L; sha256 = 'd4e5ca91fdde3d8fab4a2276cc329abe4e63481279294634842b35673539a316'; file_version = $null },
    [ordered]@{ label = 'python_header'; path = 'C:\Python\Python313\Include\Python.h'; bytes = 4178L; sha256 = '1092f5e36a87909d0b0f5d0b0d8f8505454753c99a65c115df396bc13ced8cd0'; file_version = $null },
    [ordered]@{ label = 'ensurepip'; path = 'C:\Python\Python313\Lib\ensurepip\_bundled\pip-25.2-py3-none-any.whl'; bytes = 1752557L; sha256 = '6d67a2b4e7f14d8b31b8b52648866fa717f45a1eb70e83002f4331d07e953717'; file_version = $null },
    [ordered]@{ label = 'cl'; path = 'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.50.35717\bin\HostX86\x64\cl.exe'; bytes = 604744L; sha256 = 'fd30d75e6aa319673cf3a4f56aeb3a1d6106aff87360b78966a7c5783567b78a'; file_version = '19.50.35725.0' },
    [ordered]@{ label = 'link'; path = 'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.50.35717\bin\HostX86\x64\link.exe'; bytes = 2921032L; sha256 = '69ee768e3ba674087b8644e1d46b81379bef1580470461e61535a1075d1a0957'; file_version = '14.50.35725.0' },
    [ordered]@{ label = 'vcruntime_header'; path = 'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.50.35717\include\vcruntime.h'; bytes = 11833L; sha256 = 'c301388c27f581c3a85257a70e892705a5712f32391d9d0211b486907be3d60e'; file_version = $null },
    [ordered]@{ label = 'vcruntime_lib'; path = 'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.50.35717\lib\x64\vcruntime.lib'; bytes = 285180L; sha256 = 'f2bd58d07cb4cf5a85978ee5807a2666742f9da0aa71fea1e3fe034997ec6653'; file_version = $null },
    [ordered]@{ label = 'libcpmt'; path = 'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.50.35717\lib\x64\libcpmt.lib'; bytes = 22491130L; sha256 = '3c6f83f971466c768b532b4a672a511acf4b145e8a7dc38cb065a613f1e310a7'; file_version = $null },
    [ordered]@{ label = 'windows_header'; path = 'C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\um\Windows.h'; bytes = 7511L; sha256 = 'b337d661d03a4abefb7b86a2742ce1ad5d19b57cd8b858bd13e7bbcc1dbeeaaa'; file_version = $null },
    [ordered]@{ label = 'corecrt_header'; path = 'C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\ucrt\corecrt.h'; bytes = 127273L; sha256 = '822e503b81dd7b3d7df93ca22fced3672a5154484fd42054d5941e619bcf6cbc'; file_version = $null },
    [ordered]@{ label = 'kernel32_lib'; path = 'C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\um\x64\kernel32.lib'; bytes = 311908L; sha256 = '341c7d56125a03b458e4d5093e4c79b33123ccfdfd610fe236937b8e6f3134bb'; file_version = $null },
    [ordered]@{ label = 'ucrt_lib'; path = 'C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\ucrt\x64\ucrt.lib'; bytes = 285588L; sha256 = '7ef4eac926bf597d2f243f16cdfed7e0db22cb3ca34a1d7e088a84c994a03d66'; file_version = $null }
)

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

function Get-OrCreatePlainChild {
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
    if (-not (Test-Path -LiteralPath $path)) {
        [IO.Directory]::CreateDirectory($path) | Out-Null
    }
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
        file_version = $item.VersionInfo.FileVersion
        product_version = $item.VersionInfo.ProductVersion
    }
}

function Assert-ExpectedFile {
    param(
        [Parameter(Mandatory = $true)]$Expected
    )
    $actual = Get-FileIdentity -Path $Expected.path
    if ($actual.bytes -ne $Expected.bytes -or $actual.sha256 -cne $Expected.sha256) {
        $expectedLabel = if ($Expected.Contains('label')) { $Expected.label } else { $Expected.name }
        throw "File identity mismatch: $expectedLabel"
    }
    if ($null -ne $Expected.file_version -and $actual.file_version -cne $Expected.file_version) {
        throw "File-version mismatch: $($Expected.label)"
    }
    return $actual
}

function Get-RepositoryState {
    param(
        [Parameter(Mandatory = $true)][string]$RepositoryPath
    )
    $head = (& $gitPath -C $RepositoryPath rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'git rev-parse HEAD failed' }
    $tree = (& $gitPath -C $RepositoryPath rev-parse ($head + '^{tree}')).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'git rev-parse tree failed' }
    $tracked = @(& $gitPath -C $RepositoryPath status --porcelain=v1 --untracked-files=no)
    if ($LASTEXITCODE -ne 0) { throw 'git tracked-status query failed' }
    $untracked = @(& $gitPath -C $RepositoryPath ls-files --others --exclude-standard)
    if ($LASTEXITCODE -ne 0) { throw 'git untracked query failed' }
    $ignored = @(& $gitPath -C $RepositoryPath ls-files --others --ignored --exclude-standard)
    if ($LASTEXITCODE -ne 0) { throw 'git ignored query failed' }
    [Array]::Sort($untracked, [StringComparer]::Ordinal)
    [Array]::Sort($ignored, [StringComparer]::Ordinal)
    return [ordered]@{ head = $head; tree = $tree; tracked = $tracked; untracked = $untracked; ignored = $ignored }
}

function Compare-RepositoryState {
    param(
        [Parameter(Mandatory = $true)]$Before,
        [Parameter(Mandatory = $true)]$After
    )
    return ($Before.head -ceq $After.head -and
        $Before.tree -ceq $After.tree -and
        ($Before.tracked -join [char]10) -ceq ($After.tracked -join [char]10) -and
        ($Before.untracked -join [char]10) -ceq ($After.untracked -join [char]10) -and
        ($Before.ignored -join [char]10) -ceq ($After.ignored -join [char]10))
}

function Write-CreateNewUtf8 {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Text
    )
    $bytes = $utf8.GetBytes($Text)
    $stream = [IO.FileStream]::new($Path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try {
        $stream.Write($bytes, 0, $bytes.Length)
        $stream.Flush($true)
    }
    finally {
        $stream.Dispose()
    }
}

function Remove-ExactOwnedTree {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$ExpectedParent,
        [Parameter(Mandatory = $true)][string]$ExpectedLeaf
    )
    if (-not (Test-Path -LiteralPath $Path)) { return }
    $item = Get-Item -LiteralPath $Path -Force
    $full = [IO.Path]::GetFullPath($item.FullName).TrimEnd([IO.Path]::DirectorySeparatorChar)
    $parent = [IO.Path]::GetFullPath($ExpectedParent).TrimEnd([IO.Path]::DirectorySeparatorChar)
    if (-not [string]::Equals([IO.Path]::GetDirectoryName($full), $parent, [StringComparison]::OrdinalIgnoreCase) -or
        [IO.Path]::GetFileName($full) -cne $ExpectedLeaf) {
        throw "Refusing unsafe owned-tree cleanup: $full"
    }
    Remove-Item -LiteralPath $full -Recurse -Force
}

function Write-AtomicFailureReport {
    param(
        [Parameter(Mandatory = $true)][string]$PartialPath,
        [Parameter(Mandatory = $true)][string]$FinalPath,
        [Parameter(Mandatory = $true)]$Payload
    )
    if ((Test-Path -LiteralPath $PartialPath) -or (Test-Path -LiteralPath $FinalPath)) {
        throw 'Failure report target is not fresh'
    }
    Write-CreateNewUtf8 -Path $PartialPath -Text (($Payload | ConvertTo-Json -Depth 30) + [char]10)
    [IO.File]::Move($PartialPath, $FinalPath)
}

$jobType = @'
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Runtime.InteropServices;

public sealed class AnyMeshJob : IDisposable
{
    private IntPtr handle;

    [StructLayout(LayoutKind.Sequential)]
    private struct IO_COUNTERS
    {
        public UInt64 ReadOperationCount;
        public UInt64 WriteOperationCount;
        public UInt64 OtherOperationCount;
        public UInt64 ReadTransferCount;
        public UInt64 WriteTransferCount;
        public UInt64 OtherTransferCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct JOBOBJECT_BASIC_LIMIT_INFORMATION
    {
        public Int64 PerProcessUserTimeLimit;
        public Int64 PerJobUserTimeLimit;
        public UInt32 LimitFlags;
        public UIntPtr MinimumWorkingSetSize;
        public UIntPtr MaximumWorkingSetSize;
        public UInt32 ActiveProcessLimit;
        public UIntPtr Affinity;
        public UInt32 PriorityClass;
        public UInt32 SchedulingClass;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION
    {
        public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
        public IO_COUNTERS IoInfo;
        public UIntPtr ProcessMemoryLimit;
        public UIntPtr JobMemoryLimit;
        public UIntPtr PeakProcessMemoryUsed;
        public UIntPtr PeakJobMemoryUsed;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct JOBOBJECT_BASIC_ACCOUNTING_INFORMATION
    {
        public Int64 TotalUserTime;
        public Int64 TotalKernelTime;
        public Int64 ThisPeriodTotalUserTime;
        public Int64 ThisPeriodTotalKernelTime;
        public UInt32 TotalPageFaultCount;
        public UInt32 TotalProcesses;
        public UInt32 ActiveProcesses;
        public UInt32 TotalTerminatedProcesses;
    }

    public sealed class Snapshot
    {
        public ulong PeakProcessMemoryBytes;
        public ulong PeakJobMemoryBytes;
        public long TotalUserTime100ns;
        public long TotalKernelTime100ns;
        public uint TotalProcesses;
        public uint ActiveProcesses;
        public uint TotalTerminatedProcesses;
    }

    private const int JobObjectBasicAccountingInformation = 1;
    private const int JobObjectBasicProcessIdList = 3;
    private const int JobObjectExtendedLimitInformation = 9;
    private const uint JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200;
    private const uint JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000;

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr CreateJobObject(IntPtr attributes, string name);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool SetInformationJobObject(IntPtr job, int infoClass, IntPtr info, uint length);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool QueryInformationJobObject(IntPtr job, int infoClass, IntPtr info, uint length, out uint returnedLength);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool TerminateJobObject(IntPtr job, uint exitCode);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool CloseHandle(IntPtr handle);

    private AnyMeshJob(IntPtr value) { handle = value; }

    public static AnyMeshJob Create(string name, ulong memoryLimit)
    {
        IntPtr job = CreateJobObject(IntPtr.Zero, name);
        if (job == IntPtr.Zero) throw new Win32Exception(Marshal.GetLastWin32Error(), "CreateJobObject failed");
        var limits = new JOBOBJECT_EXTENDED_LIMIT_INFORMATION();
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_JOB_MEMORY;
        limits.JobMemoryLimit = new UIntPtr(memoryLimit);
        int size = Marshal.SizeOf(typeof(JOBOBJECT_EXTENDED_LIMIT_INFORMATION));
        IntPtr buffer = Marshal.AllocHGlobal(size);
        try
        {
            Marshal.StructureToPtr(limits, buffer, false);
            if (!SetInformationJobObject(job, JobObjectExtendedLimitInformation, buffer, (uint)size))
            {
                int error = Marshal.GetLastWin32Error();
                CloseHandle(job);
                throw new Win32Exception(error, "SetInformationJobObject failed");
            }
        }
        finally { Marshal.FreeHGlobal(buffer); }
        return new AnyMeshJob(job);
    }

    public void Assign(IntPtr processHandle)
    {
        if (!AssignProcessToJobObject(handle, processHandle))
            throw new Win32Exception(Marshal.GetLastWin32Error(), "AssignProcessToJobObject failed");
    }

    public void Terminate(uint exitCode)
    {
        if (!TerminateJobObject(handle, exitCode))
            throw new Win32Exception(Marshal.GetLastWin32Error(), "TerminateJobObject failed");
    }

    public long[] GetProcessIds()
    {
        const int capacity = 4096;
        int size = 8 + IntPtr.Size * capacity;
        IntPtr buffer = Marshal.AllocHGlobal(size);
        try
        {
            uint returned;
            if (!QueryInformationJobObject(handle, JobObjectBasicProcessIdList, buffer, (uint)size, out returned))
                throw new Win32Exception(Marshal.GetLastWin32Error(), "Query process list failed");
            uint count = (uint)Marshal.ReadInt32(buffer, 4);
            if (count > capacity) throw new InvalidOperationException("Job process list exceeded fixed capacity");
            var result = new long[count];
            for (int index = 0; index < count; ++index)
                result[index] = Marshal.ReadIntPtr(buffer, 8 + index * IntPtr.Size).ToInt64();
            return result;
        }
        finally { Marshal.FreeHGlobal(buffer); }
    }

    public Snapshot GetSnapshot()
    {
        var result = new Snapshot();
        int extendedSize = Marshal.SizeOf(typeof(JOBOBJECT_EXTENDED_LIMIT_INFORMATION));
        IntPtr extendedBuffer = Marshal.AllocHGlobal(extendedSize);
        int accountingSize = Marshal.SizeOf(typeof(JOBOBJECT_BASIC_ACCOUNTING_INFORMATION));
        IntPtr accountingBuffer = Marshal.AllocHGlobal(accountingSize);
        try
        {
            uint returned;
            if (!QueryInformationJobObject(handle, JobObjectExtendedLimitInformation, extendedBuffer, (uint)extendedSize, out returned))
                throw new Win32Exception(Marshal.GetLastWin32Error(), "Query extended limits failed");
            if (!QueryInformationJobObject(handle, JobObjectBasicAccountingInformation, accountingBuffer, (uint)accountingSize, out returned))
                throw new Win32Exception(Marshal.GetLastWin32Error(), "Query accounting failed");
            var extended = (JOBOBJECT_EXTENDED_LIMIT_INFORMATION)Marshal.PtrToStructure(extendedBuffer, typeof(JOBOBJECT_EXTENDED_LIMIT_INFORMATION));
            var accounting = (JOBOBJECT_BASIC_ACCOUNTING_INFORMATION)Marshal.PtrToStructure(accountingBuffer, typeof(JOBOBJECT_BASIC_ACCOUNTING_INFORMATION));
            result.PeakProcessMemoryBytes = extended.PeakProcessMemoryUsed.ToUInt64();
            result.PeakJobMemoryBytes = extended.PeakJobMemoryUsed.ToUInt64();
            result.TotalUserTime100ns = accounting.TotalUserTime;
            result.TotalKernelTime100ns = accounting.TotalKernelTime;
            result.TotalProcesses = accounting.TotalProcesses;
            result.ActiveProcesses = accounting.ActiveProcesses;
            result.TotalTerminatedProcesses = accounting.TotalTerminatedProcesses;
            return result;
        }
        finally
        {
            Marshal.FreeHGlobal(extendedBuffer);
            Marshal.FreeHGlobal(accountingBuffer);
        }
    }

    public void Dispose()
    {
        if (handle != IntPtr.Zero)
        {
            CloseHandle(handle);
            handle = IntPtr.Zero;
        }
    }
}
'@

Add-Type -TypeDefinition $jobType -Language CSharp

$localAppData = [Environment]::GetFolderPath([Environment+SpecialFolder]::LocalApplicationData)
if ([string]::IsNullOrWhiteSpace($localAppData)) { throw 'LocalApplicationData is unavailable' }
$localRoot = Assert-PlainDirectory -Path $localAppData
$tempParent = Assert-PlainDirectory -Path (Join-Path $localRoot 'Temp') -ExpectedParent $localRoot -ExpectedLeaf 'Temp'
$tempLeaf = 'anymesher-wheel-attempt-4-stage-a-574fac99-v1'
$tempRoot = [IO.Path]::GetFullPath((Join-Path $tempParent $tempLeaf))

$qualificationArtifacts = Get-OrCreatePlainChild -Parent $localRoot -Leaf 'ANYmesh'
$qualificationArtifacts = Get-OrCreatePlainChild -Parent $qualificationArtifacts -Leaf 'qualification-artifacts'
$qualificationArtifacts = Get-OrCreatePlainChild -Parent $qualificationArtifacts -Leaf 'anymesher'
$qualificationArtifacts = Get-OrCreatePlainChild -Parent $qualificationArtifacts -Leaf '0.2.1'
$qualificationArtifacts = Get-OrCreatePlainChild -Parent $qualificationArtifacts -Leaf 'windows-cp313'
$attempt4Root = Get-OrCreatePlainChild -Parent $qualificationArtifacts -Leaf 'attempt-4'
$bundleParent = Get-OrCreatePlainChild -Parent $attempt4Root -Leaf 'bundles'
$failureParent = Get-OrCreatePlainChild -Parent $attempt4Root -Leaf 'failures'
$stagingLeaf = '.stage-a-574fac99-v1.partial'
$stagingBundle = [IO.Path]::GetFullPath((Join-Path $bundleParent $stagingLeaf))
$failureLeaf = 'stage-a-574fac99-v1-failure.json'
$failurePath = [IO.Path]::GetFullPath((Join-Path $failureParent $failureLeaf))
$failurePartialLeaf = $failureLeaf + '.partial'
$failurePartial = [IO.Path]::GetFullPath((Join-Path $failureParent $failurePartialLeaf))

foreach ($candidate in @(
    [ordered]@{ path = $tempRoot; parent = $tempParent; leaf = $tempLeaf },
    [ordered]@{ path = $stagingBundle; parent = $bundleParent; leaf = $stagingLeaf },
    [ordered]@{ path = $failurePath; parent = $failureParent; leaf = $failureLeaf },
    [ordered]@{ path = $failurePartial; parent = $failureParent; leaf = $failurePartialLeaf }
)) {
    if (-not [string]::Equals([IO.Path]::GetDirectoryName($candidate.path), $candidate.parent, [StringComparison]::OrdinalIgnoreCase) -or
        [IO.Path]::GetFileName($candidate.path) -cne $candidate.leaf) {
        throw "Unsafe Stage A target path: $($candidate.path)"
    }
    if (Test-Path -LiteralPath $candidate.path) {
        throw "Refusing pre-existing Stage A target: $($candidate.path)"
    }
}

$taskEnvironmentNames = @(
    'PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV', 'PYTHONNOUSERSITE', 'PYTHONDONTWRITEBYTECODE',
    'PYTHONHASHSEED', 'ANYMESHER_REQUIRE_NATIVE', 'PIP_NO_INDEX', 'PIP_DISABLE_PIP_VERSION_CHECK',
    'PIP_CONFIG_FILE', 'PIP_REQUIRE_VIRTUALENV', 'SOURCE_DATE_EPOCH', 'HTTP_PROXY', 'HTTPS_PROXY',
    'ALL_PROXY', 'NO_PROXY'
)
$originalPath = [Environment]::GetEnvironmentVariable('PATH', 'Process')
$startedUtc = [DateTimeOffset]::UtcNow
$coordinatorEvents = [Collections.Generic.List[string]]::new()
$ownedTrees = [Collections.Generic.List[object]]::new()
$eligibilityPassed = $false
$published = $false
$primaryError = $null
$finalizationErrors = [Collections.Generic.List[string]]::new()
$job = $null
$gate = $null
$workerProcess = $null
$workerExit = $null
$terminationReason = $null
$workerResult = $null
$workerLogText = ''
$workerStdoutText = ''
$workerStderrText = ''
$repoPre = $null
$repoPost = $null
$toolPre = [Collections.Generic.List[object]]::new()
$toolPost = [Collections.Generic.List[object]]::new()
$inputPre = [Collections.Generic.List[object]]::new()
$inputPost = [Collections.Generic.List[object]]::new()
$priorPre = [Collections.Generic.List[object]]::new()
$priorPost = [Collections.Generic.List[object]]::new()
$observedPids = [Collections.Generic.HashSet[long]]::new()
$peakCoordinatorWorkingSet = 0L
$peakObservedJobWorkingSet = 0L
$peakObservedCombinedWorkingSet = 0L
$jobSnapshot = $null
$workerElapsedSeconds = $null
$cleanupElapsedSeconds = $null
$workerResultPath = $null
$workerLogPath = $null
$workerStdoutPath = $null
$workerStderrPath = $null
$finalBundle = $null
$bundleIndexSha256 = $null
$bundleEntries = $null

try {
    $repository = Assert-PlainDirectory -Path $repository
    $coordinatorIdentity = Get-FileIdentity -Path $PSCommandPath
    $workerIdentity = Get-FileIdentity -Path $workerPath
    if ($coordinatorIdentity.sha256 -cne $expectedCoordinatorHash) { throw 'Coordinator script hash mismatch' }
    if ($workerIdentity.sha256 -cne $expectedWorkerHash) { throw 'Worker script hash mismatch' }
    $planIdentity = Get-FileIdentity -Path $planPath
    $compiledPlanIdentity = Get-FileIdentity -Path $compiledPlanPath
    if ($planIdentity.sha256 -cne $planSha256) { throw 'Attempt-4 plan hash mismatch' }
    if ($compiledPlanIdentity.sha256 -cne $compiledPlanSha256) { throw 'Current compiled addendum hash mismatch' }

    foreach ($anchor in $toolAnchors) {
        [void]$toolPre.Add((Assert-ExpectedFile -Expected $anchor))
    }
    $pythonRuntime = @(& $basePython -c 'import json,platform,sys;print(json.dumps(dict(version=sys.version,implementation=sys.implementation.name,platform=platform.platform()),sort_keys=True))')
    if ($LASTEXITCODE -ne 0 -or $pythonRuntime.Count -ne 1) { throw 'CPython runtime identity probe failed' }
    $pythonRuntime = $pythonRuntime[0] | ConvertFrom-Json
    if ($pythonRuntime.version -cne '3.13.9 (tags/v3.13.9:8183fa5, Oct 14 2025, 14:09:13) [MSC v.1944 64 bit (AMD64)]' -or
        $pythonRuntime.implementation -cne 'cpython' -or $pythonRuntime.platform -cne 'Windows-11-10.0.26200-SP0') {
        throw 'CPython runtime identity drift'
    }
    if ($PSVersionTable.PSVersion.ToString() -cne '5.1.26100.8875') { throw 'PowerShell runtime version drift' }
    $gitVersion = (& $gitPath --version).Trim()
    if ($LASTEXITCODE -ne 0 -or $gitVersion -cne 'git version 2.55.0.windows.3') { throw 'Git runtime version drift' }
    $sdkIncludes = @(Get-ChildItem -LiteralPath 'C:\Program Files (x86)\Windows Kits\10\Include' -Directory -Force | ForEach-Object Name)
    $sdkLibs = @(Get-ChildItem -LiteralPath 'C:\Program Files (x86)\Windows Kits\10\Lib' -Directory -Force | ForEach-Object Name)
    [Array]::Sort($sdkIncludes, [StringComparer]::Ordinal)
    [Array]::Sort($sdkLibs, [StringComparer]::Ordinal)
    if (($sdkIncludes -join ',') -cne '10.0.26100.0' -or ($sdkLibs -join ',') -cne '10.0.26100.0') {
        throw 'Windows SDK version inventory drift'
    }
    if ($null -ne (Get-Command cmake -ErrorAction SilentlyContinue) -or $null -ne (Get-Command ninja -ErrorAction SilentlyContinue)) {
        throw 'CMake or Ninja is unexpectedly present'
    }

    $repoPre = Get-RepositoryState -RepositoryPath $repository
    if ($repoPre.head -cne $expectedHead -or $repoPre.tree -cne $expectedTree -or $repoPre.tracked.Count -ne 0) {
        throw 'Pinned source state is not exact and tracked-clean'
    }
    foreach ($prior in $priorReports) {
        $identity = Get-FileIdentity -Path $prior.path
        if ($identity.sha256 -cne $prior.expected_sha256) { throw "$($prior.label) hash mismatch" }
        [void]$priorPre.Add([ordered]@{ label = $prior.label; path = $identity.path; bytes = $identity.bytes; sha256 = $identity.sha256 })
    }
    if (Test-Path -LiteralPath $attempt3Report) { throw 'Attempt-3 report unexpectedly exists' }
    foreach ($input in $buildInputs) {
        $identity = Get-FileIdentity -Path $input.path
        if ($identity.bytes -ne $input.bytes -or $identity.sha256 -cne $input.sha256) { throw "$($input.name) input mismatch" }
        [void]$inputPre.Add([ordered]@{ name = $input.name; version = $input.version; path = $identity.path; bytes = $identity.bytes; sha256 = $identity.sha256 })
    }

    $commitEpoch = (& $gitPath -C $repository show -s --format=%ct $expectedHead).Trim()
    if ($LASTEXITCODE -ne 0 -or $commitEpoch -notmatch '^\d+$') { throw 'Could not derive SOURCE_DATE_EPOCH' }
    $eligibilityPassed = $true

    foreach ($name in $taskEnvironmentNames) { Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue }
    $env:PYTHONNOUSERSITE = '1'
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $env:PYTHONHASHSEED = '0'
    $env:ANYMESHER_REQUIRE_NATIVE = '1'
    $env:PIP_NO_INDEX = '1'
    $env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
    $env:PIP_CONFIG_FILE = 'NUL'
    $env:PIP_REQUIRE_VIRTUALENV = '1'
    $env:SOURCE_DATE_EPOCH = $commitEpoch
    $env:NO_PROXY = '*'
    $clDirectory = [IO.Path]::GetDirectoryName(($toolPre | Where-Object { $_.path -ieq $toolAnchors[7].path }).path)
    $gitDirectory = [IO.Path]::GetDirectoryName($gitPath)
    $env:PATH = $clDirectory + [IO.Path]::PathSeparator + $gitDirectory + [IO.Path]::PathSeparator + $originalPath

    [IO.Directory]::CreateDirectory($tempRoot) | Out-Null
    $tempRoot = Assert-PlainDirectory -Path $tempRoot -ExpectedParent $tempParent -ExpectedLeaf $tempLeaf
    [void]$ownedTrees.Add([ordered]@{ path = $tempRoot; parent = $tempParent; leaf = $tempLeaf })
    $workerResultPath = Join-Path $tempRoot 'worker_result.json'
    $workerLogPath = Join-Path $tempRoot 'worker_build.log'
    $workerStdoutPath = Join-Path $tempRoot 'worker_stdout.log'
    $workerStderrPath = Join-Path $tempRoot 'worker_stderr.log'

    $gateName = 'Local\ANYmesh-Attempt4-StageA-' + $PID + '-Gate'
    $createdNew = $false
    $gate = [Threading.EventWaitHandle]::new($false, [Threading.EventResetMode]::ManualReset, $gateName, [ref]$createdNew)
    if (-not $createdNew) { throw 'Named startup gate already existed' }
    $job = [AnyMeshJob]::Create(('Local\ANYmesh-Attempt4-StageA-' + $PID + '-Job'), [uint64]$jobMemoryLimit)

    $arguments = @(
        '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', $workerPath,
        '-GateName', $gateName,
        '-Repository', $repository,
        '-ExpectedHead', $expectedHead,
        '-ExpectedTree', $expectedTree,
        '-ExpectedPathManifestSha256', $expectedPathManifestSha256,
        '-ExpectedPathCount', $expectedPathCount.ToString([Globalization.CultureInfo]::InvariantCulture),
        '-TempRoot', $tempRoot,
        '-ResultPath', $workerResultPath,
        '-LogPath', $workerLogPath
    )
    [void]$coordinatorEvents.Add(([DateTimeOffset]::UtcNow.ToString('o') + ' START_WORKER'))
    $workerProcess = Start-Process -FilePath $powershellPath -ArgumentList $arguments -PassThru -WindowStyle Hidden -RedirectStandardOutput $workerStdoutPath -RedirectStandardError $workerStderrPath
    $job.Assign($workerProcess.Handle)
    [void]$observedPids.Add([long]$workerProcess.Id)
    [void]$gate.Set()

    $workerClock = [Diagnostics.Stopwatch]::StartNew()
    while (-not $workerProcess.HasExited) {
        $pids = @($job.GetProcessIds())
        [int64]$jobWorkingSet = 0
        foreach ($processId in $pids) {
            [void]$observedPids.Add([long]$processId)
            $ownedProcess = Get-Process -Id $processId -ErrorAction SilentlyContinue
            if ($null -ne $ownedProcess) {
                $jobWorkingSet += [int64]$ownedProcess.WorkingSet64
                if ([int64]$ownedProcess.WorkingSet64 -gt $peakObservedJobWorkingSet) { $peakObservedJobWorkingSet = [int64]$ownedProcess.WorkingSet64 }
            }
        }
        $coordinatorProcess = Get-Process -Id $PID
        [int64]$coordinatorWorkingSet = $coordinatorProcess.WorkingSet64
        if ($coordinatorWorkingSet -gt $peakCoordinatorWorkingSet) { $peakCoordinatorWorkingSet = $coordinatorWorkingSet }
        [int64]$combinedWorkingSet = $coordinatorWorkingSet + $jobWorkingSet
        if ($combinedWorkingSet -gt $peakObservedCombinedWorkingSet) { $peakObservedCombinedWorkingSet = $combinedWorkingSet }
        if ($combinedWorkingSet -ge $combinedMemoryLimit) {
            $terminationReason = 'combined_memory_limit'
            $job.Terminate(137)
            break
        }
        if ($workerClock.Elapsed.TotalSeconds -ge $workerDeadlineSeconds) {
            $terminationReason = 'worker_deadline'
            $job.Terminate(124)
            break
        }
        Start-Sleep -Milliseconds 100
        $workerProcess.Refresh()
    }
    if ($null -ne $terminationReason) {
        if (-not $workerProcess.WaitForExit(15000)) { throw 'Worker tree did not exit within 15 seconds of termination' }
    }
    else {
        $workerProcess.WaitForExit()
    }
    $workerClock.Stop()
    $workerElapsedSeconds = $workerClock.Elapsed.TotalSeconds
    $workerExit = $workerProcess.ExitCode
    $jobSnapshot = $job.GetSnapshot()
    $remainingJobPids = @($job.GetProcessIds())
    foreach ($processId in $remainingJobPids) { [void]$observedPids.Add([long]$processId) }
    if ($jobSnapshot.ActiveProcesses -ne 0 -or $remainingJobPids.Count -ne 0) { throw 'Job still has active processes after worker exit' }
    $job.Dispose()
    $job = $null
    $gate.Dispose()
    $gate = $null

    $lingeringOwnedProcesses = @($observedPids | Where-Object { $null -ne (Get-Process -Id $_ -ErrorAction SilentlyContinue) })
    if ($lingeringOwnedProcesses.Count -ne 0) { throw "Owned process remains: $($lingeringOwnedProcesses -join ',')" }
    if ($null -ne $terminationReason) { throw "Worker terminated by $terminationReason" }
    if ($workerExit -ne 0) { throw "Worker failed with exit code $workerExit" }
    if (-not (Test-Path -LiteralPath $workerResultPath -PathType Leaf) -or -not (Test-Path -LiteralPath $workerLogPath -PathType Leaf)) {
        throw 'Worker did not produce its result and build log'
    }
    $workerResult = Get-Content -LiteralPath $workerResultPath -Raw | ConvertFrom-Json
    if (-not $workerResult.success -or $workerResult.expected_head -cne $expectedHead -or $workerResult.actual_tree -cne $expectedTree) {
        throw 'Worker result identity mismatch'
    }
    $workerLogText = Get-Content -LiteralPath $workerLogPath -Raw
    $workerStdoutText = if (Test-Path -LiteralPath $workerStdoutPath -PathType Leaf) { Get-Content -LiteralPath $workerStdoutPath -Raw } else { '' }
    $workerStderrText = if (Test-Path -LiteralPath $workerStderrPath -PathType Leaf) { Get-Content -LiteralPath $workerStderrPath -Raw } else { '' }

    $wheelSource = [IO.Path]::GetFullPath([string]$workerResult.wheel.path)
    $tempPrefix = $tempRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    if (-not $wheelSource.StartsWith($tempPrefix, [StringComparison]::OrdinalIgnoreCase) -or
        [IO.Path]::GetFileName($wheelSource) -cne 'anymesher-0.2.1-cp313-cp313-win_amd64.whl') {
        throw "Worker wheel path escaped TEMP: $wheelSource"
    }
    $wheelSourceIdentity = Get-FileIdentity -Path $wheelSource
    if ($wheelSourceIdentity.bytes -ne [int64]$workerResult.wheel.bytes -or $wheelSourceIdentity.sha256 -cne [string]$workerResult.wheel.sha256) {
        throw 'Worker wheel identity mismatch'
    }

    [IO.Directory]::CreateDirectory($stagingBundle) | Out-Null
    $stagingBundle = Assert-PlainDirectory -Path $stagingBundle -ExpectedParent $bundleParent -ExpectedLeaf $stagingLeaf
    [void]$ownedTrees.Add([ordered]@{ path = $stagingBundle; parent = $bundleParent; leaf = $stagingLeaf })
    $bundleWheel = Join-Path $stagingBundle 'anymesher-0.2.1-cp313-cp313-win_amd64.whl'
    Copy-Item -LiteralPath $wheelSource -Destination $bundleWheel
    $bundleWheelIdentity = Get-FileIdentity -Path $bundleWheel
    if ($bundleWheelIdentity.bytes -ne $wheelSourceIdentity.bytes -or $bundleWheelIdentity.sha256 -cne $wheelSourceIdentity.sha256) {
        throw 'Staged wheel identity mismatch'
    }

    $buildLogPath = Join-Path $stagingBundle 'build.log'
    $buildLogText = @(
        '=== COORDINATOR EVENTS ===',
        ($coordinatorEvents -join [char]10),
        '=== WORKER BUILD LOG ===',
        $workerLogText,
        '=== WORKER STDOUT ===',
        $workerStdoutText,
        '=== WORKER STDERR ===',
        $workerStderrText
    ) -join [char]10
    Write-CreateNewUtf8 -Path $buildLogPath -Text ($buildLogText + [char]10)
    $buildLogIdentity = Get-FileIdentity -Path $buildLogPath

    $cleanupClock = [Diagnostics.Stopwatch]::StartNew()
    Remove-ExactOwnedTree -Path $tempRoot -ExpectedParent $tempParent -ExpectedLeaf $tempLeaf
    [void]$ownedTrees.Remove(($ownedTrees | Where-Object { $_.path -ieq $tempRoot } | Select-Object -First 1))
    $cleanupClock.Stop()
    $cleanupElapsedSeconds = $cleanupClock.Elapsed.TotalSeconds
    if ($cleanupElapsedSeconds -gt $cleanupMarginSeconds -or (Test-Path -LiteralPath $tempRoot)) {
        throw 'TEMP cleanup exceeded its margin or remained incomplete'
    }

    $env:PATH = $originalPath
    foreach ($name in $taskEnvironmentNames) { Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue }
    if ([Environment]::GetEnvironmentVariable('PATH', 'Process') -cne $originalPath -or
        @($taskEnvironmentNames | Where-Object { $null -ne [Environment]::GetEnvironmentVariable($_, 'Process') }).Count -ne 0) {
        throw 'Task environment restoration failed'
    }

    $repoPost = Get-RepositoryState -RepositoryPath $repository
    if (-not (Compare-RepositoryState -Before $repoPre -After $repoPost)) { throw 'Source inventory changed during Stage A' }
    foreach ($prior in $priorReports) {
        $identity = Get-FileIdentity -Path $prior.path
        if ($identity.sha256 -cne $prior.expected_sha256) { throw "$($prior.label) changed during Stage A" }
        [void]$priorPost.Add([ordered]@{ label = $prior.label; path = $identity.path; bytes = $identity.bytes; sha256 = $identity.sha256 })
    }
    if (Test-Path -LiteralPath $attempt3Report) { throw 'Attempt-3 report appeared during Stage A' }
    foreach ($input in $buildInputs) {
        $identity = Get-FileIdentity -Path $input.path
        if ($identity.bytes -ne $input.bytes -or $identity.sha256 -cne $input.sha256) { throw "$($input.name) changed during Stage A" }
        [void]$inputPost.Add([ordered]@{ name = $input.name; version = $input.version; path = $identity.path; bytes = $identity.bytes; sha256 = $identity.sha256 })
    }
    foreach ($anchor in $toolAnchors) {
        [void]$toolPost.Add((Assert-ExpectedFile -Expected $anchor))
    }
    if ($jobSnapshot.PeakJobMemoryBytes -ge [uint64]$jobMemoryLimit -or $peakObservedCombinedWorkingSet -ge $combinedMemoryLimit) {
        throw 'Recorded memory reached or exceeded a hard Stage A limit'
    }

    $completedUtc = [DateTimeOffset]::UtcNow
    $report = [ordered]@{
        schema_version = 1
        qualification = 'windows_cpython313_immutable_wheel_build_attempt_4_stage_a'
        success = $true
        platform_scope = 'Windows AMD64 CPython 3.13.9 only'
        performance_claim = $false
        byte_reproducibility_claim = $false
        network_allowed = $false
        started_utc = $startedUtc.ToString('o')
        completed_utc = $completedUtc.ToString('o')
        elapsed_seconds = ($completedUtc - $startedUtc).TotalSeconds
        source = [ordered]@{ expected_head = $expectedHead; expected_tree = $expectedTree; pre = $repoPre; post = $repoPost }
        authority = [ordered]@{
            governing_plan = 'C:\Users\AudunArnesenNyhus\Downloads\ANYmesher_native_hybrid_mesher_plan.md'
            attempt4_plan = [ordered]@{ path = $planIdentity.path; bytes = $planIdentity.bytes; sha256 = $planIdentity.sha256 }
            compiled_addendum_current = [ordered]@{ path = $compiledPlanIdentity.path; bytes = $compiledPlanIdentity.bytes; sha256 = $compiledPlanIdentity.sha256 }
            compiled_addendum_historical_approved_sha256 = $historicalCompiledPlanSha256
        }
        command = [ordered]@{
            coordinator_path = $coordinatorIdentity.path
            coordinator_bytes = $coordinatorIdentity.bytes
            coordinator_sha256 = $coordinatorIdentity.sha256
            worker_path = $workerIdentity.path
            worker_bytes = $workerIdentity.bytes
            worker_sha256 = $workerIdentity.sha256
            command_line = [Environment]::CommandLine
        }
        toolchain_pre = @($toolPre)
        toolchain_post = @($toolPost)
        python_runtime = $pythonRuntime
        git_version = $gitVersion
        windows_sdk_versions = [ordered]@{ include = $sdkIncludes; lib = $sdkLibs }
        build_inputs_pre = @($inputPre)
        build_inputs_post = @($inputPost)
        immutable_prior_reports_pre = @($priorPre)
        immutable_prior_reports_post = @($priorPost)
        attempt_3 = [ordered]@{ command_sha256 = 'b9146655853cd3822a1b6d49da04fef7bd8718d7dd936c86401dbf43ce9f42f2'; exit_code = 124; wall_seconds = 191.5; report_absent = $true; qualification = $false }
        worker_result = $workerResult
        worker_exit = $workerExit
        termination_reason = $terminationReason
        resource = [ordered]@{
            worker_deadline_seconds = $workerDeadlineSeconds
            worker_elapsed_seconds = $workerElapsedSeconds
            cleanup_margin_seconds = $cleanupMarginSeconds
            cleanup_elapsed_seconds = $cleanupElapsedSeconds
            job_memory_limit_bytes = $jobMemoryLimit
            combined_memory_limit_bytes = $combinedMemoryLimit
            peak_coordinator_working_set_bytes = $peakCoordinatorWorkingSet
            peak_observed_job_process_working_set_bytes = $peakObservedJobWorkingSet
            peak_observed_combined_working_set_bytes = $peakObservedCombinedWorkingSet
            job_peak_process_memory_bytes = [uint64]$jobSnapshot.PeakProcessMemoryBytes
            job_peak_memory_bytes = [uint64]$jobSnapshot.PeakJobMemoryBytes
            job_total_processes = $jobSnapshot.TotalProcesses
            job_active_processes = $jobSnapshot.ActiveProcesses
            job_terminated_processes = $jobSnapshot.TotalTerminatedProcesses
            job_total_user_time_100ns = $jobSnapshot.TotalUserTime100ns
            job_total_kernel_time_100ns = $jobSnapshot.TotalKernelTime100ns
            observed_job_pids = @($observedPids | Sort-Object)
            owned_processes_remaining = @()
        }
        finalization = [ordered]@{
            temp_cleanup_complete = $true
            task_environment_cleared = $true
            path_restored = $true
            source_identity_passed = $true
            input_identity_passed = $true
            prior_identity_passed = $true
            attempt3_report_absent = $true
        }
        wheel = [ordered]@{ name = [IO.Path]::GetFileName($bundleWheelIdentity.path); bytes = $bundleWheelIdentity.bytes; sha256 = $bundleWheelIdentity.sha256; validation = $workerResult.wheel.validation }
        build_log = [ordered]@{ name = 'build.log'; bytes = $buildLogIdentity.bytes; sha256 = $buildLogIdentity.sha256; streams = @('coordinator_events', 'worker_build_log', 'worker_stdout', 'worker_stderr') }
    }
    $reportPath = Join-Path $stagingBundle 'build_report.json'
    Write-CreateNewUtf8 -Path $reportPath -Text (($report | ConvertTo-Json -Depth 35) + [char]10)
    $reportIdentity = Get-FileIdentity -Path $reportPath

    $bundleEntries = @(
        [ordered]@{ path = 'anymesher-0.2.1-cp313-cp313-win_amd64.whl'; bytes = $bundleWheelIdentity.bytes; sha256 = $bundleWheelIdentity.sha256 },
        [ordered]@{ path = 'build.log'; bytes = $buildLogIdentity.bytes; sha256 = $buildLogIdentity.sha256 },
        [ordered]@{ path = 'build_report.json'; bytes = $reportIdentity.bytes; sha256 = $reportIdentity.sha256 }
    )
    $index = [ordered]@{
        schema_version = 1
        bundle = 'anymesher_windows_cp313_wheel_attempt_4_stage_a'
        address_algorithm = 'sha256(bundle_index.json)'
        entries = $bundleEntries
    }
    $indexPath = Join-Path $stagingBundle 'bundle_index.json'
    Write-CreateNewUtf8 -Path $indexPath -Text (($index | ConvertTo-Json -Depth 8) + [char]10)
    $indexIdentity = Get-FileIdentity -Path $indexPath
    $bundleIndexSha256 = $indexIdentity.sha256
    $finalLeaf = 'sha256-' + $bundleIndexSha256
    $finalBundle = [IO.Path]::GetFullPath((Join-Path $bundleParent $finalLeaf))
    if (-not [string]::Equals([IO.Path]::GetDirectoryName($finalBundle), $bundleParent, [StringComparison]::OrdinalIgnoreCase) -or
        [IO.Path]::GetFileName($finalBundle) -cne $finalLeaf -or (Test-Path -LiteralPath $finalBundle)) {
        throw "Final content-addressed bundle path is unsafe or non-fresh: $finalBundle"
    }
    $actualNames = @(Get-ChildItem -LiteralPath $stagingBundle -File -Force | ForEach-Object Name)
    [Array]::Sort($actualNames, [StringComparer]::Ordinal)
    $expectedNames = @('anymesher-0.2.1-cp313-cp313-win_amd64.whl', 'build.log', 'build_report.json', 'bundle_index.json')
    [Array]::Sort($expectedNames, [StringComparer]::Ordinal)
    if (($actualNames -join [char]10) -cne ($expectedNames -join [char]10) -or
        @(Get-ChildItem -LiteralPath $stagingBundle -Directory -Force).Count -ne 0) {
        throw 'Staging bundle member set mismatch'
    }
    foreach ($entry in $bundleEntries) {
        $identity = Get-FileIdentity -Path (Join-Path $stagingBundle $entry.path)
        if ($identity.bytes -ne $entry.bytes -or $identity.sha256 -cne $entry.sha256) { throw "Staged bundle entry mismatch: $($entry.path)" }
    }
    if ((Get-FileIdentity -Path $indexPath).sha256 -cne $bundleIndexSha256) { throw 'Staged bundle index changed before publication' }

    [IO.Directory]::Move($stagingBundle, $finalBundle)
    [void]$ownedTrees.Add([ordered]@{ path = $finalBundle; parent = $bundleParent; leaf = $finalLeaf })
    $stagingOwned = $ownedTrees | Where-Object { $_.path -ieq $stagingBundle } | Select-Object -First 1
    [void]$ownedTrees.Remove($stagingOwned)
    foreach ($entry in $bundleEntries) {
        $identity = Get-FileIdentity -Path (Join-Path $finalBundle $entry.path)
        if ($identity.bytes -ne $entry.bytes -or $identity.sha256 -cne $entry.sha256) { throw "Published bundle entry mismatch: $($entry.path)" }
    }
    if ((Get-FileIdentity -Path (Join-Path $finalBundle 'bundle_index.json')).sha256 -cne $bundleIndexSha256) {
        throw 'Published bundle index mismatch'
    }
    $published = $true
}
catch {
    $primaryError = $_.Exception.ToString()
}
finally {
    if ($null -ne $job) {
        try { $job.Terminate(125) } catch { [void]$finalizationErrors.Add($_.Exception.ToString()) }
        try { $job.Dispose() } catch { [void]$finalizationErrors.Add($_.Exception.ToString()) }
        $job = $null
    }
    if ($null -ne $gate) {
        try { $gate.Dispose() } catch { [void]$finalizationErrors.Add($_.Exception.ToString()) }
        $gate = $null
    }
    foreach ($path in @(
        [ordered]@{ path = if ($null -ne $workerLogPath) { $workerLogPath } else { $null }; target = 'worker_log' },
        [ordered]@{ path = if ($null -ne $workerStdoutPath) { $workerStdoutPath } else { $null }; target = 'worker_stdout' },
        [ordered]@{ path = if ($null -ne $workerStderrPath) { $workerStderrPath } else { $null }; target = 'worker_stderr' }
    )) {
        if ($null -ne $path.path -and (Test-Path -LiteralPath $path.path -PathType Leaf)) {
            try {
                $text = Get-Content -LiteralPath $path.path -Raw
                switch ($path.target) {
                    'worker_log' { $workerLogText = $text }
                    'worker_stdout' { $workerStdoutText = $text }
                    'worker_stderr' { $workerStderrText = $text }
                }
            }
            catch { [void]$finalizationErrors.Add($_.Exception.ToString()) }
        }
    }
    if (-not $published) {
        $ownedArray = @($ownedTrees)
        [Array]::Reverse($ownedArray)
        foreach ($owned in $ownedArray) {
            try { Remove-ExactOwnedTree -Path $owned.path -ExpectedParent $owned.parent -ExpectedLeaf $owned.leaf }
            catch { [void]$finalizationErrors.Add($_.Exception.ToString()) }
        }
    }
    $env:PATH = $originalPath
    foreach ($name in $taskEnvironmentNames) { Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue }
}

if ($null -ne $primaryError) {
    try {
        $repoPost = Get-RepositoryState -RepositoryPath $repository
        if ($null -eq $repoPre -or -not (Compare-RepositoryState -Before $repoPre -After $repoPost)) { throw 'Failure finalization source identity mismatch' }
        foreach ($prior in $priorReports) {
            if ((Get-FileIdentity -Path $prior.path).sha256 -cne $prior.expected_sha256) { throw "$($prior.label) changed during failed run" }
        }
        if (Test-Path -LiteralPath $attempt3Report) { throw 'Attempt-3 report appeared during failed run' }
        foreach ($input in $buildInputs) {
            $identity = Get-FileIdentity -Path $input.path
            if ($identity.bytes -ne $input.bytes -or $identity.sha256 -cne $input.sha256) { throw "$($input.name) changed during failed run" }
        }
        foreach ($anchor in $toolAnchors) { [void](Assert-ExpectedFile -Expected $anchor) }
        if ([Environment]::GetEnvironmentVariable('PATH', 'Process') -cne $originalPath -or
            @($taskEnvironmentNames | Where-Object { $null -ne [Environment]::GetEnvironmentVariable($_, 'Process') }).Count -ne 0) {
            throw 'Failure finalization environment mismatch'
        }
        $lingering = @($observedPids | Where-Object { $null -ne (Get-Process -Id $_ -ErrorAction SilentlyContinue) })
        if ($lingering.Count -ne 0) { throw "Failure finalization has owned processes: $($lingering -join ',')" }
    }
    catch {
        [void]$finalizationErrors.Add($_.Exception.ToString())
    }

    if ($eligibilityPassed -and $finalizationErrors.Count -eq 0) {
        $failureCompleted = [DateTimeOffset]::UtcNow
        $failureReport = [ordered]@{
            schema_version = 1
            qualification = 'windows_cpython313_immutable_wheel_build_attempt_4_stage_a_failure'
            success = $false
            expected_head = $expectedHead
            expected_tree = $expectedTree
            coordinator_sha256 = $expectedCoordinatorHash
            worker_sha256 = $expectedWorkerHash
            started_utc = $startedUtc.ToString('o')
            completed_utc = $failureCompleted.ToString('o')
            elapsed_seconds = ($failureCompleted - $startedUtc).TotalSeconds
            primary_error = $primaryError
            worker_exit = $workerExit
            termination_reason = $terminationReason
            worker_elapsed_seconds = $workerElapsedSeconds
            resource = [ordered]@{
                job_memory_limit_bytes = $jobMemoryLimit
                combined_memory_limit_bytes = $combinedMemoryLimit
                peak_coordinator_working_set_bytes = $peakCoordinatorWorkingSet
                peak_observed_job_process_working_set_bytes = $peakObservedJobWorkingSet
                peak_observed_combined_working_set_bytes = $peakObservedCombinedWorkingSet
                job_snapshot = $jobSnapshot
                observed_job_pids = @($observedPids | Sort-Object)
            }
            worker_log = $workerLogText
            worker_stdout = $workerStdoutText
            worker_stderr = $workerStderrText
            cleanup_complete = -not (Test-Path -LiteralPath $tempRoot) -and -not (Test-Path -LiteralPath $stagingBundle) -and ($null -eq $finalBundle -or -not (Test-Path -LiteralPath $finalBundle))
            process_complete = $true
            environment_complete = $true
            source_identity_passed = $true
            input_identity_passed = $true
            prior_identity_passed = $true
            bundle_published = $false
        }
        try {
            Write-AtomicFailureReport -PartialPath $failurePartial -FinalPath $failurePath -Payload $failureReport
        }
        catch {
            [void]$finalizationErrors.Add($_.Exception.ToString())
        }
    }
    $message = $primaryError
    if ($finalizationErrors.Count -ne 0) { $message += [Environment]::NewLine + ($finalizationErrors -join [Environment]::NewLine) }
    throw $message
}

if ($finalizationErrors.Count -ne 0) { throw ($finalizationErrors -join [Environment]::NewLine) }
if (-not $published) { throw 'Stage A ended without publishing or reporting a failure' }
Write-Output "STAGE_A_BUNDLE=$finalBundle"
Write-Output "STAGE_A_BUNDLE_INDEX_SHA256=$bundleIndexSha256"
