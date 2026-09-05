# Create a desktop shortcut for the app launcher.
# This file is kept pure ASCII on purpose: Chinese display names are built
# from Unicode code points to avoid any script-encoding dependency.
$ErrorActionPreference = 'Stop'

$root = $PSScriptRoot

# Launcher bat name and shortcut name (code-point built, see note above).
$batName = (-join ([char[]](0x542F,0x52A8,0x7CFB,0x7EDF))) + '.bat'
$lnkName = (-join ([char[]](0x76EE,0x6807,0x68C0,0x6D4B,0x7CFB,0x7EDF))) + '.lnk'

$target = Join-Path $root $batName
$icon   = Join-Path $root 'assets\app.ico'
$desktop = [Environment]::GetFolderPath('Desktop')
$lnkPath = Join-Path $desktop $lnkName

if (-not (Test-Path $target)) {
    Write-Host "[ERROR] launcher not found: $target"
    exit 1
}

$ws = New-Object -ComObject WScript.Shell
$sc = $ws.CreateShortcut($lnkPath)
$sc.TargetPath = $target
$sc.WorkingDirectory = $root
if (Test-Path $icon) { $sc.IconLocation = "$icon,0" }
$sc.Description = 'Object Detection System'
$sc.WindowStyle = 1
$sc.Save()

# Read back to verify the target was stored correctly.
$check = $ws.CreateShortcut($lnkPath)
if ($check.TargetPath -ne $target) {
    Write-Host "[ERROR] shortcut target mismatch: $($check.TargetPath)"
    exit 1
}

Write-Host "[OK] Desktop shortcut created: $lnkPath"
Write-Host "[OK] Target: $($check.TargetPath)"
