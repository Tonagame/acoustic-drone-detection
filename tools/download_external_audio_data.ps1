param(
    [switch]$FSD50KAudio,
    [switch]$FSD50KSmall,
    [switch]$MADMetadata
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$FsdDir = Join-Path $Root "data\external\FSD50K"
$MadDir = Join-Path $Root "data\external\MAD"
New-Item -ItemType Directory -Force -Path $FsdDir, $MadDir | Out-Null

function Download-File($Url, $OutPath) {
    if (Test-Path $OutPath) {
        $existing = (Get-Item $OutPath).Length
        if ($existing -gt 0) {
            Write-Host "Exists: $OutPath ($existing bytes)"
            return
        }
    }
    Write-Host "Downloading: $Url"
    curl.exe --ssl-no-revoke -L --fail --continue-at - --output $OutPath $Url
}

$FsdFiles = @(
    @("FSD50K.doc.zip", "https://zenodo.org/api/records/4060432/files/FSD50K.doc.zip/content"),
    @("FSD50K.metadata.zip", "https://zenodo.org/api/records/4060432/files/FSD50K.metadata.zip/content"),
    @("FSD50K.ground_truth.zip", "https://zenodo.org/api/records/4060432/files/FSD50K.ground_truth.zip/content")
)

if ($FSD50KAudio) {
    $FsdFiles += @(
        @("FSD50K.eval_audio.z01", "https://zenodo.org/api/records/4060432/files/FSD50K.eval_audio.z01/content"),
        @("FSD50K.eval_audio.zip", "https://zenodo.org/api/records/4060432/files/FSD50K.eval_audio.zip/content"),
        @("FSD50K.dev_audio.z01", "https://zenodo.org/api/records/4060432/files/FSD50K.dev_audio.z01/content"),
        @("FSD50K.dev_audio.z02", "https://zenodo.org/api/records/4060432/files/FSD50K.dev_audio.z02/content"),
        @("FSD50K.dev_audio.z03", "https://zenodo.org/api/records/4060432/files/FSD50K.dev_audio.z03/content"),
        @("FSD50K.dev_audio.z04", "https://zenodo.org/api/records/4060432/files/FSD50K.dev_audio.z04/content"),
        @("FSD50K.dev_audio.z05", "https://zenodo.org/api/records/4060432/files/FSD50K.dev_audio.z05/content"),
        @("FSD50K.dev_audio.zip", "https://zenodo.org/api/records/4060432/files/FSD50K.dev_audio.zip/content")
    )
}

foreach ($file in $FsdFiles) {
    Download-File $file[1] (Join-Path $FsdDir $file[0])
}

if ($MADMetadata) {
    Download-File "https://raw.githubusercontent.com/kaen2891/military_audio_dataset/main/README.md" (Join-Path $MadDir "README.md")
    Download-File "https://raw.githubusercontent.com/kaen2891/military_audio_dataset/main/mad_dataset_annotation.csv" (Join-Path $MadDir "mad_dataset_annotation.csv")
    Download-File "https://raw.githubusercontent.com/kaen2891/military_audio_dataset/main/youtube_audio_download.py" (Join-Path $MadDir "youtube_audio_download.py")
}

Write-Host ""
Write-Host "Done. FSD50K: $FsdDir"
Write-Host "Done. MAD metadata: $MadDir"
Write-Host "MAD audio requires Kaggle access for junewookim/mad-dataset-military-audio-dataset."
