# Preenche o dataset de treino do Manga Index com os capítulos que ele já traduziu, a partir do cache de
# tradução: nenhum modelo é chamado. Para depois de -Pages páginas; rodar de novo continua de onde parou.
#   powershell -File scripts\replay_manga_index.ps1 [-Pages 5000] [-MangaIndex "D:\Dev\Manga Index"]
param(
    [int]$Pages = 5000,
    [string]$MangaIndex = $(if ($env:MANGA_INDEX) { $env:MANGA_INDEX } else { "D:\Dev\Manga Index" })
)
$ErrorActionPreference = "Stop"
$appData = Join-Path $env:APPDATA "Manga Index"
# Pasta de dados própria, para rodar ao lado do app: as configurações sem as chaves de API e o cache real.
$userData = Join-Path $PSScriptRoot "..\data\mi-replay-userdata"
New-Item -ItemType Directory -Force $userData | Out-Null
$settings = Get-Content (Join-Path $appData "settings.json") -Raw -Encoding UTF8 | ConvertFrom-Json
$settings.PSObject.Properties.Remove("keys")
# Sem BOM: o app lê um arquivo com BOM como se não houvesse configurações.
[IO.File]::WriteAllText((Join-Path (Resolve-Path $userData) "settings.json"), ($settings | ConvertTo-Json -Depth 20), (New-Object Text.UTF8Encoding $false))
$cache = Join-Path $userData "translation-cache"
if (-not (Test-Path $cache)) { New-Item -ItemType Junction -Path $cache -Target (Join-Path $appData "translation-cache") | Out-Null }

$env:MI_USER_DATA = (Resolve-Path $userData).Path
$env:MI_CACHE_ONLY = "1"
$env:MI_REPLAY_LIBRARY = Join-Path $appData "library"
$env:MI_DATASET_DIR = Join-Path $appData "training-dataset"
$log = Join-Path (Resolve-Path $userData) "replay.log"
$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$stats = "import sys; sys.path.insert(0, r'$PSScriptRoot'); from dataset_stats import compute; t = compute()['timeline']; print(t[-1]['total'] if t else 0)"
Set-Location $MangaIndex
# Páginas fora do cache saem no stderr, que o PowerShell trataria como erro fatal.
$ErrorActionPreference = "Continue"
# Build próprio em vez de electron-vite dev, que reinicia o app a cada arquivo salvo no Manga Index.
# Fica dentro de node_modules para achar os pacotes do app e ficar fora do git.
$build = Join-Path $MangaIndex "node_modules\.mi-replay-build"
npx electron-vite build --outDir $build 2>&1 | ForEach-Object { "$_" } | Out-File $log -Append
$electron = Join-Path $MangaIndex "node_modules\electron\dist\electron.exe"
# Se uma execução terminar antes da hora, recomeça pedindo só as páginas que faltam, até não somar nada.
while ($true) {
    $before = [int](& $python -c $stats)
    if ($before -ge $Pages) { break }
    $env:MI_REPLAY_PAGES = "$($Pages - $before)"
    & $electron (Join-Path $build "main\index.js") 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append | Select-String -Pattern '\[eval\]|FAILED' | ForEach-Object { $_.Line }
    $after = [int](& $python -c $stats)
    Write-Host "Dataset: $after páginas."
    if ($after -le $before) { break }
}
