# Supervised, resumable driver for the tiled annual export. One PROCESS per year.
#
# WHY THIS EXISTS. The 2026-08-10 rebuild lost ~6 hours because three workers each died the
# second their Planetary Computer SAS token expired. PC signs asset URLs with a time-limited
# token and `planetary_computer` reuses it, so retrying inside the same process gets the SAME
# expired token and another HTTP 403 forever. Only a NEW PROCESS can obtain fresh credentials.
# stac_export.py detects that and exits 75; this script relaunches on it. Completed sub-tiles
# are identity-checked and reused, so a relaunch RESUMES rather than restarts.
#
# Never use threads for parallelism: dask's threaded scheduler deadlocks with rasterio/GDAL on
# Windows, hanging forever at ~0% CPU with no error.
#
# GIVE-UP RULE. Attempts are NOT a fixed count. Progress is measured in completed sub-tiles on
# disk, because a partial sub-tile is always discarded on relaunch. A fixed cap conflated
# "finished a tile" with "burned an hour making none": with 12 sub-tiles and a worst case of one
# tile per token window, a cap of 10 could never finish a year, and export_full refuses to
# mosaic a partial AOI — so the cap yielded nothing at all after hours of work. Instead: stop
# only after N CONSECUTIVE attempts that add zero sub-tiles, bounded by a generous absolute
# ceiling and a wall-clock deadline.
#
#   powershell -File scripts/run_export.ps1 -Years 2019,2022
#
[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)][int[]] $Years,
  [int] $MaxNoProgress   = 3,      # consecutive zero-sub-tile attempts before giving up
  [int] $MaxAttempts     = 40,     # absolute ceiling, guards against a hot loop
  [int] $DeadlineHours   = 24,     # wall-clock budget per year
  [int] $RetryPauseSec   = 30,     # pause after a generic non-zero exit
  [int] $CredsPauseSec   = 10,     # small pause after exit 75; a hot no-pause loop is worse
  # Watchdog for a genuinely hung attempt. This MUST exceed the time a healthy full year takes:
  # at 180m it killed 2019's first attempt while it was actively working (it had just gone from
  # 2 to 9 sub-tiles), throwing away hours. 12 sub-tiles x ~25-55 min is 5-11 h, so the real
  # stall detector is the consecutive-no-progress rule; this is only a backstop against the
  # documented silent dask/GDAL hang.
  [int] $AttemptTimeoutMin = 720,
  # Optional override of imagery.full_subtile_deg. Smaller work units complete between
  # transient read failures; a failed read restarts its whole sub-tile. 2023 needed 0.15.
  [double] $SubtileDeg = 0
)

$ErrorActionPreference = 'Continue'
$repo    = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$py      = Join-Path $repo '.venv\Scripts\python.exe'
$logDir  = Join-Path $repo 'data\logs'
$imgDir  = Join-Path $repo 'data\imagery'
New-Item -ItemType Directory -Force $logDir | Out-Null
Set-Location $repo

$EXPIRED = 75   # must equal EXPIRED_CREDS_EXIT in src/data/stac_export.py

# Per-slot log. Four slots appending to ONE file with Add-Content silently dropped lines
# (observed: 6 lines where 8 were written) because Add-Content takes an exclusive handle and
# its failure is non-terminating under $ErrorActionPreference='Continue'.
$supLog = Join-Path $logDir ("supervisor_" + ($Years -join '_') + ".log")

function Write-Sup([string]$msg) {
  $line = "[sup $(Get-Date -Format 'MM-dd HH:mm:ss')] $msg"
  Write-Output $line
  for ($i = 0; $i -lt 5; $i++) {
    try { Add-Content -Path $supLog -Value $line -Encoding utf8 -ErrorAction Stop; break }
    catch { Start-Sleep -Milliseconds 120 }
  }
}

function Count-SubTiles([int]$year) {
  # Completed sub-tiles for this year. This is the ONLY honest progress signal: log lines
  # accumulate across relaunches (a resumed run reprints "reusing" for every finished tile),
  # so counting log lines over-reports and can exceed 12/12.
  @(Get-ChildItem (Join-Path $imgDir "_sub_*_${year}_*.tif") -ErrorAction SilentlyContinue).Count
}

# Fail fast rather than misread a stale exit code: if the interpreter cannot be launched,
# PowerShell throws and $LASTEXITCODE keeps its PREVIOUS value — following a 75 that reads as
# another 75 and relaunches instantly forever.
if (-not (Test-Path $py)) { Write-Sup "FATAL: interpreter not found at $py"; exit 2 }

Write-Sup "start years=$($Years -join ',') pid=$PID maxNoProgress=$MaxNoProgress ceiling=$MaxAttempts deadline=${DeadlineHours}h timeout=${AttemptTimeoutMin}m"

foreach ($y in $Years) {
  $log      = Join-Path $logDir "export_$y.log"
  $err      = Join-Path $logDir "export_$y.err"
  $mosaic   = Join-Path $imgDir "catatumbo_${y}_annual_full.tif"
  $deadline = (Get-Date).AddHours($DeadlineHours)
  $noProg   = 0
  $attempt  = 0
  $done     = $false

  if (Test-Path $mosaic) { Write-Sup "year $y already complete (mosaic exists) - skipping"; continue }

  while (-not $done -and $noProg -lt $MaxNoProgress -and $attempt -lt $MaxAttempts) {
    # THE PRODUCT, not the exit code, decides completion. Checked at the top of every
    # iteration: a mosaic on disk means the year is finished even if the process reported a
    # bad/empty exit code or was killed a moment after writing it. Omitting this check inside
    # the loop is what made three finished years start over.
    if (Test-Path $mosaic) {
      Write-Sup "year $y COMPLETE (mosaic present at loop top - not re-running)"
      $done = $true
      break
    }
    if ((Get-Date) -gt $deadline) { Write-Sup "year $y DEADLINE (${DeadlineHours}h) reached"; break }
    $attempt++
    $before = Count-SubTiles $y
    Write-Sup "year $y attempt $attempt (sub-tiles on disk: $before/12, consecutive no-progress: $noProg)"

    # Start-Process gives a killable handle for the watchdog. It cannot append, so each attempt
    # writes to its own temp file which is then appended - keeping the accumulated history that
    # makes a resumed run legible.
    $so = [IO.Path]::GetTempFileName()
    $se = [IO.Path]::GetTempFileName()
    $code = $null
    try {
      $pyArgs = @('-u', '-m', 'src.data.stac_export', '--full', '--year', "$y")
      if ($SubtileDeg -gt 0) { $pyArgs += @('--subtile-deg', "$SubtileDeg") }
      $p = Start-Process -FilePath $py -ArgumentList $pyArgs `
        -WorkingDirectory $repo -RedirectStandardOutput $so -RedirectStandardError $se `
        -PassThru -NoNewWindow -ErrorAction Stop
      # REQUIRED, and verified in isolation: `Start-Process -PassThru` does not retain the
      # native process handle, so $p.ExitCode comes back EMPTY after exit — even after
      # WaitForExit(). Reading .Handle once, while the process is still alive, caches it and
      # makes ExitCode readable. Measured: without this, ExitCode='' ; with it, ExitCode=75.
      # The empty code is what made a finished year look like a failure and restart.
      $null = $p.Handle
      if (-not $p.WaitForExit($AttemptTimeoutMin * 60 * 1000)) {
        Write-Sup "year $y attempt $attempt EXCEEDED ${AttemptTimeoutMin}m - killing (suspected hang)"
        try { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } catch {}
        Start-Sleep -Seconds 3
        $code = 124   # conventional timeout code
      } else {
        # The no-argument WaitForExit() is REQUIRED here. After the timed overload returns
        # true, ExitCode can still be unpopulated, and $code then came out EMPTY — which is
        # not 0, so a year whose mosaic had just been written successfully was treated as a
        # failure and re-downloaded from scratch. That happened to 2019, 2022 and 2024.
        $p.WaitForExit()
        $code = $p.ExitCode
        if ($null -eq $code) { $code = -1 }
      }
    } catch {
      Write-Sup "year $y attempt $attempt could not launch: $($_.Exception.Message)"
      $code = 2
    }
    Get-Content $so -ErrorAction SilentlyContinue | Add-Content -Path $log -Encoding utf8
    Get-Content $se -ErrorAction SilentlyContinue | Add-Content -Path $err -Encoding utf8
    Remove-Item $so, $se -Force -ErrorAction SilentlyContinue

    $after = Count-SubTiles $y
    if ($after -gt $before) { $noProg = 0 } else { $noProg++ }

    if (Test-Path $mosaic) {
      # Deliberately does NOT require $code -eq 0: the mosaic existing is proof of success,
      # and sub-tiles are deleted right after mosaicking so `$after` legitimately drops to 0.
      Write-Sup "year $y COMPLETE (mosaic written; exit=$code)"
      $done = $true
      break
    }
    if ($code -eq $EXPIRED) {
      Write-Sup "year $y credentials expired (exit 75), progress $before->$after - relaunching for fresh credentials"
      Start-Sleep -Seconds $CredsPauseSec
      continue
    }
    Write-Sup "year $y exited $code, progress $before->$after - pausing ${RetryPauseSec}s"
    Start-Sleep -Seconds $RetryPauseSec
  }

  if (-not $done) {
    Write-Sup "year $y GAVE UP after $attempt attempts ($(Count-SubTiles $y)/12 sub-tiles, $noProg consecutive no-progress) - see $err"
  }
}

Write-Sup "slot finished years=$($Years -join ',')"
