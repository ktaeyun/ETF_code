$env:PYTHONIOENCODING='utf-8'; $env:MPLBACKEND='Agg'
Set-Location 'C:\Code\ETF_code'
"[{0}] scenario 시작" -f (Get-Date -Format 'MM-dd HH:mm:ss') | Out-File 'C:\Code\ETF_code\run_fix9.log' -Append -Encoding utf8
& 'C:\Code\ETF_code\etf\Scripts\python.exe' -u simulator\scenario_main.py --scenarios S01,S02,S03,S04,S05,S06,S07,S08,S09 --force-refit *> 'C:\Code\ETF_code\run_scenario_fix9.log'
"[{0}] scenario 종료 exit=$LASTEXITCODE" -f (Get-Date -Format 'MM-dd HH:mm:ss') | Out-File 'C:\Code\ETF_code\run_fix9.log' -Append -Encoding utf8
& 'C:\Code\ETF_code\etf\Scripts\python.exe' -u simulator\results_main.py --M 100 --scenarios S01,S02,S03,S04,S05,S06,S07,S08,S09 *> 'C:\Code\ETF_code\run_results_fix9.log'
"[{0}] results 종료 exit=$LASTEXITCODE" -f (Get-Date -Format 'MM-dd HH:mm:ss') | Out-File 'C:\Code\ETF_code\run_fix9.log' -Append -Encoding utf8
"[{0}] 전체 완료" -f (Get-Date -Format 'MM-dd HH:mm:ss') | Out-File 'C:\Code\ETF_code\run_fix9.log' -Append -Encoding utf8
